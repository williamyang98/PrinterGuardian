import cv2
import numpy as np
import threading
import logging
from collections import deque
from collections.abc import Iterable
from bounding_boxes import BoundingBoxSplitter, BoundingBox
from functools import wraps
import traceback

def handle_thread_error(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            with self.cv_thread_count:
                self.total_threads_running += 1
            return func(self, *args, **kwargs)
        except Exception as ex:
            error_trace = traceback.format_exc()
            self.logger.error(f"Thread {func.__name__} exited with unhandled error: {error_trace}")
            self.force_stop()
        finally:
            with self.cv_thread_count:
                self.total_threads_running -= 1
                self.cv_thread_count.notify_all()
    return wrapper

class DetectorResult:
    def __init__(self, frame: np.typing.ArrayLike, bboxes: list[BoundingBox], probs: np.typing.ArrayLike):
        self.frame = frame
        self.bboxes = bboxes
        self.probs = probs
    
    def draw_bboxes(self, threshold=0.2):
        # floating point grayscale to uint8 BGR
        frame = (self.frame * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        COLOUR_FAILURE = (0, 0, 180)
        COLOUR_SUCCESS = (0, 160, 0)
        THICKNESS_FAILURE = 5
        THICKNESS_SUCCESS = 1
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SIZE = 0.5 
        FONT_WEIGHT_FAILURE = 2
        FONT_WEIGHT_SUCCESS = 1
        FONT_PADDING = 6
        FONT_BACKGROUND_COLOUR = (255,255,255,100)

        # bboxes
        for bbox, probs in zip(self.bboxes, self.probs):
            is_failure = probs < threshold
            thickness = THICKNESS_FAILURE if is_failure else THICKNESS_SUCCESS
            colour = COLOUR_FAILURE if is_failure else COLOUR_SUCCESS
            cv2.rectangle(frame, (bbox.left, bbox.bottom), (bbox.right, bbox.top), colour, thickness)
        # labels for text
        for bbox, probs in zip(self.bboxes, self.probs):
            is_failure = probs < threshold
            font_thickness = FONT_WEIGHT_FAILURE if is_failure else FONT_WEIGHT_SUCCESS
            colour = COLOUR_FAILURE if is_failure else COLOUR_SUCCESS
            text = f"{probs:.3f}"
            (text_width, text_height), _baseline = cv2.getTextSize(text, FONT, FONT_SIZE, font_thickness)
            position = (bbox.left + FONT_PADDING, bbox.bottom + text_height + FONT_PADDING)
            cv2.rectangle(
                frame,
                (bbox.left, bbox.bottom),
                (bbox.left+2*FONT_PADDING+text_width, bbox.bottom+2*FONT_PADDING+text_height),
                FONT_BACKGROUND_COLOUR, 
                -1, # solid fill
            )
            cv2.putText(frame, text, position, FONT, FONT_SIZE, colour, thickness=font_thickness)
        return frame
    
    @staticmethod
    def from_model_and_splitter(frame: np.typing.ArrayLike, model, bbox_splitter: BoundingBoxSplitter):
        assert len(frame.shape) == 2, f"Expected single channel grayscale image with [height,width] but got shape=[{','.join(frame.shape)}]"
        assert frame.dtype == np.float32, f"Expected float32 image but got dtype={frame.dtype}"

        bboxes = list(bbox_splitter.get_child_boxes())
        images = list(bbox_splitter.split_image(frame, height_axis=0, width_axis=1))
        total_images = len(images)
        batch = np.zeros((total_images, model.input_height, model.input_width), dtype=np.float32)
        for i, image in enumerate(images):
            image = cv2.resize(image, (model.input_width, model.input_height))  # H,W
            batch[i,:,:] = image
        batch = model.preprocess_batch(batch)
        probs = model.infer_batch_probabilities(batch)
        result = DetectorResult(frame, bboxes, probs)
        return result

class Detector:
    def __init__(self, model, video_stream: Iterable[np.typing.ArrayLike], bbox_splitter: BoundingBoxSplitter, allow_drop_frames=False, total_buffer_frames=10, total_buffer_results=10):
        self.model = model # ort.InferenceSession
        self.bbox_splitter = bbox_splitter
        self.allow_drop_frames = allow_drop_frames

        self.video_stream = iter(video_stream)
        self.total_frames_read = 0
        self.total_frames_dropped = 0
        self.total_frames_processed = 0
    
        self.webcam_frames = deque(maxlen=total_buffer_frames) # item = np.array(shape=[H,W], dtype=np.float32)
        self.results = deque(maxlen=total_buffer_results) # item = DetectorResult

        self.lock_frames_deque = threading.RLock()
        self.is_frames_deque_open = False
        self.is_capturing = False
        self.is_processing = False
        self.cv_frames_available = threading.Condition(lock=self.lock_frames_deque)
        self.cv_deque_not_full = threading.Condition(lock=self.lock_frames_deque)
        self.cv_results_available = threading.Condition()

        self.total_threads_running = 0
        self.cv_thread_count = threading.Condition()

        self.logger = logging.getLogger("detector")
        self.capture_thread = None
        self.processing_thread = None
    
    @property
    def is_running(self):
        return self.total_threads_running > 0

    @handle_thread_error 
    def _start_capture_thread(self):
        while True:
            try:
                frame = next(self.video_stream)
            except StopIteration:
                self.logger.info(f"Video stream ended gracefully")
                break
            except Exception as ex:
                error_trace = traceback.format_exc()
                self.logger.error(f"Video stream ended abruptly with error: {error_trace}")
                break
            
            assert len(frame.shape) == 2, f"Expected frame to be single channel with shape=[height,width] but got shape=[{','.join(frame.shape)}]"
            assert frame.dtype == np.float32, f"Expected frame to be floating point grayscale but got dtype={frame.dtype}"
            self.total_frames_read += 1

            with self.cv_deque_not_full:
                while len(self.webcam_frames) == self.webcam_frames.maxlen and self.is_frames_deque_open and self.is_processing:
                    if self.allow_drop_frames:
                        self.total_frames_dropped += 1
                        break
                    else:
                        self.cv_deque_not_full.wait()
                with self.cv_frames_available:
                    self.webcam_frames.append(frame)
                    self.cv_frames_available.notify()
                if not self.is_frames_deque_open:
                    break
                if not self.is_processing:
                    break

        self.logger.info(f"Closed webcam capture thread gracefully total_frames_read={self.total_frames_read}, total_frames_dropped={self.total_frames_dropped}")
        with self.cv_frames_available:
            self.is_capturing = False
            self.cv_frames_available.notify()

    @handle_thread_error 
    def _start_processing_thread(self):
        while True:
            with self.cv_frames_available:
                while len(self.webcam_frames) == 0 and self.is_frames_deque_open and self.is_capturing:
                    self.cv_frames_available.wait()
                if not self.is_frames_deque_open:
                    break
                if not self.is_capturing and len(self.webcam_frames) == 0:
                    break
                if len(self.webcam_frames) == 0:
                    continue
                with self.cv_deque_not_full:
                    frame = self.webcam_frames.popleft()
                    self.cv_deque_not_full.notify()

            result = DetectorResult.from_model_and_splitter(frame, self.model, self.bbox_splitter)
            self.total_frames_processed += 1

            with self.cv_results_available:
                self.results.append(result)
                self.cv_results_available.notify()
        self.logger.info(f"Stopped frame processing thread after total_frames_processed={self.total_frames_processed}")

        with (self.cv_deque_not_full, self.cv_results_available):
            self.is_processing = False
            self.cv_deque_not_full.notify()
            self.cv_results_available.notify()

    def get_results_blocking(self):
        while True:
            with self.cv_results_available:
                while len(self.results) == 0 and self.is_processing:
                    self.cv_results_available.wait()
                if not self.is_processing and len(self.results) == 0:
                    break
                if len(self.results) == 0:
                    continue
                with self.cv_results_available:
                    result = self.results.popleft()
                yield result
    
    def force_stop(self):
        self.is_frames_deque_open = False
        self.is_capturing = False
        self.is_processing = False
        with self.cv_frames_available:
            self.cv_frames_available.notify()
        with self.cv_deque_not_full:
            self.cv_deque_not_full.notify()
        with self.cv_results_available:
            self.cv_results_available.notify()
    
    def wait_finished(self):
        with self.cv_thread_count:
            while self.total_threads_running > 0:
                self.cv_thread_count.wait()

    def __enter__(self):
        assert not self.is_frames_deque_open, "Tried to open an active detector again"

        self.is_frames_deque_open = True
        self.capture_thread = threading.Thread(target=self._start_capture_thread)
        self.processing_thread = threading.Thread(target=self._start_processing_thread)
        self.is_capturing = True
        self.is_processing = True
        self.capture_thread.start()
        self.processing_thread.start()

    def __exit__(self, exception_type, exception_value, exception_traceback): 
        self.force_stop()
        self.capture_thread.join()
        self.processing_thread.join()