import cv2
import numpy as np
import logging
from collections.abc import Iterable
import traceback
import asyncio

from bounding_boxes import BoundingBoxSplitter, BoundingBox


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
        self.logger = logging.getLogger("detector")
        self.model = model # ort.InferenceSession
        self.bbox_splitter = bbox_splitter
        self.allow_drop_frames = allow_drop_frames

        self.video_stream = iter(video_stream)
        self.total_frames_read = 0
        self.total_frames_dropped = 0
        self.total_frames_processed = 0

        self.webcam_frames = asyncio.Queue(maxsize=total_buffer_frames) # item = np.array(shape=[H,W], dtype=np.float32)
        self.results = asyncio.Queue(maxsize=total_buffer_results) # item = DetectorResult

        self.is_capturing = False
        self.is_processing = False
        self.capture_task = None
        self.processing_task = None
    
    async def _start_capture_task(self):
        try:
            self.is_capturing = True
            while True:
                try:
                    # https://peps.python.org/pep-0479/
                    # PEP479 - StopIteration exception not supported in Futures
                    # frame = await asyncio.to_thread(lambda: next(self.video_stream))
                    frame = await asyncio.to_thread(lambda: next(self.video_stream, None))
                    if frame is None:
                        self.logger.info(f"Video stream ended gracefully")
                        break
                except Exception as ex:
                    error_trace = traceback.format_exc()
                    self.logger.error(f"Video stream ended abruptly with error: {error_trace}")
                    break
            
                assert len(frame.shape) == 2, f"Expected frame to be single channel with shape=[height,width] but got shape=[{','.join(frame.shape)}]"
                assert frame.dtype == np.float32, f"Expected frame to be floating point grayscale but got dtype={frame.dtype}"
                self.total_frames_read += 1

                try:
                    if self.webcam_frames.full() and self.allow_drop_frames:
                        self.webcam_frames.get_nowait()
                        self.webcam_frames.put_nowait(frame)
                        self.total_frames_dropped += 1
                    else:
                        await self.webcam_frames.put(frame)
                except asyncio.QueueShutDown:
                    self.logger.info("Capture task exiting since webcam frames queue was shutdown")
                    break
        except Exception as ex:
            error_trace = traceback.format_exc()
            self.logger.error(f"Capture exited with unhandled error: {error_trace}")
            self.stop(force=True)
        finally:
            self.is_capturing = False
            self.logger.info(f"Stopped webcam capture task after total_frames_read={self.total_frames_read}, total_frames_dropped={self.total_frames_dropped}")
            self.webcam_frames.shutdown(immediate=False) # trickle left overs

    async def _start_processing_task(self):
        try:
            self.is_processing = True
            while True:
                try:
                    frame = await self.webcam_frames.get()
                except asyncio.QueueShutDown:
                    self.logger.info("Processing task exiting since webcam frames queue was shutdown")
                    break
                
                result = await asyncio.to_thread(lambda: DetectorResult.from_model_and_splitter(frame, self.model, self.bbox_splitter))
                self.total_frames_processed += 1

                try:
                    await self.results.put(result)
                except asyncio.QueueShutDown:
                    self.logger.info("Processing task exiting since results queue was shutdown")
                    break
        except Exception as ex:
            error_trace = traceback.format_exc()
            self.logger.error(f"Processing task exited with unhandled error: {error_trace}")
            self.stop(force=True)
        finally:
            self.is_processing = False
            self.logger.info(f"Stopped frame processing task after total_frames_processed={self.total_frames_processed}")
            self.results.shutdown(immediate=False) # trickle left overs

    async def get_results(self):
        while True:
            try:
                result = await self.results.get()
                yield result
            except asyncio.QueueShutDown:
                break
    
    def stop(self, force=False):
        self.webcam_frames.shutdown(immediate=force)
        self.results.shutdown(immediate=force)
    
    async def wait_finished(self):
        assert self.capture_task != None, "Capture task not started yet"
        assert self.processing_task != None, "Processing task not started yet"
        await asyncio.gather(self.capture_task, self.processing_task)

    async def __aenter__(self):
        assert self.capture_task == None, "Already running capture task"
        assert self.processing_task == None, "Already running processing task"
        self.capture_task = asyncio.create_task(self._start_capture_task())
        self.processing_task = asyncio.create_task(self._start_processing_task())

    async def __aexit__(self, exception_type, exception_value, exception_traceback): 
        assert self.capture_task != None, "Failed to get running capture task"
        assert self.processing_task != None, "Failed to get running processing task"
        self.stop(force=True)
        await self.capture_task
        await self.processing_task