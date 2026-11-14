import cv2
import numpy as np
import threading
import logging
from collections import deque

class Detector:
    def __init__(self, model, path, silence_cv2_warnings=True):
        self.model = model
        self.path = path

        self.stream = cv2.VideoCapture(path)
        self.total_frames_read = 0
        self.total_frames_dropped = 0
        self.total_frames_processed = 0
    
        self.webcam_fps = 60
        self.buffer_duration_seconds = 0.1
        self.max_buffered_frames = int(self.webcam_fps*self.buffer_duration_seconds)
        self.webcam_frames = deque(maxlen=self.max_buffered_frames)
        self.webcam_frames_cv = threading.Condition()

        self.is_running = True
        self.logger = logging.getLogger("detector")
        self.capture_thread = threading.Thread(target=self._start_capture_thread)
        self.processing_thread = threading.Thread(target=self._start_processing_thread)
    
    def _start_capture_thread(self):
        while self.is_running:
            ok, frame = self.stream.read()
            if not ok:
                self.logger.error(f"Failed to capture frame from webcam")
                break
            self.total_frames_read += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32) / 255.0

            with self.webcam_frames_cv:
                if len(self.webcam_frames) == self.webcam_frames.maxlen:
                    self.total_frames_dropped += 1
                self.webcam_frames.append(frame)
                self.webcam_frames_cv.notify()

        self.is_running = False
        with self.webcam_frames_cv:
            self.webcam_frames_cv.notify()
        self.logger.info(f"Closed webcam capture thread gracefully total_frames_read={self.total_frames_read}, total_frames_dropped={self.total_frames_dropped}")


    def _start_processing_thread(self):
        while self.is_running:
            with self.webcam_frames_cv:
                self.webcam_frames_cv.wait()
                if not self.is_running:
                    break
                if len(self.webcam_frames) == 0:
                    continue
                frame = self.webcam_frames.popleft()

            frame = cv2.resize(frame, (self.model.input_width, self.model.input_height))  # H,W
            batch = frame[np.newaxis, ...]
            batch = self.model.preprocess_batch(batch)
            probs = self.model.infer_batch_probabilities(batch)
            print(probs)
            self.total_frames_processed += 1
        self.logger.info(f"Stopped processing thread after total_frames_processed={self.total_frames_processed}")

    def start_threads(self):
        self.capture_thread.start()
        self.processing_thread.start()
    
    def join_threads(self, force=False):
        if force:
            self.is_running = False
        self.capture_thread.join()
        self.processing_thread.join()