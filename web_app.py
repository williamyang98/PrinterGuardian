from fastapi import Response
from nicegui import app, ui
from detector import Detector, DetectorResult
from bounding_boxes import BoundingBoxSplitter
import asyncio
import numpy as np
import cv2
from collections.abc import Iterable
import logging
import time
import bisect
from datetime import datetime

SNAPSHOT_CV2_EXTENSION = ".jpg"
SNAPSHOT_WEB_MEDIA_TYPE = "image/jpeg"

class Snapshot:
    def __init__(self, result: DetectorResult, frame_number: int, probs_threshold: float):
        self.result = result
        self.frame_number = frame_number
        self.probs_threshold = probs_threshold
        self.unix_time = time.time()

        # frame = (result.frame * 255.0).astype(np.uint8)
        frame = result.draw_bboxes(threshold=self.probs_threshold)
        success, image_data = cv2.imencode(SNAPSHOT_CV2_EXTENSION, frame)
        if not success:
            raise Exception(f"Failed to encode frame as {SNAPSHOT_CV2_EXTENSION}")
        self.image_data = image_data.tobytes()

class WebApp:
    def __init__(self, model, bbox_splitter: BoundingBoxSplitter, video_path: str, ui_path="/"):
        self.model = model
        self.bbox_splitter = bbox_splitter
        self.video_path = video_path
        self.ui_path = ui_path
        self.logger = logging.getLogger("web_app")

        self.probs_threshold = 0.2
        self.skip_frame_stride = 0
        self.latest_snapshot = None
        self.failure_snapshots = []
        self.is_ui_ready = False

        video_stream = self.get_video_stream()
        self.detector = Detector(self.model, video_stream, self.bbox_splitter)

        @ui.page(self.ui_path)
        def page():
            self._build_ui()
            self._build_routes()

        app.on_startup(self._start)
        app.on_shutdown(self._shutdown)
    
    def _start(self):
        self.logger.info("Starting web app")
        self.read_results_task = asyncio.create_task(asyncio.to_thread(self._read_results))

    async def _shutdown(self):
        self.logger.info("Shutting down web app...")
        self.detector.force_stop()
        await self.read_results_task
        await asyncio.to_thread(lambda: self.detector.wait_finished())
        self.logger.info("Finished shutting down web app")
        
    
    def get_video_stream(self):
        total_skipped_frames = 0
        capture = cv2.VideoCapture(self.video_path)
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if total_skipped_frames < self.skip_frame_stride:
                total_skipped_frames += 1
                continue

            total_skipped_frames = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32) / 255.0
            yield frame

    def _build_ui(self):
        root = self.ui_path.rstrip("/")

        def create_config_panel():
            with ui.row():
                field = ui.number(
                    label="Failure probability detection threshold",
                    min=0, max=1, step=0.01,
                    format="%.3f",
                )
                field.bind_value(self, "probs_threshold")
                field.classes("min-w-60")

                field = ui.number(
                    label="Number of frames to skip",
                    min=0, step=1,
                    format="%d",
                )
                field.bind_value(self, "skip_frame_stride")
                field.classes("min-w-10")

                field = ui.checkbox("Allow dropping frames")
                field.bind_value(self.detector, "allow_drop_frames")
                field.classes("min-w-10")

        def create_table_frame_stats():
            columns = [
                { "name": "name", "label": "Name", "field": "name", "required": True },
                { "name": "value", "label": "Value", "field": "value", "required": True },
            ]
            def create_rows():
                return [
                    { "name": "frames_read", "value": self.detector.total_frames_read },
                    { "name": "frames_dropped", "value": self.detector.total_frames_dropped },
                    { "name": "frames_processed", "value": self.detector.total_frames_processed },
                    { "name": "threads_running", "value": self.detector.total_threads_running },
                ]
            rows = create_rows()
            table = ui.table(columns=columns, rows=rows, row_key="name")
            table.add_slot("header", "")
            with table.add_slot("body-cell-name"):
                with table.cell("name"):
                    ui.label().props(''':textContent=props.value ''').classes("font-medium text-left")

            def refresh_table():
                table.rows = create_rows()
                table.update()
            
            return refresh_table

        def create_table_failure_snapshots():
            columns = [
                { "name": "frame", "label": "Frame", "field": "frame", "required": True, "sortable": True },
                { "name": "probs", "label": "Probability", "field": "probs", "required": True, "sortable": True },
                { "name": "time", "label": "Time", "field": "time", "required": True, "sortable": True },
            ]
            rows = []

            def create_row_from_snapshot(snapshot):
                time = datetime.fromtimestamp(snapshot.unix_time)
                row = {
                    "frame": snapshot.frame_number,
                    "probs": np.min(snapshot.result.probs),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                return row

            for snapshot in self.failure_snapshots:
                row = create_row_from_snapshot(snapshot)
                rows.append(row)

            pagination = 5
            table = ui.table(
                columns=columns, rows=rows, row_key="frame",
                pagination=pagination,
                selection="single",
                # on_select=select_failure_snapshot, # Handle this manually with on_row_click
            )
            table = table.classes("w-fit")
            table.add_slot("header-selection", "")
            table.add_slot("body-selection", "")
            with table.add_slot("body-cell-probs"):
                with table.cell("probs"):
                    ui.label().props(''':textContent="props.value.toFixed(3)"''')

            def on_row_click(click_event):
                image = self.ui_image_failure_snapshot
                _event, row, _index = click_event.args
                # deselect
                if table.selected and table.selected[0]["frame"] == row["frame"]:
                    table.selected.clear()
                    image.set_source(None)
                else:
                    table.selected = [row]
                    image.set_source(f"{root}/failure_snapshots/{row["frame"]}")
            table.on("row-click", on_row_click)

            def refresh_table():
                total_added = len(self.failure_snapshots) - len(table.rows)
                if total_added > 0:
                    new_snapshots = self.failure_snapshots[-total_added:]
                    new_rows = []
                    for snapshot in new_snapshots:
                        row = create_row_from_snapshot(snapshot)
                        new_rows.append(row)
                    table.add_rows(new_rows)

            return refresh_table
        
        def create_stream_status_badge():
            def get_text_and_colour():
                is_live = self.detector.is_capturing
                text = "Live" if is_live else "Offline"
                colour = "green" if is_live else "red"
                return (text, colour)
            
            text, colour = get_text_and_colour()
            badge = ui.badge(text, color=colour).props("floating").classes("m-2")

            def refresh():
                text, colour = get_text_and_colour()
                badge.set_text(text)
                badge.set_background_color(colour)
            return refresh
        
        def create_failure_snapshots_badge():
            total_failures = len(self.failure_snapshots)
            badge = ui.badge(f"{total_failures}", color="red").classes("ml-2")
            def refresh():
                total_failures = len(self.failure_snapshots)
                badge.set_text(f"{total_failures}")
            return refresh

        # layout
        create_config_panel()
        with ui.tabs().props("inline-label") as tabs:
            tab_latest_snapshot = ui.tab("Latest Preview")
            with ui.tab("Failures") as tab_failures_snapshot:
                total_failures = len(self.failure_snapshots)
                badge_failure_snapshots_refresh = create_failure_snapshots_badge()
        with ui.tab_panels(tabs, value=tab_latest_snapshot):
            with ui.tab_panel(tab_latest_snapshot):
                with ui.row():
                    table_frame_stats_refresh = create_table_frame_stats()
                    image_latest_snapshot = ui.interactive_image(f"{root}/latest_snapshot")
                    badge_stream_status_refresh = create_stream_status_badge()
            with ui.tab_panel(tab_failures_snapshot):
                with ui.row():
                    table_failure_snapshots_refresh = create_table_failure_snapshots()
                    image_failure_snapshot = ui.interactive_image()

        # track these handles to refresh
        self.ui_table_frame_stats_refresh = table_frame_stats_refresh
        self.ui_image_latest_snapshot = image_latest_snapshot
        self.ui_image_failure_snapshot = image_failure_snapshot
        self.ui_table_failure_snapshots_refresh = table_failure_snapshots_refresh
        self.ui_badge_failure_snapshots_refresh = badge_failure_snapshots_refresh
        self.ui_badge_stream_status_refresh = badge_stream_status_refresh
        self.is_ui_ready = True

        # fallback refresh for sanity
        # ui.timer(1, self._refresh_ui)

    def _refresh_ui(self, is_failure=True):
        if not self.is_ui_ready:
            return
        self.ui_table_frame_stats_refresh()
        self.ui_image_latest_snapshot.force_reload()
        self.ui_badge_stream_status_refresh()
        if is_failure:
            self.ui_table_failure_snapshots_refresh()
            self.ui_badge_failure_snapshots_refresh()

    def _build_routes(self):
        root = self.ui_path.rstrip("/")
        @app.get(f"{root}/latest_snapshot")
        def get_latest_snapshot():
            if self.latest_snapshot == None:
                return Response(status_code=404)
            return Response(content=self.latest_snapshot.image_data, media_type=SNAPSHOT_WEB_MEDIA_TYPE)

        @app.get(f"{root}/failure_snapshots/{{frame_number}}")
        def get_failure_snapshots(frame_number: int):
            index = bisect.bisect_left(self.failure_snapshots, frame_number, key=lambda snapshot: snapshot.frame_number)
            if index < 0 or index >= len(self.failure_snapshots):
                return Response(status_code=404)
            snapshot = self.failure_snapshots[index]
            if snapshot.frame_number != frame_number:
                return Response(status_code=404)
            return Response(content=snapshot.image_data, media_type=SNAPSHOT_WEB_MEDIA_TYPE)

    def _read_results(self):
        with self.detector:
            for frame_number, result in enumerate(self.detector.get_results_blocking()):
                probs_threshold = self.probs_threshold
                snapshot = Snapshot(result, frame_number, probs_threshold)
                self.latest_snapshot = snapshot

                worst_bbox_index = np.argmin(result.probs)
                worst_probs = result.probs[worst_bbox_index]
                worst_bbox = result.bboxes[worst_bbox_index]
                is_failure = worst_probs < probs_threshold

                if is_failure:
                    self.logger.info(f"Detected failure frame={frame_number}, probs={worst_probs:.2f}, bbox={worst_bbox}")
                    self.failure_snapshots.append(snapshot)

                # lazily refresh only when frames are read
                self._refresh_ui(is_failure=is_failure)