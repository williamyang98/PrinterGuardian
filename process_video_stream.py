import onnxruntime as ort
import argparse
import numpy as np
import json
from model import Model
import logging
import os
import cv2

from detector import Detector
from bounding_boxes import BoundingBoxSplitter


logger = logging.getLogger(__name__)

def main():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)

    ort_providers = ort.get_available_providers()
    DEFAULT_ORT_PROVIDER = ort_providers[0]
    if not DEFAULT_ORT_PROVIDER in ort_providers:
        logger.error(f"{DEFAULT_ORT_PROVIDER} not in list of onnxruntime providers: [{','.join(ort_providers)}]")
        return

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, help="Filepath or url of video stream")
    parser.add_argument("--output", type=str, help="Dirpath of output frames", default="results")
    parser.add_argument("--threshold", type=float, help="Minimum confidence threshold for successful print", default=0.2)
    parser.add_argument("--model", type=str, default="./model/printguard_model_f32.onnx", help="Filepath of onnx model")
    parser.add_argument("--model-metadata", type=str, default="./model/metadata.json", help="Filepath of model metadata")
    parser.add_argument("--model-prototypes", type=str, default="./model/prototypes.json", help="Filepath of model embedding vector prototypes")
    parser.add_argument("--bbox-config", type=str, default="./configs/centauri_bboxes.json", help="Filepath for bounding box configuration file")
    parser.add_argument("--list-providers", action="store_true", help="List all onnx runtime providers")
    parser.add_argument("--provider", type=str, default=DEFAULT_ORT_PROVIDER, choices=ort_providers, help="Onnx runtime inference provider")
    parser.add_argument("--device-id", type=str, default="0", help="Provider device id")
    parser.add_argument("--print-model", action="store_true", help="Print descriptions about the model's inputs and outputs")
    parser.add_argument("--allow-drop-frames", action="store_true", help="Allow detector to drop frames")
    args = parser.parse_args()

    if args.list_providers:
        print(f"List of onnxruntime providers: [{','.join(ort_providers)}]")
        return

    with open(args.model, "rb") as fp:
        model_data = fp.read()
    
    with open(args.model_metadata, "r", encoding="utf-8") as fp:
        model_metadata = json.load(fp)

    with open(args.model_prototypes, "r", encoding="utf-8") as fp:
        model_prototypes = json.load(fp)

    with open(args.bbox_config, "r", encoding="utf-8") as fp:
        bbox_config_json = json.load(fp)

    provider_options = None
    if args.provider != "CPUExecutionProvider":
        provider_options = {
            "device_id": args.device_id,
        }

    session = ort.InferenceSession(
        model_data,
        providers=[args.provider],
        provider_options=[provider_options],
    )
    logger.info(f"Onnxruntime session running with providers: [{','.join(session.get_providers())}]")

    model = Model(session, model_metadata, model_prototypes)

    if args.print_model:
        print(model.debug_print())
        return

    bbox_splitter = BoundingBoxSplitter.load_from_json(bbox_config_json)

    def get_video_stream():
        capture = cv2.VideoCapture(args.video)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32) / 255.0
            yield frame
    
    video_stream = get_video_stream()
    detector = Detector(model, video_stream, bbox_splitter, allow_drop_frames=args.allow_drop_frames)
    os.makedirs(args.output, exist_ok=True)
    with detector:
        for frame_number, result in enumerate(detector.get_results_blocking()):
            worst_bbox_index = np.argmin(result.probs)
            worst_probs = result.probs[worst_bbox_index]
            worst_bbox = result.bboxes[worst_bbox_index]
            probs_threshold = args.threshold
            if worst_probs < probs_threshold:
                logger.info(f"Detected failure frame={frame_number}, probs={worst_probs:.2f}, bbox={worst_bbox}")
                annotated_frame = result.draw_bboxes(probs_threshold)
                output_filepath = os.path.join(args.output, f"frame_{frame_number}.png")
                cv2.imwrite(output_filepath, annotated_frame)

if __name__ == "__main__":
    main()