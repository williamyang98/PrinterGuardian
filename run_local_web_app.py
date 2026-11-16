import argparse
import json
import logging
import os

import onnxruntime as ort
import numpy as np
from nicegui import ui

from model import Model
from bounding_boxes import BoundingBoxSplitter
from web_app import WebApp

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
    parser.add_argument("--port", type=int, default=8080, help="Port for web app to run on")
    parser.add_argument("--model", type=str, default="./model/printguard_model_f32.onnx", help="Filepath of onnx model")
    parser.add_argument("--model-metadata", type=str, default="./model/metadata.json", help="Filepath of model metadata")
    parser.add_argument("--model-prototypes", type=str, default="./model/prototypes.json", help="Filepath of model embedding vector prototypes")
    parser.add_argument("--bbox-config", type=str, default="./configs/centauri_bboxes.json", help="Filepath for bounding box configuration file")
    parser.add_argument("--list-providers", action="store_true", help="List all onnx runtime providers")
    parser.add_argument("--provider", type=str, default=DEFAULT_ORT_PROVIDER, choices=ort_providers, help="Onnx runtime inference provider")
    parser.add_argument("--device-id", type=str, default="0", help="Provider device id")
    parser.add_argument("--print-model", action="store_true", help="Print descriptions about the model's inputs and outputs")
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

    web_app = WebApp(model, bbox_splitter, args.video)
    ui.run(port=args.port)

if __name__ in {"__main__", "__mp_main__"}:
    main()