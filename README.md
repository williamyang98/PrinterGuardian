## Printer Guardian
![Screenshot of web app](./docs/screenshot.png)

Web application to monitor Elegoo Centauri Carbon's webcam for spaghetti 3D printing errors using an AI model running using GPU acceleration on Windows using DirectML.

### Usage Instructions
1. Create python virtual environment for python 3.13: ```python -m venv venv```
2. Activate python virtual environment: ```source ./venv/Scripts/activate``` or ```source ./venv/bin/activate```
3. Install python dependencies ```pip install -r requirements.txt```
4. Run one of the commands below

| Action | Command |
| --- | --- |
| Monitor Elegoo Carbon Centauri from app | ```python ./run_printer_web_app.py --ip [ip]``` |
| Test web app on video file or url stream | ```python ./run_local_web_app.py [url\|path]``` |
| Run model on video file or stream | ```python ./process_video_stream.py [url\|path]``` |
| Run model on single image file | ```python ./process_single_image.py [path]``` |
| Test Elegoo Centauri Carbon connection | ```python ./test_centauri_connection.py --ip [ip]``` |

- For help run: ```python [file] --help```
- There are test files in ```/data``` which can be used with the above commands
- There are bounding box configurations in ```/configs``` which can be passed via ```--bbox-config [path]```
- To use non-GPU based onnxruntime for inference install regular cpu based onnxruntime package by commenting the line ```onnxruntime==x.x.x``` in the file ```requirements.txt``` and install dependencies with ```pip install -r requirements.txt```

### Attributions
- [oliverbravery/Edge-FDM-Fault-Detection](https://github.com/oliverbravery/Edge-FDM-Fault-Detection): AI model is based off their paper
- [oliverbravery/PrintGuard](https://github.com/oliverbravery/PrintGuard): Specific fine tuned model is from their repository
