from centauri import Centauri
import asyncio
import logging
import os
import argparse

logger = logging.getLogger(__name__)

async def main():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, default="192.168.1.130", help="IP address of Centauri Carbon printer")
    args = parser.parse_args()

    printer = Centauri(args.ip)

    connect_websocket_task = asyncio.create_task(printer.connect_websocket())
    await printer.wait_websocket_connected()
    await printer.request_status()
    await printer.request_attributes()

    await printer.enable_led(True)
    video_url = await printer.enable_webcam(True)
    logger.info(f"Got video_url={video_url}")

    while True:
        res = await asyncio.to_thread(input, "Enable led: ")
        if res.lower().strip() in ["exit", "quit"]:
            break
        if res.lower().strip() in ["y", "yes"]:
            await printer.enable_led(True)
        else:
            await printer.enable_led(False)

    await printer.close_websocket()
    await connect_websocket_task


if __name__ == "__main__":
    asyncio.run(main())