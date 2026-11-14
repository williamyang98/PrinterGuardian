import asyncio
import time
import json
import websockets
from websockets.asyncio.client import connect
import uuid
import logging
import traceback
import cv2
import numpy as np

def get_unix_timestamp_millis():
    return int(time.time() * 1000)

def generate_request_id():
    request_id = uuid.uuid4()
    request_id = request_id.hex
    return request_id

class WaiterSet:
    def __init__(self):
        self.entries = {}

    def notify(self, key, value):
        entry = self.entries.get(key, None)
        if entry == None:
            return False
        event = entry[0]
        entry[1] = value
        event.set()
        return True

    def wait(self, key, timeout=3.0): 
        event = asyncio.Event()
        entry = [event, None]
        self.entries[key] = entry
        async def waiter():
            await asyncio.wait_for(event.wait(), timeout=timeout)
            event.clear()
            value = entry[1]
            self.entries[key] = None
            return value
        return waiter

class Centauri:
    def __init__(self, ip: str):
        self.logger = logging.getLogger("centauri")
        self.ip = ip
        self.websocket_url = f"ws://{ip}:3030/websocket"

        self.is_manually_closed = False
        self.websocket = None
        self.event_websocket_connected = asyncio.Event()
        self.client_id = 1
        self.connect_args = {
            "ping_interval": 5,
        }
        self.status = None
        self.attributes = None
        self.websocket_listeners = WaiterSet()

    async def connect_websocket(self, autoconnect=True):
        total_connects = 0
        async for websocket in websockets.asyncio.client.connect(self.websocket_url, **self.connect_args):
            self.websocket = websocket
            total_connects += 1
            self.logger.info(f"Connected {total_connects} {'time' if total_connects == 1 else 'times'} to websocket {self.websocket_url}")
            self.event_websocket_connected.set()
            try:
                await self._listen_websocket_messages(websocket)
            except websockets.exceptions.ConnectionClosed:
                self.event_websocket_connected.clear()
            if self.is_manually_closed:
                self.logger.info(f"Websocket disconnected after intentional manual close")
                break
            if not autoconnect:
                self.logger.warn(f"Websocket disconnected after unintended close")
                break
            self.logger.info("Attempting to reconnect to websocket after unintended close...")
    
    async def _listen_websocket_messages(self, websocket):
        async for message in websocket:
            if isinstance(message, str):
                try:
                    message = json.loads(message)
                    await self._handle_json_websocket_message(message)
                except Exception as ex:
                    error_trace = traceback.format_exc()
                    self.logger.error(f"Failed to parse text websocket message as json: {error_trace}")
            elif isinstance(message, bytes):
                self.logger.warn(f"Unhandled binary websocket message: {message}")
            else:
                self.logger.warn(f"Unknown websocket message type: {message}")
    
    async def _handle_json_websocket_message(self, message):
        status = message.get("Status", None)
        if status != None:
            await self._handle_json_status_broadcast(status)
            return

        attributes = message.get("Attributes", None)
        if attributes != None:
            await self._handle_json_attributes(attributes)
            return

        request_id = message.get("Data", {}).get("RequestID", None)
        if request_id != None:
            if self.websocket_listeners.notify(request_id, message):
                return
            self.logger.warn(f"Got json message for untracked request_id={request_id}, message={message}")
            return

        self.logger.warn(f"Unhandled json message={message}")        

    async def _handle_json_status_broadcast(self, status):
        self.status = status
        hotbed_temp = status["TempOfHotbed"]
        nozzle_temp = status["TempOfNozzle"]
        light_status = status["LightStatus"]["SecondLight"] == 1
        self.logger.info(f"Status update: hotbed_temp={hotbed_temp:.3f}'C, nozzle_temp={nozzle_temp:.3f}'C, light={light_status}")
    
    def get_light_enabled(self):
        if self.status == None:
            return None
        light_status = self.status["LightStatus"]["SecondLight"] == 1
        return light_status
    
    async def _handle_json_attributes(self, attributes):
        self.attributes = attributes
        brand_name = attributes["BrandName"]
        firmware_version = attributes["FirmwareVersion"]
        machine_name = attributes["MachineName"]
        mainboard_id = attributes["MainboardID"]
        mainboard_ip = attributes["MainboardIP"]
        mainboard_mac = attributes["MainboardMAC"]
        network_status = attributes["NetworkStatus"]
        self.logger.info(f"Attributes: brand={brand_name}, name={machine_name}, mainboard_id={mainboard_id}, mac={mainboard_mac}, ip={mainboard_ip}, network_status={network_status}")

    def get_mainboard_id(self):
        if self.attributes == None:
            return ""
        mainboard_id = self.attributes["MainboardID"]
        return mainboard_id

    async def wait_websocket_connected(self):
        await self.event_websocket_connected.wait() 
        self.event_websocket_connected.clear()

    async def close_websocket(self):
        if self.websocket == None:
            self.logger.warn(f"Attempted to close non-existent websocket connection")
            return
        self.is_manually_closed = True
        # no force closing which turns off led
        # close_code = websockets.frames.CloseCode.NORMAL_CLOSURE
        close_code = websockets.frames.CloseCode.GOING_AWAY
        await self.websocket.close(code=close_code)

    async def request_status(self):
        assert self.websocket != None
        request_id = generate_request_id()
        payload = {
            "Id": "",
            "Data": {
                "Cmd": 0,
                "Data": {},
                "From": self.client_id,
                "RequestID": request_id,
                "TimeStamp": get_unix_timestamp_millis(),
                "MainboardID": self.get_mainboard_id(),
            }
        }
        wait_response = self.websocket_listeners.wait(request_id)
        await self.websocket.send(json.dumps(payload))
        response = await wait_response()

    async def request_attributes(self):
        assert self.websocket != None
        request_id = generate_request_id()
        payload = {
            "Id": "",
            "Data": {
                "Cmd": 1,
                "Data": {},
                "From": self.client_id,
                "RequestID": request_id,
                "TimeStamp": get_unix_timestamp_millis(),
                "MainboardID": self.get_mainboard_id(),
            }
        }
        wait_response = self.websocket_listeners.wait(request_id)
        await self.websocket.send(json.dumps(payload))
        response = await wait_response()

    async def enable_led(self, is_enable):
        assert self.websocket != None
        # for some reason we can only toggle the led, not set its status directly
        if self.get_light_enabled() == is_enable:
            return

        request_id = generate_request_id()
        payload = {
            "Id": "",
            "Data": {
                "Cmd": 403,
                "Data": {
                    "LightStatus": {
                        "SecondLight": is_enable,
                        "RgbLight": [0,0,0],
                    },
                },
                "From": self.client_id,
                "RequestID": request_id,
                "TimeStamp": get_unix_timestamp_millis(),
                "MainboardID": self.get_mainboard_id(),
            }
        }
        wait_response = self.websocket_listeners.wait(request_id)
        await self.websocket.send(json.dumps(payload))
        response = await wait_response()
    
    async def enable_webcam(self, is_enable):
        assert self.websocket != None
        request_id = generate_request_id()
        payload = {
            "Id": "",
            "Data": {
                "Cmd": 386,
                "Data": {
                    "Enable": 1 if is_enable else 0,
                },
                "From": self.client_id,
                "RequestID": request_id,
                "TimeStamp": get_unix_timestamp_millis(),
                "MainboardID": self.get_mainboard_id(),
            }
        }
        wait_response = self.websocket_listeners.wait(request_id)
        await self.websocket.send(json.dumps(payload))
        response = await wait_response()
        video_url = response["Data"]["Data"]["VideoUrl"]
        return f"http://{video_url}"
