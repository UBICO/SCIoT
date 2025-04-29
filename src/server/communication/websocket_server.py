from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.communication.request_handler import RequestHandler
import ntplib
import threading
import time
import json
from server.logger.log import logger


class WebsocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        endpoint: str,
        ntp_server: str,
        input_height: int,
        input_width: int,
        last_offloading_layer: int,
        request_handler: RequestHandler
    ):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.endpoint = endpoint

        # Set up model
        self.input_height = input_height
        self.input_width = input_width
        self.best_offloading_layer = last_offloading_layer

        # Set up request handler
        self.request_handler = request_handler

        # Set up NTP client
        self.ntp_client = ntplib.NTPClient()
        self.ntp_server = ntp_server
        self.offset = self._sync_with_ntp()
        self.start_timestamp = self._get_current_time()
        self._setup_routes()

    def _sync_with_ntp(self) -> float:
        ntp_timestamp = None
        while ntp_timestamp is None:
            try:
                response = self.ntp_client.request(self.ntp_server)
                # Get the offset between local clock time and ntp server time (seconds since 1900)
                offset = response.offset
                return offset
            except ntplib.NTPException as _:
                time.sleep(1)
        threading.Timer(600, self.sync_with_ntp).start()

    def _get_current_time(self) -> float:
        return time.time() + self.offset

    def _setup_routes(self):
        @self.app.websocket(self.endpoint)
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            logger.info('WebSocket connection established')

            try:
                while True:
                    message = await websocket.receive()  # Receives both text and binary
                    received_timestamp = self._get_current_time()

                    if 'text' in message:  # JSON/Text message
                        try:
                            json_data = json.loads(message['text'])  # Parse JSON
                            logger.debug('Registration request received')
                            cleaned_device_id = self.request_handler.handle_registration(json_data["device_id"])
                            response = {'channel': 'registration', 'device': cleaned_device_id}
                            await websocket.send_json(response)
                            logger.debug('Registration response sent')
                        except json.JSONDecodeError:
                            logger.debug('Error: Received non-JSON text data')

                    elif 'bytes' in message:  # Binary message
                        binary_data = message['bytes']
                        if len(binary_data) == 18432:
                            logger.debug('Device input received')
                            self.request_handler.handle_device_input(binary_data, self.input_height, self.input_width)
                            logger.debug('Device input saved')
                        else:
                            logger.debug('Device inference result received')
                            self.best_offloading_layer = self.request_handler.handle_device_inference_result(body=binary_data, received_timestamp=received_timestamp)
                            cleaned_offloading_layer_index = self.request_handler.handle_offloading_layer(best_offloading_layer=self.best_offloading_layer)
                            response = {'channel': 'offloading_layer', 'offloading_layer_index': cleaned_offloading_layer_index}
                            await websocket.send_json(response)
                            logger.debug('Best offloading layer sent')
            except WebSocketDisconnect:
                logger.info('WebSocket connection closed')

    def run(self):
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port
        )
