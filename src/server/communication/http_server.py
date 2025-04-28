from fastapi import FastAPI, HTTPException, Request
from server.communication.request_handler import RequestHandler
import ntplib
import threading
import time

class HttpServer:
    def __init__(
        self,
        host: str,
        port: int,
        endpoints: dict,
        ntp_server: str,
        input_height: int,
        input_width: int,
        last_offloading_layer: int,
        request_handler: RequestHandler
    ):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.endpoints = endpoints

        self.devices = set()

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
        @self.app.post(self.endpoints['registration'])
        async def registration(data: dict):
            try:
                cleaned_device_id = self.request_handler.handle_registration(data["device_id"])
                self.devices.add(cleaned_device_id)
                return {'message': 'Success', 'device': cleaned_device_id}
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

        @self.app.post(self.endpoints['device_input'])
        async def device_input(request: Request):
            try:
                body = await request.body()  # Reads raw bytes
                self.request_handler.handle_device_input(body, self.input_height, self.input_width)
                return {'message': 'Success'}
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

        @self.app.post(self.endpoints['device_inference_result'])
        async def device_inference_result(request: Request):
            try:
                received_timestamp = self._get_current_time()
                body = await request.body()  # Reads raw bytes
                self.best_offloading_layer = self.request_handler.handle_device_inference_result(body=body, received_timestamp=received_timestamp)
                return {'message': 'Success'}
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

        @self.app.get(self.endpoints['offloading_layer'])
        async def offloading_layer():
            try:
                cleaned_offloading_layer_index = self.request_handler.handle_offloading_layer(best_offloading_layer=self.best_offloading_layer)
                return {'offloading_layer_index': cleaned_offloading_layer_index}
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

    def run(self):
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port
        )
