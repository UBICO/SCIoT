from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.communication.request_handler import RequestHandler
import ntplib
import threading
import time


class WebsocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        endpoints: dict,
        ntp_server: str,
        last_offloading_layer: int,
        request_handler: RequestHandler
    ):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.endpoints = endpoints

        self.request_handler = request_handler
        self.best_offloading_layer = last_offloading_layer

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
        @self.app.websocket(self.endpoints['registration'])
        async def registration(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    json_data = await websocket.receive_json()
                    cleaned_device_id = self.request_handler.handle_registration(json_data["device_id"])
                    response = {'device': cleaned_device_id}
                    await websocket.send_json(response)
            except WebSocketDisconnect:
                pass

        @self.app.websocket(self.endpoints['device_input'])
        async def device_input(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    byte_data = await websocket.receive_bytes()
                    self.request_handler.handle_device_input(byte_data)
            except WebSocketDisconnect:
                pass
            
        # With chunking
        @self.app.websocket(self.endpoints['device_inference_result'])
        async def device_inference_result(websocket: WebSocket):
            await websocket.accept()
            bytes_concat = bytearray()
            try:
                while True:
                    byte_data = await websocket.receive_bytes()
                    bytes_concat.extend(byte_data)
            except WebSocketDisconnect:
                received_timestamp = self._get_current_time()
                self.best_offloading_layer = self.request_handler.handle_device_inference_result(body=bytes(bytes_concat), received_timestamp=received_timestamp)

        # Polling is necessary if closing the connection with device_inference_result endpoint is used to signal last chunk has been sent
        @self.app.websocket(self.endpoints['offloading_layer'])
        async def offloading_layer(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    str_data = await websocket.receive_text()
                    cleaned_offloading_layer_index = self.request_handler.handle_offloading_layer(best_offloading_layer=self.best_offloading_layer)
                    response = {'offloading_layer_index': cleaned_offloading_layer_index}
                    await websocket.send_json(response)
            except WebSocketDisconnect:
                pass

        # Without chunking
        # @self.app.websocket(self.endpoints['device_inference_result'])
        # async def device_inference_result(websocket: WebSocket):
        #     await websocket.accept()
        #     try:
        #         while True:
        #             byte_data = await websocket.receive_bytes()
        #             received_timestamp = self._get_current_time()
        #             self.best_offloading_layer = self.request_handler.handle_device_inference_result(body=byte_data, received_timestamp=received_timestamp)
        #             # print(len(byte_data))
        #             # print(self.best_offloading_layer)
        #             cleaned_offloading_layer_index = self.request_handler.handle_offloading_layer(best_offloading_layer=self.best_offloading_layer)
        #             response = {'offloading_layer_index': cleaned_offloading_layer_index}
        #             await websocket.send_json(response)
        #     except WebSocketDisconnect as e:
        #         pass

    def run(self):
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port
        )
