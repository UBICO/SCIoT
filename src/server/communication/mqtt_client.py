import json
import random
import time

import ntplib
import paho.mqtt.client as mqtt
import threading
import queue

from server.logger.log import logger
from server.communication.request_handler import RequestHandler


class MqttClient:
    def __init__(
            self,
            broker_url: str,
            broker_port: int,
            client_id: str,
            protocol: str,
            subscribed_topics: list,
            ntp_server: str,
            last_offloading_layer: int,
            request_handler: RequestHandler
    ):
        self.broker_url = broker_url
        self.broker_port = broker_port
        self.client_id = client_id

        # Create the client with the specific MQTT protocol version
        self.client = mqtt.Client(client_id=client_id, protocol=protocol)

        # Attach callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Set up topics
        self.subscribed_topics = subscribed_topics

        # Set up NTP client
        self.ntp_client = ntplib.NTPClient()
        self.ntp_server = ntp_server
        self.offset = self.sync_with_ntp()
        self.start_timestamp = self.get_current_time()

        self.request_handler = request_handler
        self.best_offloading_layer = last_offloading_layer

        # Set up helper thread
        self.task_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    @staticmethod
    def create_random_payload():
        """Creates a random payload for testing."""
        message = json.dumps({"id": random.randint(1, 1000)})
        return message

    def publish(self, topic: str, message: str, qos: int = 2):
        """Publishes a message to a topic."""
        logger.debug(f"Publishing message to {topic}: {message}")
        try:
            self.client.publish(topic, message, qos=qos, retain=False)
        except Exception as e:
            logger.debug(f"Error publishing message: {e}")

    def subscribe(self, topic: str):
        """Subscribes to a topic."""
        logger.debug(f"Subscribing to topic: {topic}")
        self.client.subscribe(topic)

    def run(self):
        """Connect to the broker and start the MQTT client loop."""
        self.client.connect(self.broker_url, self.broker_port, 60)
        self.client.loop_forever()

    def stop(self):
        """Stops the MQTT client loop and disconnects."""
        logger.debug("Disconnecting MQTT client")
        self.client.disconnect()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.debug(f"Connected to {self.broker_url}:{self.broker_port} with client ID {self.client_id}")
            for topic in self.subscribed_topics.values():
                self.subscribe(topic)
            logger.debug(f"Initial NTP timestamp from NTP server {self.ntp_server}: {self.start_timestamp}")
        else:
            logger.debug(f"Connection failed with code {rc}")
    
    def sync_with_ntp(self) -> float:
        ntp_timestamp = None
        while ntp_timestamp is None:
            try:
                response = self.ntp_client.request(self.ntp_server)
                # Get the offset between local clock time and ntp server time (seconds since 1900)
                offset = response.offset
                logger.debug(f"Synchronized with NTP server. Offset: {offset} seconds")
                return offset
            except ntplib.NTPException as _:
                time.sleep(1)
        threading.Timer(600, self.sync_with_ntp).start()

    def get_current_time(self) -> float:
        return time.time() + self.offset
    
    def _worker(self):
        while True:
            task = self.task_queue.get()
            task()
            self.task_queue.task_done()

    def on_message(self, client, userdata, message):
        received_timestamp = self.get_current_time()

        def task():
            self.handle_message_task(message, received_timestamp)

        self.task_queue.put(task)   # Submit the task to the worker thread queue

    def handle_message_task(self, message, received_timestamp):
        if message.topic == self.subscribed_topics['device_inference_result']:  # Updates device time, runs offloading algorithm and sends best offloading layer
            logger.debug('Device inference result received')
            self.best_offloading_layer = self.request_handler.handle_device_inference_result(body=message.payload, received_timestamp=received_timestamp)
            cleaned_offloading_layer_index = self.request_handler.handle_offloading_layer(best_offloading_layer=self.best_offloading_layer)
            message_data = {'offloading_layer_index': cleaned_offloading_layer_index}
            self.publish(self.subscribed_topics['offloading_layer'], json.dumps(message_data))
            logger.debug('Best offloading layer sent')
        elif message.topic == self.subscribed_topics['device_input']: # Save input image
            logger.debug('Device input received')
            self.request_handler.handle_device_input(message.payload)
            logger.debug('Device input saved')
        elif message.topic == self.subscribed_topics['registration']: # Sends best offloading layer
            logger.debug('Registration request received')
            decoded_payload = message.payload.decode()
            json_data = json.loads(decoded_payload)
            cleaned_device_id = self.request_handler.handle_registration(json_data["device_id"])
            cleaned_offloading_layer_index = self.request_handler.handle_offloading_layer(best_offloading_layer=self.best_offloading_layer)
            message_data = {'offloading_layer_index': cleaned_offloading_layer_index}
            self.publish(self.subscribed_topics['offloading_layer'], json.dumps(message_data))
            logger.debug('Best offloading layer sent')

