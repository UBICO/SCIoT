import enum
from dataclasses import dataclass

from paho.mqtt import client as mqtt


class Topics(enum.Enum):
    registration = "devices/"
    offloading_layer = "device_01/offloading_layer"
    device_input = "device_01/input_data"
    device_inference_result = "device_01/model_inference_result"


@dataclass
class MqttClientConfig:
    broker_url: str = "hostname.local"
    broker_port: int = 1883
    client_id: str = "edge"
    subscribe_topics: list = (
        Topics.registration.value,
        Topics.offloading_layer.value,
        Topics.device_input.value,
        Topics.device_inference_result.value
    )
    ntp_server: str = "0.it.pool.ntp.org"
    protocol: mqtt.MQTTv311 = mqtt.MQTTv311


@dataclass
class DefaultMessages:
    offloading_layer_msg = {
        "device_id": "edge",
        "message_id": "edge",
        "timestamp": None,
        "message_content": "OffloadingLayer",
        "offloading_layer_index": None,
    }