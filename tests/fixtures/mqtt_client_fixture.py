import os

import pytest

from server.commons import OffloadingDataFiles
from server.communication.mqtt_client import MqttClient
from tests.commons import TestSamples
from paho.mqtt import client as mqtt
from server.communication.request_handler import RequestHandler


@pytest.fixture
def offloading_data_fixture(monkeypatch):
    """ Fixture to override OffloadingDataFiles paths for all tests. """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    monkeypatch.setattr(OffloadingDataFiles, "data_file_path_device",
                        os.path.join(base_path, TestSamples.data_file_path_device))
    monkeypatch.setattr(OffloadingDataFiles, "data_file_path_edge",
                        os.path.join(base_path, TestSamples.data_file_path_edge))
    monkeypatch.setattr(OffloadingDataFiles, "data_file_path_sizes",
                        os.path.join(base_path, TestSamples.data_file_path_sizes))

    return OffloadingDataFiles  # Optional: return for reference if needed


@pytest.fixture
def mqtt_client_fixture(offloading_data_fixture):
    """ Fixture to create an MQTT client with overridden file paths. """
    broker_url = 'hostname.local'
    broker_port = 1883
    client_id = 'edge'
    topics = {
      'registration': 'devices/',
      'offloading_layer': 'device_01/offloading_layer',
      'device_input': 'device_01/input_data',
      'device_inference_result': 'device_01/model_inference_result'
    }
    ntp_server = '0.it.pool.ntp.org'
    input_height = 96
    input_width = 96
    last_offloading_layer = 58
    return MqttClient(
        broker_url=broker_url,
        broker_port=broker_port,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
        subscribed_topics=topics,
        ntp_server=ntp_server,
        input_height=input_height,
        input_width=input_width,
        last_offloading_layer=last_offloading_layer,
        request_handler=RequestHandler()
    )


@pytest.fixture
def device_fixture(mqtt_client_fixture):
    broker_url = 'hostname.local'
    broker_port = 1883
    client_id = 'edge'
    topics = {
      'registration': 'devices/',
      'offloading_layer': 'device_01/offloading_layer',
      'device_input': 'device_01/input_data',
      'device_inference_result': 'device_01/model_inference_result'
    }
    ntp_server = '0.it.pool.ntp.org'
    input_height = 96
    input_width = 96
    last_offloading_layer = 58
    return MqttClient(
        broker_url=broker_url,
        broker_port=broker_port,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
        subscribed_topics=topics,
        ntp_server=ntp_server,
        input_height=input_height,
        input_width=input_width,
        last_offloading_layer=last_offloading_layer,
        request_handler=RequestHandler()
    )
