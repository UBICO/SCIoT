import os

import pytest

from server.src.commons import OffloadingDataFiles
from server.src.mqtt_client.mqtt_client import MqttClient
from server.tests.commons import TestSamples


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
    return MqttClient()


@pytest.fixture
def device_fixture(mqtt_client_fixture):
    return MqttClient()
