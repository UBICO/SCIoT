from server.edge.edge_initialization import Edge
from server.logger.log import logger

import yaml
from server.communication.websocket_server import WebsocketServer
from server.communication.http_server import HttpServer
from server.communication.mqtt_client import MqttClient
from paho.mqtt import client as mqtt
from server.communication.request_handler import RequestHandler

from server.commons import ConfigurationFiles

if __name__ == "__main__":
    logger.info("Starting the [EDGE] MQTT client")

    # initialize edge inference times
    Edge.initialization()

    with open(ConfigurationFiles.server_configuration_file_path, "r") as f:
        config = yaml.safe_load(f)
    
    if 'websocket' in config['communication']['mode']:
        websocket_config = config['communication']['websocket']
        websocket_server = WebsocketServer(
            host=websocket_config['host'],
            port=websocket_config['port'],
            endpoint=websocket_config['endpoint'],
            ntp_server=websocket_config['ntp_server'],
            last_offloading_layer=websocket_config['last_offloading_layer'],
            request_handler=RequestHandler()
        )
        websocket_server.run()

    if 'http' in config['communication']['mode']:
        http_config = config['communication']['http']
        http_server = HttpServer(
            host=http_config['host'],
            port=http_config['port'],
            endpoints=http_config['endpoints'],
            ntp_server=http_config['ntp_server'],
            last_offloading_layer=http_config['last_offloading_layer'],
            request_handler=RequestHandler()
        )
        http_server.run()

    if 'mqtt' in config['communication']['mode']:
        mqtt_config = config['communication']['mqtt']
        mqtt_client = MqttClient(
            broker_url=mqtt_config['broker_url'],
            broker_port=mqtt_config['broker_port'],
            client_id=mqtt_config['client_id'],
            protocol=mqtt.MQTTv311,
            subscribed_topics=mqtt_config['topics'],
            ntp_server=http_config['ntp_server'],
            last_offloading_layer=mqtt_config['last_offloading_layer'],
            request_handler=RequestHandler()
        )
        mqtt_client.run()
