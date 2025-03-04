import json

import numpy as np
from server.edge.edge_initialization import Edge
from server.offloading_algo.offloading_algo import OffloadingAlgo
from PIL import Image

from server.commons import OffloadingDataFiles
from server.commons import EvaluationFiles
from server.commons import InputData
from server.commons import InputDataFiles
from server.logger.log import logger
from server.communication.message_data import MessageData

import struct


class RequestHandler():
    def handle_registration(self, device_id):
        return device_id

    def handle_device_input(self, rgb565_image):
        image_array = InputData.make_array(rgb565_image=rgb565_image)
        image = Image.fromarray(image_array, 'RGB')
        image.save(InputDataFiles.input_data_file_path)
        logger.debug("Input image saved")
        return
    
    def handle_device_inference_result(self, body, received_timestamp):
        message_data = RequestHandler._from_raw('device_inference_result', body)
        message_data = RequestHandler._extend_message_data(message_data, received_timestamp, body)
        with open(OffloadingDataFiles.data_file_path_device, 'r') as f:
            device_inference_times = json.load(f)
        for l_id, inference_time in enumerate(message_data.device_layers_inference_time):
            device_inference_times[f"layer_{l_id}"] = inference_time
        with open(OffloadingDataFiles.data_file_path_device, 'w') as f:
            json.dump(device_inference_times, f, indent=4)
        # finish inference
        prediction = Edge.run_inference(message_data.offloading_layer_index, np.array(message_data.layer_output, dtype=np.float32))
        logger.debug(f"Prediction: {prediction.tolist()}")
        MessageData.save_to_file(EvaluationFiles.evaluation_file_path, message_data.to_dict())
        # run offloading algorithm
        device_inference_times, edge_inference_times, layers_sizes = RequestHandler._load_stats()
        logger.debug(f"Loaded stats data")
        offloading_algo = OffloadingAlgo(
            avg_speed=message_data.avg_speed,
            num_layers=len(layers_sizes),
            layers_sizes=list(layers_sizes),
            inference_time_device=list(device_inference_times),
            inference_time_edge=list(edge_inference_times)
        )
        return offloading_algo.static_offloading()
    
    def handle_offloading_layer(self, best_offloading_layer):
        return best_offloading_layer

    @staticmethod
    def _load_stats():
        """ Loads the offloading stats from the JSON files """
        with open(OffloadingDataFiles.data_file_path_device, 'r') as file:
            device_inference_times = json.load(file)
            device_inference_times = list({k: v for k, v in device_inference_times.items()}.values())

        with open(OffloadingDataFiles.data_file_path_edge, 'r') as file:
            edge_inference_times = json.load(file)
            edge_inference_times = list({k: v for k, v in edge_inference_times.items()}.values())

        with open(OffloadingDataFiles.data_file_path_sizes, 'r') as file:
            layers_sizes = json.load(file)
            layers_sizes = list({k: v for k, v in layers_sizes.items()}.values())
        
        return device_inference_times, edge_inference_times, layers_sizes

    @staticmethod
    def _from_raw(topic: str, payload: bytes):
        """Parse the raw message payload into a MessageData instance."""
        try:
            message_data = {}
            message_content = {}

            # decode the payload from bytes to values and parse as JSON
            message_data["timestamp"] = struct.unpack('d', payload[:8])[0]
            offset = 8
            message_data["device_id"] = payload[offset:offset+9].decode()
            offset += 9
            message_data["message_id"] = payload[offset:offset+4].decode()
            offset += 4
            message_content["offloading_layer_index"] = struct.unpack('i', payload[offset:offset+4])[0]
            offset += 4
            layer_output_size = struct.unpack('I', payload[offset:offset+4])[0]
            offset += 4
            message_content["layer_output"] = struct.unpack(f'<{int(layer_output_size/4)}f', payload[offset:offset+layer_output_size])
            offset += layer_output_size
            layers_inference_time_size = struct.unpack('i', payload[offset:offset+4])[0]
            offset += 4
            message_content["layers_inference_time"] = struct.unpack(f'<{int(layers_inference_time_size/4)}f', payload[offset:offset+layers_inference_time_size])
            message_data["message_content"] = message_content

            decoded_payload = json.dumps(message_data)

            message_content = message_data["message_content"]
            # return an instance of MessageData with extracted fields
            return MessageData(
                topic=topic,
                payload=decoded_payload,
                device_id=message_data["device_id"],
                message_id=message_data["message_id"],
                message_content=message_content,
                timestamp=message_data["timestamp"],
            )
        except json.JSONDecodeError:
            # handles payload that cannot be parsed as JSON
            raise

    @staticmethod
    def _extend_message_data(message_data: MessageData, received_timestamp: float, payload) -> MessageData:
        """Extend the message data with additional information.

        Args:
            message_data (MessageData): The message data to extend.
            received_timestamp (float): The timestamp of the message reception.

        Returns:
            MessageData: The extended message data.
        """
        # update stats info
        message_data.received_timestamp = received_timestamp
        message_data.payload_size = MessageData.get_bytes_size(payload)
        message_data.synthetic_latency = MessageData.get_synthetic_latency()
        message_data.latency = MessageData.get_latency(message_data.timestamp, message_data.received_timestamp)
        message_data.avg_speed = MessageData.get_avg_speed(
            message_data.payload_size,
            message_data.latency,
            message_data.synthetic_latency
        )
        # update offloading info
        (
            message_data.offloading_layer_index,
            message_data.layer_output,
            message_data.device_layers_inference_time
        ) = MessageData.get_offloading_info(message_data.message_content)
        return message_data