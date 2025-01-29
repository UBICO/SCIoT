import json

import tensorflow as tf
import numpy as np

from src.commons import OffloadingDataFiles
from src.commons import InputData
from src.models.model_manager import ModelManager


class Edge:
    @staticmethod
    def run_inference(offloading_layer_index: int, offloading_layer_output: np.array):
        # check the shape and dtype
        print(offloading_layer_output.shape)
        print(offloading_layer_output.dtype)

        # load model inference times
        with open(OffloadingDataFiles.data_file_path_edge, 'r') as file:
            edge_inference_times = json.load(file)

        # load the model and make predictions
        model_manager = ModelManager(inference_times=edge_inference_times)
        model_manager.load_model()

        # set the layers to use
        predictions = {}
        first_layer_index = 0
        start_layer_index = offloading_layer_index + 1

        # set end_layer_index to the number of layers if not provided
        num_of_layers = len(model_manager.model.layers)

        # adjust model start layer index (offset is applied if the first layer is an InputLayer)
        start_layer_offset = 1 if isinstance(model_manager.model.layers[first_layer_index], tf.keras.layers.InputLayer) else 0
        start_layer_index = start_layer_index+start_layer_offset

        # set the layers to use
        layers_to_use = model_manager.model.layers[start_layer_index:num_of_layers]

        # return offloading layer output if it's the last layer
        if start_layer_index == num_of_layers:
            return offloading_layer_output

        # loop through the layers and make predictions
        input_data = offloading_layer_output

        for layer_index, layer in enumerate(layers_to_use, start=start_layer_index):
            prediction_data = []
            if layer_index == start_layer_index:   # if it's the starting layer
                prediction_data.append(input_data)
            else:
                # Get the previous layers' output tensor
                inbound_node = layer._inbound_nodes[0]
                if isinstance(inbound_node.inbound_layers, list):
                    for inbound_layer in inbound_node.inbound_layers:
                        prediction_data.append(predictions[inbound_layer])
                else:
                    prediction_data.append(predictions[inbound_node.inbound_layers])

            prediction = model_manager.predict_single_layer(layer_index, start_layer_offset, prediction_data)
            predictions[layer] = prediction

        # save the inference times to a file
        model_manager.save_inference_times()

        return predictions[layers_to_use[num_of_layers-start_layer_index-1]]

    @staticmethod
    def initialization():
        # original array
        image_array = InputData().image_array
        image_array = image_array / 255.0 # Normalize pixel values

        # check the shape and dtype
        print(image_array.shape)
        print(image_array.dtype)

        # load the model and make predictions
        model_manager = ModelManager()
        model_manager.load_model()

        # set the layers to use
        predictions = {} 
        layer_sizes = {}
        first_layer_index = 0
        start_layer_index = first_layer_index

        # set end_layer_index to the number of layers if not provided
        end_layer_index = len(model_manager.model.layers)

        # adjust model start layer index (offset is applied if the first layer is an InputLayer)
        start_layer_offset = 1 if isinstance(model_manager.model.layers[first_layer_index], tf.keras.layers.InputLayer) else 0
        start_layer_index = start_layer_index+start_layer_offset

        # set the layers to use
        layers_to_use = model_manager.model.layers[start_layer_index:end_layer_index]

        # loop through the layers and make predictions
        input_data = image_array

        for layer_index, layer in enumerate(layers_to_use, start=start_layer_index):
            prediction_data = []
            if layer_index == start_layer_index:   # if it's the first layer
                prediction_data.append(input_data)
            else:
                # Get the previous layers' output tensor
                inbound_node = layer._inbound_nodes[0]
                if isinstance(inbound_node.inbound_layers, list):
                    for inbound_layer in inbound_node.inbound_layers:
                        prediction_data.append(predictions[inbound_layer])
                else:
                    prediction_data.append(predictions[inbound_node.inbound_layers])

            prediction = model_manager.predict_single_layer(layer_index, start_layer_offset, prediction_data)
            layer_sizes[layer_index-start_layer_offset] = float(model_manager.get_layer_size_in_bytes(layer, prediction))
            predictions[layer] = prediction

        # save the inference times to a file
        model_manager.save_inference_times()

        # save the layer sizes to a file
        with open(OffloadingDataFiles.data_file_path_sizes, "w") as f:
            json.dump(layer_sizes, f, indent=4)

    reset = initialization


if __name__ == "__main__":
    Edge.initialization()