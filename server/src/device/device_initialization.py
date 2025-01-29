import json

import tensorflow as tf

from src.commons import OffloadingDataFiles
from src.models.model_manager import ModelManager


class Device:
    @staticmethod
    def initialization():
        # load the model and make predictions
        model_manager = ModelManager()
        model_manager.load_model()
        
        device_inference_times = {}
        start_layer_index = 0

        # set end_layer_index to the number of layers if not provided
        end_layer_index = len(model_manager.model.layers)

        # adjust model start layer index (offset is applied if the start index is 0 and the 0th layer is an InputLayer)
        start_layer_offset = 1 if isinstance(model_manager.model.layers[start_layer_index], tf.keras.layers.InputLayer) else 0
        start_layer_index = start_layer_index+start_layer_offset if start_layer_index == 0 else start_layer_index

        # set the layers to use
        layers_to_use = model_manager.model.layers[start_layer_index:end_layer_index]

        for layer_index, layer in enumerate(layers_to_use, start=start_layer_index):
            layer_name = "layer_" + str(layer_index-start_layer_offset)
            device_inference_times[layer_name] = 0
        
        with open(OffloadingDataFiles.data_file_path_device, "w") as f:
            json.dump(device_inference_times, f, indent=4)
    
    reset = initialization


if __name__ == "__main__":
    Device.initialization()