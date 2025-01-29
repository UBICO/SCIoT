import os

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from model_manager_config import ModelManagerConfig


def init_folders(root_folder: str) -> None:
    os.makedirs(f"{root_folder}/layers/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/h5/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/tflite/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/h/", exist_ok=True)


def to_tflite(keras_model: Model, save: bool, save_dir: str, name: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    if save:
        with open(f"{save_dir}/{name}.tflite", 'wb') as f:
            f.write(tflite_model)
    return tflite_model

def load_h5(name: str, dir_path: str) -> Model:
    return tf.keras.models.load_model(f'{dir_path}/{name}.h5')


def create_h5_submodels(save_dir: str, model: Model) -> dict:
    submodels = {}

    start_layer_index = 0
    # iterate through layers and create submodels
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.InputLayer):
            start_layer_index = 1
            continue

        # Create input tensor
        if isinstance(layer.input_shape, list):
            input_tensor = []
            for shape in layer.input_shape:
                input_tensor.append(Input(shape=shape[1:]))
        else:
            input_tensor = Input(shape=layer.input_shape[1:])

        # pass the input tensor through each layer sequentially
        output_tensor = layer(input_tensor)
        # create the submodel from the input tensor to the current layer's output
        submodel = Model(inputs=input_tensor, outputs=output_tensor)
        submodels[layer.name] = submodel
        # save each submodel to a file
        submodel.save(f'{save_dir}/submodel_{i-start_layer_index}.h5')
    return submodels


if __name__ == "__main__":

    # initialize folders
    main_folder = "test/" + ModelManagerConfig.MODEL_DIR_PATH
    print(f"main_folder: {main_folder}")
    init_folders(main_folder)

    # load the model
    model = load_h5(name="test_model", dir_path=main_folder)

    # creates and save submodels '.tflite'
    print("creating submodels ...")
    submodels = create_h5_submodels(model=model, save_dir=f"{main_folder}/layers/h5")

    # creates and save submodels '.h'
    macro_definition = '#define LOAD_LAYER() '

    with open(f'{main_folder}/layers/h/layers.h', 'w') as super_header_file:
        for layer_index, item in enumerate(submodels.items()):
            model_name, model = item
            # convert the model content to a C array format
            print(f"created [tflite] submodel for layer: {layer_index}")
            tflite_bytes = to_tflite(model, save=True, save_dir=f"{main_folder}/layers/tflite",
                                    name=f"submodel_{layer_index}")

            # convert the model content to a C array format
            model_array = ", ".join([str(b) for b in tflite_bytes])

            # write the C header file
            print(f"created [h] submodel for layer: {layer_index}")
            with open(f'{main_folder}/layers/h/layer_{layer_index}.h', 'w') as header_file:
                header_file.write('#pragma once\n\n')
                header_file.write('#include <cstdint>\n\n')
                header_file.write('const uint8_t layer_' + str(layer_index) + '[] = {\n')
                header_file.write(model_array)
                header_file.write('\n};\n')

            super_header_file.write('#include "layer_' + str(layer_index) + '.h"\n')
            macro_definition += 'if(layer_name.equals("layer_' + str(layer_index) + '"))model = tflite::GetModel(layer_' + str(layer_index) + ');\\\n'
        super_header_file.write(macro_definition)
