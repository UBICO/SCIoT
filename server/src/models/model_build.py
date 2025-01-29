import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from model_manager_config import ModelManagerConfig


def init_folders(root_folder: str) -> None:
    os.makedirs(f"{root_folder}/layers/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/keras/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/tflite/", exist_ok=True)
    os.makedirs(f"{root_folder}/layers/h/", exist_ok=True)

def save_keras(name: str, model: Model, dir_path: str) -> None:
    model.save(f'{dir_path}/{name}.keras')

def load_keras(name: str, dir_path: str) -> Model:
    return tf.keras.models.load_model(f'{dir_path}/{name}.keras')

def build_resnet_from_scratch(img_height=10, img_width=10, num_classes=5) -> Model:
    # inputs = layers.Input(shape=(img_height, img_width, 3))
    resnet_model = tf.keras.models.Sequential()
    # initial Conv Layer
    resnet_model.add(
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(img_height, img_width, 3)))
    resnet_model.add(layers.BatchNormalization())
    resnet_model.add(layers.ReLU())
    resnet_model.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
    resnet_model.add(layers.Dense(num_classes, activation='softmax'))
    return resnet_model


if __name__ == "__main__":

    # initialize folders
    main_folder = "test/" + ModelManagerConfig.MODEL_DIR_PATH
    print(f"main_folder: {main_folder}")
    init_folders(main_folder)

    # build 'keras' model and store it
    print("building keras model ...")
    model = build_resnet_from_scratch(img_height=ModelManagerConfig.IMAGE_SIZE, img_width=ModelManagerConfig.IMAGE_SIZE)
    save_keras(name="resnet_model", model=model, dir_path=main_folder)

    # load the model
    model = load_keras(name="resnet_model", dir_path=main_folder)

    print(model.summary())