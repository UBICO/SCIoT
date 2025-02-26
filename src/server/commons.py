from pathlib import Path
import struct

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


BASE_DIR = Path(__file__).resolve().parent


class OffloadingDataFiles:
    data_file_path_device: str = str(BASE_DIR / "device_inference_times.json")
    data_file_path_edge: str = str(BASE_DIR / "edge_inference_times.json")
    data_file_path_sizes: str = str(BASE_DIR / "layer_sizes.json")


class EvaluationFiles:
    evaluation_file_path: str = str(BASE_DIR / "evaluations/evaluations.csv")
    web_file_path: str = str(BASE_DIR / "evaluations/web.csv")


class ModelFiles:
    model_save_path: str = str(BASE_DIR / "models")


class InputDataFiles:
    test_data_file_path: str = str(BASE_DIR / "models/test/test_model/pred_data/input_data.png")  # Path to test image
    input_data_file_path: str = str(BASE_DIR / "input_data.png")  # Input image save path


class InputData:
    height = 96
    width = 96

    def __init__(self, image_path=InputDataFiles.test_data_file_path, color_mode="rgb",
                 target_size=(height, width)):  # Model input configuration
        input_image = load_img(image_path, color_mode=color_mode, target_size=target_size)
        image_array = img_to_array(input_image)
        self.image_array = np.array([image_array])

    @staticmethod
    def make_array(rgb565_image, h=height, w=width):
        image_array = []

        for i in range(h):
            row = []
            s = rgb565_image[i * w * 2:(i + 1) * w * 2]
            pixels = struct.unpack(f'>{w}H', s)
            for p in pixels:
                r = p >> 11
                g = (p >> 5) & 0x3f
                b = p & 0x1f
                r = (r * 255) / 31.0
                g = (g * 255) / 63.0
                b = (b * 255) / 31.0
                row.append([int(round(x)) for x in [r, g, b]])
            image_array.append(row)

        return np.array(image_array, dtype=np.uint8)
