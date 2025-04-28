import struct
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class ModelInputConverter:
    @staticmethod
    def convert_png_to_nparray(png_image_path, height, width, color_mode="rgb"):
        png_image = load_img(png_image_path, color_mode=color_mode, target_size=(height, width))
        image_array = img_to_array(png_image)
        return np.array([image_array])

    @staticmethod
    def convert_rgb565_to_nparray(rgb565_image, height, width):
        image_array = []

        for i in range(height):
            row = []
            s = rgb565_image[i * width * 2:(i + 1) * width * 2]
            pixels = struct.unpack(f'>{width}H', s)
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