from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class OffloadingDataFiles:
    data_file_path_device: str = str(BASE_DIR / "device_inference_times.json")
    data_file_path_edge: str = str(BASE_DIR / "edge_inference_times.json")
    data_file_path_sizes: str = str(BASE_DIR / "layer_sizes.json")


class EvaluationFiles:
    evaluation_file_path: str = str(BASE_DIR / "evaluations/evaluations.csv")


class ConfigurationFiles:
    server_configuration_file_path = str = str(BASE_DIR / "settings.yaml")


class ModelFiles:
    model_save_path: str = str(BASE_DIR / "models")


class InputDataFiles:
    test_data_file_path: str = str(BASE_DIR / "models/test/test_model/pred_data/test_image.png")  # Path to test image
    input_data_file_path: str = str(BASE_DIR / "input_data.png")  # Input image save path
