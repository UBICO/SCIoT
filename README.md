# SCIoT project
The Split Computing on IoT (SCIoT) project provides tools to use Edge Impulse models in ESP32 devices, using split computing techniques.

![Unit Tests](https://github.com/UBICO/SCIoT/actions/workflows/codecov.yml/badge.svg) [![Coverage](https://codecov.io/github/UBICO/SCIoT//coverage.svg?branch=main)](https://codecov.io/gh/UBICO/SCIoT) [![Powered by UBICO](https://img.shields.io/badge/powered%20by-UBICO-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  

## Publications
If you use this work, please consider citing our work:
- F. Bove, S. Colli and L. Bedogni, "Performance Evaluation of Split Computing with TinyML on IoT Devices," 2024 IEEE 21st Consumer Communications & Networking Conference (CCNC), Las Vegas, NV, USA, 2024, pp. 1-6, [DOI Link](http://dx.doi.org/10.1109/CCNC51664.2024.10454775).
- F. Bove and L. Bedogni, "Smart Split: Leveraging TinyML and Split Computing for Efficient Edge AI," 2024 IEEE/ACM Symposium on Edge Computing (SEC), Rome, Italy, 2024, pp. 456-460, [DOI Link](http://dx.doi.org/10.1109/SEC62691.2024.00052).

## Configuration
Clone the repository and navigate into it:

```sh
git clone https://github.com/UBICO/SCIoT.git
cd SCIoT
```

Create the virtual environment and install the dependencies:

```sh
uv sync
```

Activate the virtual environment:

```sh
source .venv/bin/activate
```

### Model setup
- Save your keras model as `test_model.h5` in `src/server/models/test/test_model/`
- Save your test image as `test_image.png` in `src/server/models/test/test_model/pred_data/`
- Split the model by running `python3 model_split.py` from `src/server/models/`
- Configure the paths as needed using `src/server/commons.py`

### Server setup
- Configure the server using `src/server/settings.yaml`

## Usage
From the repository's root directory, activate the virtual environment:

```sh
source .venv/bin/activate
```

Start the MQTT broker:

```sh
docker compose up
```

Run the edge server:

```sh
python3 src/server/edge/run_edge.py
```

Run the analytics dashboard:

```sh
streamlit run src/server/web/webpage.py
```
