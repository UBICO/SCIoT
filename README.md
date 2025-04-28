# SCIoT project
The Split Computing on IoT (SCIoT) project provides tools to use Edge Impulse models in ESP32 devices, using split computing techniques.

![Unit Tests](https://github.com/UBICO/SCIoT/actions/workflows/codecov.yml/badge.svg) [![Coverage](https://codecov.io/github/UBICO/SCIoT//coverage.svg?branch=main)](https://codecov.io/gh/UBICO/SCIoT) [![Powered by UBICO](https://img.shields.io/badge/powered%20by-UBICO-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)]()  

## Publications
If you use this work, please consider citing our work:
- F. Bove, S. Colli and L. Bedogni, "Performance Evaluation of Split Computing with TinyML on IoT Devices," 2024 IEEE 21st Consumer Communications & Networking Conference (CCNC), Las Vegas, NV, USA, 2024, pp. 1-6, [DOI Link](http://dx.doi.org/10.1109/CCNC51664.2024.10454775).
- F. Bove and L. Bedogni, "Smart Split: Leveraging TinyML and Split Computing for Efficient Edge AI," 2024 IEEE/ACM Symposium on Edge Computing (SEC), Rome, Italy, 2024, pp. 456-460, [DOI Link](http://dx.doi.org/10.1109/SEC62691.2024.00052).

## Configuration
Clone and go in the repository:

```sh
git clone https://github.com/UBICO/SCIoT.git
```

Install python 3.11:

```sh
pyenv install 3.11
```

- Newer versions of python don't support the tensorflow version used in the project

Switch to python 3.11:

```sh
pyenv global 3.11
```

- You can switch back after the configuration process by running `pyenv system global`

Create a virtual environment:

```sh
python3 -m venv venv
```

Activate the virtual environment:

```sh
source venv/bin/activate
```

Install the project's dependencies:

```sh
pip3 install .
```

Configure the absolute path to the project's `src` directory (e.g. `/home/username/Documents/SCIoT/src/`):

```sh
cd $(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo /absolute/path/to/project/src/ > project.pth
```
### Model setup
- Save your keras model as `test_model.h5` in `src/server/models/test/test_model/`
- Save your test image as `test_image.png` in `src/server/models/test/test_model/pred_data/`
- Split the model by running `python3 model_split.py` in `src/server/models/`
- Configure the paths as needed using `src/server/commons.py`

### Server setup
- Configure the server using `src/server/settings.yaml`

## Usage
In root directory, run the MQTT broker:

```sh
docker compose up
```

In `src/server/edge`, run the edge server:

```sh
python3 run_edge.py
```

In `src/server/web`, run the webpage:

```sh
streamlit run webpage.py
```
