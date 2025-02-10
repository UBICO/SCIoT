# Edge Server Neural Network Offloading with MQTT
This project implements an Edge Server that leverages an offloading algorithm to manage computational load between IoT devices and the server. The Edge Server evaluates the MQTT requests from connected devices and decides whether to offload the computation or let the device handle it locally based on factors such as network bandwidth, server load, and device capabilities.

## Configuration
Clone and go in the repository:

```sh
git clone https://github.com/WJW02/edge-neural-network-offloading-mqtt.git
cd edge-neural-network-offloading-mqtt/
```

Install python 3.10.12:

```sh
pyenv install 3.10.12
```

- This is the python version used to test the project
- Newer versions of python, like 3.12.X, don't support the tensorflow version used in the project

Switch to python 3.10.12:

```sh
pyenv global 3.10.12
```

- You can switch back after configuration by running `pyenv system global`

Create virtual environment:

```sh
python3 -m venv venv
```

Activate virtual environment:

```sh
source venv/bin/activate
```

Install project dependencies:

```sh
pip3 install .
```

Configure the absolute path to project (e.g. `/home/username/Documents/edge-neural-network-offloading-mqtt/`):

```sh
cd $(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo /absolute/path/to/project > project.pth
```
## Model setup
- Save your keras model as `test_model.h5` in `src/models/test/test_model/`
- Save your test images in `src/models/test/test_model/pred_data/`
- Configure `InputDataFiles` and `InputData` in `src/commons.py` as needed

## Mqtt server setup
- Configure `MqttClientConfig` in  `src/mqtt_client/mqtt_configs.py`

## Workflow
In root directory, run mosquitto broker:

```sh
docker compose up
```

In `src/edge`, run edge:

```sh
python3 run_edge.py
```

In `src/web`, run webpage:

```sh
streamlit run webpage.py
```
