import json
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd

from src.commons import OffloadingDataFiles


st.set_page_config(layout="wide")
st_autorefresh(interval=3000, key="dataframerefresh")

st.title("Split computing")

# Defining the columns to read
usecols = ["received_timestamp", "payload_size", "latency", "avg_speed", "offloading_layer_index"]

# Read data with subset of columns
df = pd.read_csv("../evaluations/web.csv", usecols=usecols)

df_tail = df.tail(10)
df_last_row = df_tail.tail(1)

df_network_speed = df_tail.get(['received_timestamp', 'avg_speed']).rename(
  columns={'received_timestamp':'Timestamp (s)', 'avg_speed':'Network speed (Bytes/s)'}
)

df_offloading_layer = df_tail.get(['received_timestamp', 'offloading_layer_index']).rename(
  columns={'received_timestamp':'Timestamp (s)', 'offloading_layer_index':'Best offloading layer'}
)

quadcol = st.columns(4)

quadcol[0].metric(label="Best offloading layer", value = df_last_row['offloading_layer_index'])
quadcol[1].metric(label="Layer size", value = f"{int(df_last_row['payload_size'].iloc[0]):,} Bytes")
quadcol[2].metric(label="Latency", value = f"{float(df_last_row['latency'].iloc[0]):,.4f} s")
quadcol[3].metric(label="Network speed", value = f"{float(df_last_row['avg_speed'].iloc[0]):,.2f} Bytes/s")

with open(OffloadingDataFiles.data_file_path_device, 'r') as file:
  device_inference = json.load(file)
  device_inference_times = list({k: v for k, v in device_inference.items()}.values())
  device_inference_layers = list(range(0, len(device_inference_times)))

with open(OffloadingDataFiles.data_file_path_edge, 'r') as file:
  edge_inference = json.load(file)
  edge_inference_times = list({k: v for k, v in edge_inference.items()}.values())
  edge_inference_layers = list(range(0, len(edge_inference_times)))

device_inference_time_total = 0
for inference_time in device_inference_times:
  device_inference_time_total += inference_time

edge_inference_time_total = 0
for inference_time in edge_inference_times:
  edge_inference_time_total += inference_time

device_data_frame = pd.DataFrame({
  'Inference time (s)': device_inference_times,
  'Layer': device_inference_layers
})

edge_data_frame = pd.DataFrame({
  'Inference time (s)': edge_inference_times,
  'Layer': edge_inference_layers
})

doublecol = st.columns(2)

doublecol[0].header(f'Device inference times ({device_inference_time_total:,.4f} s)')
doublecol[0].bar_chart(device_data_frame, y='Inference time (s)', x='Layer')

doublecol[1].header(f'Edge inference times ({edge_inference_time_total:,.4f} s)')
doublecol[1].bar_chart(edge_data_frame, y='Inference time (s)', x='Layer')

doublecol[0].header('Network speed')
doublecol[0].line_chart(df_network_speed, y='Network speed (Bytes/s)', x='Timestamp (s)')

doublecol[1].header('Best offloading layer')
doublecol[1].line_chart(df_offloading_layer, y='Best offloading layer', x='Timestamp (s)')

st.header('Captured image')
st.image('../input_data.png')