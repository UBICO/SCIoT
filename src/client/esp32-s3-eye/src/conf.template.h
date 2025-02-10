#ifndef CONF_H_
#define CONF_H_

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// WiFi Conf Hotspot
const char* SSID                = "";
const char* PWD                 = "";
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// MQTT Conf
const char* MQTT_SRV            = "hostaname.local"; // .local needed when using Hotspot so i will leave it by default
const int MQTT_PORT             = 1883;
const char* MQTT_USR            = ""; 
const char* MQTT_PWD            = ""; 
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// NTP Timer Configuration
const char* NTP_SRV             = "0.it.pool.ntp.org";
const long NTP_GMT_OFFSET       = 0;
const int NTP_DAYLIGHT_OFFSET   = 0;
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Model Configuration
const int MAX_NUM_LAYER = 59;
constexpr int BATCH_SIZE = 1;
constexpr int IMAGE_HEIGHT = 96;
constexpr int IMAGE_WIDTH = 96;
constexpr int CHANNELS = 3;
constexpr int MAX_ELEMENTS_PER_MODEL_LAYER = 1*49*49*48;
constexpr int K_TENSOR_ARENA_SIZE = 1000*1024;
// timestamp + device_id + message_id + offloading_layer_index + layer_output_size + layer_output + layers_inference_time_size + layers_inference_time
constexpr int OUTPUT_MSG_SIZE = 1*sizeof(double) + 9*sizeof(char) + 4*sizeof(char) + 1*sizeof(int) + 1*sizeof(size_t) + 2*MAX_ELEMENTS_PER_MODEL_LAYER*sizeof(float) + 1*sizeof(int) + MAX_NUM_LAYER*sizeof(float);
#define WDT_TIMEOUT 15
#define FOMO // Comment for Sequential models / Uncomment for FOMO models
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Camera Configuration
#define CAMERA_PIN_PWDN -1
#define CAMERA_PIN_RESET -1

#define CAMERA_PIN_VSYNC 6
#define CAMERA_PIN_HREF 7
#define CAMERA_PIN_PCLK 13
#define CAMERA_PIN_XCLK 15

#define CAMERA_PIN_SIOD 4
#define CAMERA_PIN_SIOC 5

#define CAMERA_PIN_D0 11
#define CAMERA_PIN_D1 9
#define CAMERA_PIN_D2 8
#define CAMERA_PIN_D3 10
#define CAMERA_PIN_D4 12
#define CAMERA_PIN_D5 18
#define CAMERA_PIN_D6 17
#define CAMERA_PIN_D7 16

#define XCLK_FREQ_HZ 16000000
#define FRAMESIZE_CUSTOM FRAMESIZE_96X96
// ------------------------------------------------------------------------------------------------------------------------------------------------------

#endif // CONF_H
