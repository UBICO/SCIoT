/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* DEPENDENCIES LIBS
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include <WiFi.h>
#include <AsyncMqttClient.h>
#include <sys/time.h>
#include <UUID.h>
#include <ArduinoJson.h>
#include "esp_task_wdt.h"
#include "esp_camera.h"

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* LIBS for TFLITE
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

#define MODEL_NAME "test_model"
#include "model_layers/layers.h"
#include "conf.h"

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*  ARDUINO JSON CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
struct SpiRamAllocator : ArduinoJson::Allocator {
  void* allocate(size_t size) override {
    return heap_caps_malloc(size, MALLOC_CAP_SPIRAM);
  }

  void deallocate(void* pointer) override {
    heap_caps_free(pointer);
  }

  void* reallocate(void* ptr, size_t new_size) override {
    return heap_caps_realloc(ptr, new_size, MALLOC_CAP_SPIRAM);
  }
};

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*  CONFIGURATIONS & GLOBAL VARIABLES
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
// Communication & Offloading Variables
AsyncMqttClient             mqttClient;
TimerHandle_t               mqttReconnectTimer;
TimerHandle_t               mqttDisconnectTimer;
TimerHandle_t               wifiReconnectTimer;
TimerHandle_t               deviceRegistrationTimer;
bool                        deviceRegistered = false;
UUID                        uuid;
String                      MessageUUID = "";
String                      DeviceUUID = "";
SpiRamAllocator             allocator;

String                      device_registration_topic = "devices/";
String                      offloading_layer_topic;
String                      input_data_topic;
String                      model_inference_result_topic;

bool                        modelDataLoaded = false;
float*                      inputBuffer = nullptr;
char*                       input_message = nullptr;
char*                       output_message = nullptr;
int                         best_offloading_layer_index = MAX_NUM_LAYER-1;
#ifdef FOMO
float*                      lastMultiOutputLayerData = nullptr;
#endif // FOMO

// Neural Network Variables
tflite::MicroErrorReporter  micro_error_reporter;
tflite::ErrorReporter*      error_reporter = &micro_error_reporter;
const tflite::Model*        model = nullptr;
tflite::MicroInterpreter*   interpreter = nullptr;
TfLiteTensor*               input;
TfLiteTensor*               output;
uint8_t*                    tensor_arena = nullptr;
bool                        modelLoaded = false;
bool                        firstInferenceDone = false; 

/*
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* TIMER CONFIGURATION & FLOATING-POINT TIMESTAMP
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
String getCurrTimeStr(){
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time_t currentTime = tv.tv_sec;
  int milliseconds = tv.tv_usec / 1000;
  int microseconds = tv.tv_usec % 1000000;
  char currentTimeStr[30];
  snprintf(currentTimeStr, sizeof(currentTimeStr), "%ld.%03d%03d", currentTime, milliseconds, microseconds);
  String currentTimeString = String(currentTimeStr);
  return currentTimeString;
}

double getCurrTime(){
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time_t currentTime = tv.tv_sec;
  int microseconds = tv.tv_usec;
  double timestamp = currentTime + (microseconds / 1000000.0);
  return timestamp;
}

void timeConfiguration(){
  // Configure NTP time synchronization
  configTime(NTP_GMT_OFFSET, NTP_DAYLIGHT_OFFSET, NTP_SRV);
  Serial.println("Connecting to NTP Server...");
  // Try obtaining the time until successful
  struct tm timeinfo;
  while (!getLocalTime(&timeinfo)) {
    delay(500);
  }

  // Print current time
  Serial.println("NTP Time Configured - Current Time: ");
  Serial.println(getCurrTime(), 6);
  return;
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* LOAD NN LAYER
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void loadNeuralNetworkLayer(String layer_name){
  // Import del modello da testare -> Nome nell'header file
  LOAD_LAYER();

  if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Model provided is schema version not equal to supported!");
      return;
  } else { Serial.println("Model Layer Loaded!"); }

  // Questo richiama tutte le implementazioni delle operazioni di cui abbiamo bisogno
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, K_TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;
  Serial.println("Interprete ok");

  // Alloco la memoria del tensor_arena per i tensori del modello
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
      Serial.println("AllocateTensors() failed");
      return;
  } else { Serial.println("AllocateTensors() done"); }
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* INFERENCE FOR NN LAYER
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
extern "C" void runNeuralNetworkLayer(int offloading_layer_index, float inputBuffer[]) {
  // Register the current running task to the watchdog
  esp_task_wdt_add(NULL); 

  // Initialize input data with image
  float* inputData = inputBuffer;
  int inputSize = BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS * sizeof(float);

  // Initialize inference times array
  float inference_times[MAX_NUM_LAYER];

#ifdef FOMO
  int multiOutputLayers[] = {16, 34, 43};
  size_t multiOutputLayersSize = sizeof(multiOutputLayers)/sizeof(multiOutputLayers[0]);
#endif

  // Assuming inputData is in the format expected by your neural network
  for (int i = 0; i <= offloading_layer_index; i++) {
    String layer_name = "layer_" + String(i);
    float inizio = micros();
    
    loadNeuralNetworkLayer(layer_name); // Load the appropriate layer
    input = interpreter->input(0);

    // Copy the input data to the input tensor
    memcpy(input->data.f, inputData, inputSize);

#ifdef FOMO
    // Copy the input data of last layer with multiple outputs to the input tensor
    if (interpreter->inputs_size() > 1) {
      if (interpreter->inputs_size() == 2) {
        input = interpreter->input(1);
        memcpy(input->data.f, lastMultiOutputLayerData, inputSize);
      } else {
        Serial.println("Error: this device can only handle sequential and FOMO models");
        return;
      }
    }
#endif

#ifdef DEBUG
    Serial.print("LAYER " + String(i) + " INPUT TENSOR: (");
    for (int j = 0; j < input->dims->size; ++j) {
      Serial.print(input->dims->data[j]);
      if (j+1 < input->dims->size) {
        Serial.print(", ");
      }
    }
    Serial.println(")");

    int numInput = 1;
    for (int j = 0; j < input->dims->size; ++j) {
      numInput *= input->dims->data[j];
    }
    Serial.println("LAYER " + String(i) + " INPUT DATA:");
    for (int j = 0; j < numInput; ++j) {
  #ifdef ALL
      Serial.print(inputData[j]);
      Serial.print(" ");
  #else // Only print first 3 and last 3 elements
      if (j < 3) {
        Serial.print(inputData[j]);
        Serial.print(" ");
      } else if (j >= numInput-3) {
        Serial.print(inputData[j]);
        Serial.print(" ");
      }
  #endif // ALL
    }
    Serial.println();
  #ifdef FOMO
    if (interpreter->inputs_size() == 2) {
      Serial.println("LAYER " + String(i) + " INPUT DATA 2:");
      for (int j = 0; j < numInput; ++j) {
    #ifdef ALL
        Serial.print(lastMultiOutputLayerData[j]);
        Serial.print(" ");
    #else // Only print first 3 and last 3 elements
        if (j < 3) {
          Serial.print(lastMultiOutputLayerData[j]);
          Serial.print(" ");
        } else if (j >= numInput-3) {
          Serial.print(lastMultiOutputLayerData[j]);
          Serial.print(" ");
        }
    #endif // ALL
      }
      Serial.println();
    }
  #endif // FOMO
#endif // DEBUG

    // Run inference
    esp_task_wdt_reset();
    interpreter->Invoke();
    esp_task_wdt_reset();

    // Extract relevant information from the output tensor
    output = interpreter->output(0);
    float* outputData = output->data.f;
    int outputSize = output->bytes;

#ifdef DEBUG
    Serial.print("LAYER " + String(i) + " OUTPUT TENSOR: (");
    for (int j = 0; j < output->dims->size; ++j) {
      Serial.print(output->dims->data[j]);
      if (j+1 < output->dims->size) {
        Serial.print(", ");
      }
    }
    Serial.println(")");

    int numOutput = 1;
    for (int j = 0; j < output->dims->size; ++j) {
      numOutput *= output->dims->data[j];
    }
    Serial.println("LAYER " + String(i) + " OUTPUT DATA:");
    for (int j = 0; j < numOutput; ++j) {
  #ifdef ALL
      Serial.print(outputData[j]);
      Serial.print(" ");
  #else // Only print first 3 and last 3 elements
      if (j < 3) {
        Serial.print(outputData[j]);
        Serial.print(" ");
      } else if (j >= numOutput-3) {
        Serial.print(outputData[j]);
        Serial.print(" ");
      }
  #endif // ALL
    }
    Serial.println();
#endif // DEBUG

    // Set next layer's input data and size
    memcpy(inputData, outputData, outputSize);
    inputSize = outputSize;

#ifdef FOMO
    // Saves the output of the layer with multiple outputs for future use
    for (int j = 0; j < multiOutputLayersSize; ++j) {
      if (i == multiOutputLayers[j]) {
        memcpy(lastMultiOutputLayerData, outputData, outputSize);
        break;
      }
    }
#endif

    // Store inference time in seconds
    inference_times[i] = (micros() - inizio) / 1000000.0; // Convert microseconds to seconds

    Serial.println("Computed layer: " + String(i) + " Inf Time: " + String((micros() - inizio) / 1000000.0) + " s");
  }
  Serial.println("Model inference successful!");

  if (deviceRegistered) {
    // Set up message
    double timestamp = getCurrTime();
    memcpy(output_message, &timestamp, sizeof(timestamp)); // sizeof(timestamp) == sizeof(double) == 8
    int offset = sizeof(timestamp);

    memcpy(output_message+offset, &DeviceUUID, 9);
    offset += 9;

    memcpy(output_message+offset, &MessageUUID, 4);
    offset += 4;

    memcpy(output_message+offset, &offloading_layer_index, sizeof(offloading_layer_index)); // sizeof(offloading_layer_index) == sizeof(int) == 4
    offset += sizeof(offloading_layer_index);

    memcpy(output_message+offset, &output->bytes, sizeof(output->bytes)); // sizeof(output->bytes) == sizeof(size_t) == 4
    offset += sizeof(output->bytes);

    memcpy(output_message+offset, output->data.f, output->bytes);
    offset += output->bytes;

    int layers_inference_time_size = (offloading_layer_index+1)*sizeof(int);
    memcpy(output_message+offset, &layers_inference_time_size, sizeof(layers_inference_time_size)); // sizeof(layers_inference_time_size) == sizeof(int) == 4
    offset += sizeof(layers_inference_time_size);

    memcpy(output_message+offset, inference_times, sizeof(inference_times)); // sizeof(inference_times) == layers_inference_time_size
    offset += sizeof(inference_times);

    // Publish device prediction
    mqttClient.publish(model_inference_result_topic.c_str(), 0, false, output_message, offset);
    Serial.println("Published Prediction");
  }
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * GENERATE MESSAGE UUID
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void generateMessageUUID(){
  // Generate a UUID
  unsigned long seed = esp_random();
  uuid.seed(seed);
  uuid.setRandomMode();
  uuid.generate();
  String uuidStr = (String)uuid.toCharArray();
  MessageUUID = uuidStr.substring(0, 4);
  DeviceUUID = "device_01"; // + MessageUUID;
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* GET MODEL DATA FOR PREDICTION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void getModelDataForPrediction(const camera_fb_t* fb) {
  uint16_t* pixels = (uint16_t*)fb->buf;
  int n = fb->height*fb->width;
  try {
    for (int i = 0; i < n; i++) {
      uint16_t p = pixels[i];
      // Convert big endian to little endian
      p = (p << 0x8) | (p >> 0x8);
      // Convert rgb565 image to normalized array of pixels rgb
      inputBuffer[i*3] = (p >> 11) / 31.0;
      inputBuffer[i*3+1] = ((p >> 5) & 0x3f) / 63.0;
      inputBuffer[i*3+2] = (p & 0x1f) / 31.0;
    }
    Serial.println("Model input data received");
  } catch (const std::exception& e) {
    Serial.print("Error receiving model input data: ");
    Serial.println(e.what());
  }
  modelDataLoaded = true;
}

void processIncomingMessage(char* topic, char* payload, AsyncMqttClientMessageProperties properties, size_t len, size_t index, size_t total) {
  // Instantiate message
  if (index == 0) {
    input_message = new char[total+1];
  }

  // Bufferize message fragments
  for (size_t i = 0; i < len; i++) {
    input_message[index+i] = payload[i];
  }

  // Return if we don't have whole message yet
  if (index+len < total) {
    return;
  }

  // Add terminator
  input_message[total] = '\0';

  // Parse the JSON message and store it in the DynamicJsonDocument
  JsonDocument doc(&allocator);
  DeserializationError error = deserializeJson(doc, input_message);

  // Check for parsing errors
  if (error) {
    Serial.print("JSON parsing error: ");
    Serial.println(error.c_str());
    delete[] input_message;
    input_message = nullptr;
    return;
  }
  Serial.print("Received message:");
  Serial.println(topic);

  // Check if the message is for model_inference
  if (strcmp(topic, offloading_layer_topic.c_str()) == 0) {
    xTimerReset(deviceRegistrationTimer, 0);
    if (!deviceRegistered) {
      deviceRegistered = true;
      Serial.println("Device Registered");
    }
    if (doc["offloading_layer_index"] >= MAX_NUM_LAYER) {
      Serial.println("Invalid offloading layer");
      return;
    }
    best_offloading_layer_index = doc["offloading_layer_index"];
  }

  delete[] input_message;
  input_message = nullptr;
}

void dispatchCallbackMessages() {
  // Set the topics
  offloading_layer_topic = DeviceUUID + "/offloading_layer";
  input_data_topic = DeviceUUID + "/input_data";
  model_inference_result_topic = DeviceUUID + "/model_inference_result";

  // Subscribe to the topic
  mqttClient.subscribe(offloading_layer_topic.c_str(), 0);

  // Set the callback function
  mqttClient.onMessage(processIncomingMessage);
}

/*
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* REGISTER THE DEVICE ON THE EDGE
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void registerDevice(){
  deviceRegistered = false;

  generateMessageUUID();        // Generate an Identifier for the message
  dispatchCallbackMessages();   // Set the callback function for the MQTT messages

  // Generate the JSON message
  JsonDocument jsonDoc(&allocator);
  jsonDoc["timestamp"] = getCurrTimeStr();
  jsonDoc["message_id"] = MessageUUID;
  jsonDoc["device_id"] = DeviceUUID;
  jsonDoc["message_content"] = "HelloWorld!";
  // Serialize the JSON document to a string
  String jsonMessage;
  serializeJson(jsonDoc, jsonMessage);
  // Publish the JSON message to the topic
  mqttClient.publish(device_registration_topic.c_str(), 0, false, (char*)jsonMessage.c_str());
  Serial.println("Registration request sent");
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* MQTT CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void connectToMqtt() {
  Serial.println("Connecting to MQTT...");
  mqttClient.connect();
}

void disconnectFromMqtt() {
  Serial.println("Disconnecting from MQTT...");
  mqttClient.disconnect();
}

void onMqttConnect(bool sessionPresent) {
  Serial.println("Connected to MQTT.");
  mqttClient.setKeepAlive(60);
  registerDevice();             // Register the device on the edge
  xTimerStart(deviceRegistrationTimer, 0);
  xTimerStart(mqttDisconnectTimer, 0);
}

void onMqttDisconnect(AsyncMqttClientDisconnectReason reason) {
  xTimerStop(deviceRegistrationTimer, 0);
  xTimerStop(mqttDisconnectTimer, 0);
  deviceRegistered = false;
  best_offloading_layer_index = MAX_NUM_LAYER-1;
  Serial.println("Disconnected from MQTT.");
  if (WiFi.isConnected()) {
    timeConfiguration();          // Synchronize Timer - NTP server
    xTimerStart(mqttReconnectTimer, 0);
  }
}

void mqttConfiguration(){
  mqttClient.onConnect(onMqttConnect);
  mqttClient.onDisconnect(onMqttDisconnect);
  mqttClient.setServer(MQTT_SRV, MQTT_PORT);
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* WIFI CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void connectToWifi() {
  Serial.println("Connecting to WiFi...");
  WiFi.begin(SSID, PWD);
}

void WiFiEvent(WiFiEvent_t event) {
  switch(event) {
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      Serial.println("WiFi connected");
      Serial.println("IP address: ");
      Serial.println(WiFi.localIP());
      timeConfiguration();          // Synchronize Timer - NTP server
      connectToMqtt();
      break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      Serial.println("WiFi lost connection");
      xTimerStop(mqttReconnectTimer, 0); // ensures we don't reconnect to MQTT while reconnecting to WiFi
      xTimerStart(wifiReconnectTimer, 0);
      break;
  }
}

void wifiConfiguration(){
  WiFi.onEvent(WiFiEvent);
}

/* 
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* CAMERA CONFIGURATION
* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
*/
void cameraConfiguration(){
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = CAMERA_PIN_D0;
  config.pin_d1 = CAMERA_PIN_D1;
  config.pin_d2 = CAMERA_PIN_D2;
  config.pin_d3 = CAMERA_PIN_D3;
  config.pin_d4 = CAMERA_PIN_D4;
  config.pin_d5 = CAMERA_PIN_D5;
  config.pin_d6 = CAMERA_PIN_D6;
  config.pin_d7 = CAMERA_PIN_D7;
  config.pin_xclk = CAMERA_PIN_XCLK;
  config.pin_pclk = CAMERA_PIN_PCLK;
  config.pin_vsync = CAMERA_PIN_VSYNC;
  config.pin_href = CAMERA_PIN_HREF;
  config.pin_sscb_sda = CAMERA_PIN_SIOD;
  config.pin_sscb_scl = CAMERA_PIN_SIOC;
  config.pin_pwdn = CAMERA_PIN_PWDN;
  config.pin_reset = CAMERA_PIN_RESET;
  config.xclk_freq_hz = XCLK_FREQ_HZ;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_CUSTOM;
  config.jpeg_quality = 10;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
      Serial.printf("Camera init failed with error 0x%x\n", err);
      return;
  }
  Serial.println("Camera initialized successfully.");
}

/*
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * SETUP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void setup() {
  Serial.begin(115200);
  
  esp_task_wdt_init(WDT_TIMEOUT, true); // Watchdog Timer

  if (psramInit()) {
    Serial.println("The PSRAM is correctly initialized");
  } else {
    Serial.println("PSRAM does not work");
  }

  inputBuffer = (float*)ps_malloc(MAX_ELEMENTS_PER_MODEL_LAYER*sizeof(float)); 
  tensor_arena = (uint8_t*)ps_malloc(K_TENSOR_ARENA_SIZE*sizeof(uint8_t));
  output_message = (char*)ps_malloc(OUTPUT_MSG_SIZE);
#ifdef FOMO
  lastMultiOutputLayerData = (float*)ps_malloc(MAX_ELEMENTS_PER_MODEL_LAYER*sizeof(float)); 
#endif // FOMO

  cameraConfiguration();        // Camera OV2640

  mqttReconnectTimer = xTimerCreate("mqttTimer", pdMS_TO_TICKS(2000), pdFALSE, (void*)0, reinterpret_cast<TimerCallbackFunction_t>(connectToMqtt));
  mqttDisconnectTimer = xTimerCreate("mqttDisconnectTimer", pdMS_TO_TICKS(600000), pdFALSE, (void*)0, reinterpret_cast<TimerCallbackFunction_t>(disconnectFromMqtt));
  wifiReconnectTimer = xTimerCreate("wifiTimer", pdMS_TO_TICKS(2000), pdFALSE, (void*)0, reinterpret_cast<TimerCallbackFunction_t>(connectToWifi));
  deviceRegistrationTimer = xTimerCreate("registrationTimer", pdMS_TO_TICKS(30000), pdTRUE, (void*)0, reinterpret_cast<TimerCallbackFunction_t>(registerDevice));
  wifiConfiguration();          // Wi-Fi Connection
  mqttConfiguration();          // MQTT
  connectToWifi();
  delay(5000);
}

/* 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * LOOP 
 * ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void loop() {
  // Capture a frame
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
      Serial.println("Failed to capture image");
      return;
  }
  Serial.printf("Captured image with size: %u bytes\n", fb->len);

  // Send captured image
  if (deviceRegistered) {
    mqttClient.publish(input_data_topic.c_str(), 0, false, (char*)fb->buf, fb->len);
    Serial.println("Captured image sent");
  }

  // Parse input for model
  getModelDataForPrediction(fb);

  // Run inference and send results
  if (modelDataLoaded) {
    runNeuralNetworkLayer(best_offloading_layer_index, inputBuffer);
    modelDataLoaded = false;
  }

  // Free the frame buffer
  esp_camera_fb_return(fb);
}