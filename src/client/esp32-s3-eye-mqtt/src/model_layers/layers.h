#include "layer_0.h"
#include "layer_1.h"
#include "layer_2.h"
#include "layer_3.h"
#include "layer_4.h"
#include "layer_5.h"
#include "layer_6.h"
#include "layer_7.h"
#include "layer_8.h"
#include "layer_9.h"
#include "layer_10.h"
#include "layer_11.h"
#include "layer_12.h"
#include "layer_13.h"
#include "layer_14.h"
#include "layer_15.h"
#include "layer_16.h"
#include "layer_17.h"
#include "layer_18.h"
#include "layer_19.h"
#include "layer_20.h"
#include "layer_21.h"
#include "layer_22.h"
#include "layer_23.h"
#include "layer_24.h"
#include "layer_25.h"
#include "layer_26.h"
#include "layer_27.h"
#include "layer_28.h"
#include "layer_29.h"
#include "layer_30.h"
#include "layer_31.h"
#include "layer_32.h"
#include "layer_33.h"
#include "layer_34.h"
#include "layer_35.h"
#include "layer_36.h"
#include "layer_37.h"
#include "layer_38.h"
#include "layer_39.h"
#include "layer_40.h"
#include "layer_41.h"
#include "layer_42.h"
#include "layer_43.h"
#include "layer_44.h"
#include "layer_45.h"
#include "layer_46.h"
#include "layer_47.h"
#include "layer_48.h"
#include "layer_49.h"
#include "layer_50.h"
#include "layer_51.h"
#include "layer_52.h"
#include "layer_53.h"
#include "layer_54.h"
#include "layer_55.h"
#include "layer_56.h"
#include "layer_57.h"
#include "layer_58.h"
#define LOAD_LAYER() if(layer_name.equals("layer_0"))model = tflite::GetModel(layer_0);\
if(layer_name.equals("layer_1"))model = tflite::GetModel(layer_1);\
if(layer_name.equals("layer_2"))model = tflite::GetModel(layer_2);\
if(layer_name.equals("layer_3"))model = tflite::GetModel(layer_3);\
if(layer_name.equals("layer_4"))model = tflite::GetModel(layer_4);\
if(layer_name.equals("layer_5"))model = tflite::GetModel(layer_5);\
if(layer_name.equals("layer_6"))model = tflite::GetModel(layer_6);\
if(layer_name.equals("layer_7"))model = tflite::GetModel(layer_7);\
if(layer_name.equals("layer_8"))model = tflite::GetModel(layer_8);\
if(layer_name.equals("layer_9"))model = tflite::GetModel(layer_9);\
if(layer_name.equals("layer_10"))model = tflite::GetModel(layer_10);\
if(layer_name.equals("layer_11"))model = tflite::GetModel(layer_11);\
if(layer_name.equals("layer_12"))model = tflite::GetModel(layer_12);\
if(layer_name.equals("layer_13"))model = tflite::GetModel(layer_13);\
if(layer_name.equals("layer_14"))model = tflite::GetModel(layer_14);\
if(layer_name.equals("layer_15"))model = tflite::GetModel(layer_15);\
if(layer_name.equals("layer_16"))model = tflite::GetModel(layer_16);\
if(layer_name.equals("layer_17"))model = tflite::GetModel(layer_17);\
if(layer_name.equals("layer_18"))model = tflite::GetModel(layer_18);\
if(layer_name.equals("layer_19"))model = tflite::GetModel(layer_19);\
if(layer_name.equals("layer_20"))model = tflite::GetModel(layer_20);\
if(layer_name.equals("layer_21"))model = tflite::GetModel(layer_21);\
if(layer_name.equals("layer_22"))model = tflite::GetModel(layer_22);\
if(layer_name.equals("layer_23"))model = tflite::GetModel(layer_23);\
if(layer_name.equals("layer_24"))model = tflite::GetModel(layer_24);\
if(layer_name.equals("layer_25"))model = tflite::GetModel(layer_25);\
if(layer_name.equals("layer_26"))model = tflite::GetModel(layer_26);\
if(layer_name.equals("layer_27"))model = tflite::GetModel(layer_27);\
if(layer_name.equals("layer_28"))model = tflite::GetModel(layer_28);\
if(layer_name.equals("layer_29"))model = tflite::GetModel(layer_29);\
if(layer_name.equals("layer_30"))model = tflite::GetModel(layer_30);\
if(layer_name.equals("layer_31"))model = tflite::GetModel(layer_31);\
if(layer_name.equals("layer_32"))model = tflite::GetModel(layer_32);\
if(layer_name.equals("layer_33"))model = tflite::GetModel(layer_33);\
if(layer_name.equals("layer_34"))model = tflite::GetModel(layer_34);\
if(layer_name.equals("layer_35"))model = tflite::GetModel(layer_35);\
if(layer_name.equals("layer_36"))model = tflite::GetModel(layer_36);\
if(layer_name.equals("layer_37"))model = tflite::GetModel(layer_37);\
if(layer_name.equals("layer_38"))model = tflite::GetModel(layer_38);\
if(layer_name.equals("layer_39"))model = tflite::GetModel(layer_39);\
if(layer_name.equals("layer_40"))model = tflite::GetModel(layer_40);\
if(layer_name.equals("layer_41"))model = tflite::GetModel(layer_41);\
if(layer_name.equals("layer_42"))model = tflite::GetModel(layer_42);\
if(layer_name.equals("layer_43"))model = tflite::GetModel(layer_43);\
if(layer_name.equals("layer_44"))model = tflite::GetModel(layer_44);\
if(layer_name.equals("layer_45"))model = tflite::GetModel(layer_45);\
if(layer_name.equals("layer_46"))model = tflite::GetModel(layer_46);\
if(layer_name.equals("layer_47"))model = tflite::GetModel(layer_47);\
if(layer_name.equals("layer_48"))model = tflite::GetModel(layer_48);\
if(layer_name.equals("layer_49"))model = tflite::GetModel(layer_49);\
if(layer_name.equals("layer_50"))model = tflite::GetModel(layer_50);\
if(layer_name.equals("layer_51"))model = tflite::GetModel(layer_51);\
if(layer_name.equals("layer_52"))model = tflite::GetModel(layer_52);\
if(layer_name.equals("layer_53"))model = tflite::GetModel(layer_53);\
if(layer_name.equals("layer_54"))model = tflite::GetModel(layer_54);\
if(layer_name.equals("layer_55"))model = tflite::GetModel(layer_55);\
if(layer_name.equals("layer_56"))model = tflite::GetModel(layer_56);\
if(layer_name.equals("layer_57"))model = tflite::GetModel(layer_57);\
if(layer_name.equals("layer_58"))model = tflite::GetModel(layer_58);\
