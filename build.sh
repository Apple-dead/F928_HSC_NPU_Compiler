#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

#========MODEL========
MODEL_NAME="yolov2_fixed.pth"

#========PARAMETERS========
BIAS_1_LENGTH=256
BIAS_1_MOVE=512
#===========DATA=============
IMAGE_PATH="./data/image.jpg"
WEIGHT_1_PATH="./data/model_params/layer1_0_weight.txt"
BIAS_1_PATH="./data/model_params/layer1_0_bias.txt"
INSTR_PATH="./data/instr.txt"

#===========COE=============
IMAGE_COE_PATH="./coe/image.coe"
WEIGHT_1_COE_PATH="./coe/layer1_weight.coe"
BIAS_1_COE_PATH="./coe/layer1_bias.coe"
INSTR_COE_PATH="./coe/instr.coe"

#===========TARGET COE=============
TARGET_COE_PATH="./target/all.coe"

python ./python/extract_pth_params.py "./model/$MODEL_NAME" "./data/model_params/"

#image to coe
python ./python/image_to_bram_coe.py "$IMAGE_PATH" "$IMAGE_COE_PATH"

#weight to coe
python ./python/weight_to_bram_coe.py "$WEIGHT_1_PATH" "$WEIGHT_1_COE_PATH"

#bias to coe
python ./python/bias_to_bram.py -length "$BIAS_1_LENGTH" -move "$BIAS_1_MOVE" "$BIAS_1_PATH" "$BIAS_1_COE_PATH"

#instr to coe
python ./python/instr_txt_to_bram_coe.py "$INSTR_PATH" "$INSTR_COE_PATH"

#merge coe
python ./python/merge.py "$IMAGE_COE_PATH" "$WEIGHT_1_COE_PATH" "$BIAS_1_COE_PATH" "$INSTR_COE_PATH" "$TARGET_COE_PATH"
