#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

#========MODEL========
MODEL_NAME="yolov2_fixed.pth"
MODEL_PY_NAME="yolov2_14layer_quantized.py"

#===========DATA=============
IMAGE_PATH="./data/image.jpg"
INSTR_PATH="./data/instr.txt"

#===========COE=============
IMAGE_COE_PATH="./coe/image.coe"
INSTR_COE_PATH="./coe/instr.coe"

#===========TARGET COE=============
TARGET_COE_PATH="./target/all.coe"

clean() {
  echo "[CLEAN] remove generated files"
  rm -rf \
    ./data/model_params \
    ./data/memory_plan.json \
    ./data/instr.asm \
    ./data/instr.txt \
    ./coe/layer*_params.coe \
    ./coe/instr.coe \
    ./target/all.coe \
    ./target/all.coe.map.txt
  find . -type d -name '__pycache__' -prune -exec rm -rf {} +
  echo "[CLEAN] done"
}

if [ "${1:-}" = "clean" ]; then
  clean
  exit 0
fi

if [ "$#" -ne 0 ]; then
  echo "Usage: ./build.sh [clean]" >&2
  exit 1
fi

python ./python/extract_pth_params.py "./model/$MODEL_NAME" "./data/model_params/"

#generate memory plan / instructions
python ./python/generate_memory_plan.py "$MODEL_PY_NAME"
python ./python/generate_instr.py

#image to coe
#python ./python/image_to_bram_coe.py "$IMAGE_PATH" "$IMAGE_COE_PATH"

#interleaved weight/bias parameter coe
python ./python/params_to_bram_coe.py --memory-plan ./data/memory_plan.json --model-params ./data/model_params --out-dir ./coe

#instr to coe
python ./python/instr_txt_to_bram_coe.py "$INSTR_PATH" "$INSTR_COE_PATH"

#merge coe
python ./python/merge.py --memory-plan ./data/memory_plan.json "$TARGET_COE_PATH"
