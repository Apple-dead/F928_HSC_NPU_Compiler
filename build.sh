#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

#===========DATA=============
INSTR_PATH="./data/instr.txt"

#===========COE=============
INSTR_COE_PATH="./coe/instr.coe"

#===========TARGET COE=============
TARGET_COE_PATH="./target/all.coe"

clean() {
  echo "[CLEAN] remove generated files"
  rm -rf \
    ./data/model_params \
    ./data/tmp_regression \
    ./data/model_ir.json \
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

distclean() {
  clean
  echo "[DISTCLEAN] remove regression error logs"
  rm -f ./test/error/*.log ./test/error/*.log.txt ./test/error/*.txt
  echo "[DISTCLEAN] done"
}

if [ "${1:-}" = "clean" ]; then
  clean
  exit 0
fi

if [ "${1:-}" = "distclean" ]; then
  distclean
  exit 0
fi

if [ "$#" -ne 0 ]; then
  echo "Usage: ./build.sh [clean|distclean]" >&2
  exit 1
fi

#generate model IR / parameters / memory plan / instructions
python ./python/generate_model_ir.py
python ./python/generate_memory_plan.py
python ./python/generate_instr.py

#interleaved weight/bias parameter coe
python ./python/params_to_bram_coe.py --memory-plan ./data/memory_plan.json --model-params ./data/model_params --out-dir ./coe

#instr to coe
python ./python/instr_txt_to_bram_coe.py "$INSTR_PATH" "$INSTR_COE_PATH"

#merge coe
python ./python/merge.py --memory-plan ./data/memory_plan.json "$TARGET_COE_PATH"
