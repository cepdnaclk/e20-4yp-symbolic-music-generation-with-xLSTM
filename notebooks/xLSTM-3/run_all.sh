#!/bin/bash
# run_all.sh — Master script for xLSTM-3 Generation & Evaluation
# Usage: ./notebooks/xLSTM-3/run_all.sh

set -e

# Path to the project root
PROJECT_ROOT="/scratch1/e20-fyp-xlstm-music-generation/e20fyptemp1/fyp-musicgen"
cd "$PROJECT_ROOT"

echo "================================================================="
echo " xLSTM-3: MUSIC GENERATION & EVALUATION PIPELINE"
echo "================================================================="

# 1. Handle optional config argument
CONFIG_FILE=$1
CONFIG_ARG=""
if [ -n "$CONFIG_FILE" ]; then
    echo "Using custom config: $CONFIG_FILE"
    CONFIG_ARG="--config $CONFIG_FILE"
else
    echo "Using default config: config.py"
fi

# 2. Run Generation (resumable)
echo -e "\n[1/2] Starting Generation..."
conda run -n xlstm python3 notebooks/xLSTM-3/generate/run_generation.py $CONFIG_ARG

# 3. Run Evaluation
echo -e "\n[2/2] Starting Evaluation & Plotting..."
conda run -n xlstm python3 notebooks/xLSTM-3/evaluate/run_evaluation.py $CONFIG_ARG

echo -e "\n================================================================="
echo " PIPELINE COMPLETE"
echo " Results:  notebooks/xLSTM-3/results/xlstm_512d_4096ctx_ck66k/"
echo " MIDI:     .../midi/"
echo " Plots:    .../metrics/plots/"
echo "================================================================="
