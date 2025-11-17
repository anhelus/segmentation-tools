#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status (crucial for batch jobs)
set -e

# --- Configuration ---
PYTHON_SCRIPT="optim.py"
# Define the CONFIG_PATH once for easy management
CONFIG_PATH="/mnt/c/Datasets/OlivePG/optim_dataset/config.yaml"

echo "================================================="
echo " STARTING THRESHOLD OPTIMIZATION BATCH RUN"
echo " Configuration file used: ${CONFIG_PATH}"
echo "================================================="

# Function to execute a Python optimization command with logging
run_optim() {
    local MODEL_TYPE=$1
    local OUTPUT_NAME=$2
    # Shift positional arguments to isolate the model-specific flags
    shift 2
    local ARGS="$@"
    
    echo "--- RUNNING: ${OUTPUT_NAME} ---"
    echo "Model: ${MODEL_TYPE} | Args: ${ARGS}"
    
    # Execute the Python script
    # NOTE: The model type for YOLO-World is corrected to 'yolo_world' to match Python's convention.
    python3 "${PYTHON_SCRIPT}" "${MODEL_TYPE}" "${CONFIG_PATH}" --output_name "${OUTPUT_NAME}" ${ARGS}
    
    echo "--- ${OUTPUT_NAME} completed. Check ${OUTPUT_NAME} directory. ---"
    echo ""
}

# ----------------------GROUNDING DINO Models (Use 'dino')----------------------
echo "#################################################"
echo " STARTING GROUNDING DINO OPTIMIZATION RUNS"
echo "#################################################"

run_optim dino \
    optim_result_GDINO_TINY \
    --model GDINO-TINY

run_optim dino \
    optim_result_GDINO_BASE \
    --model GDINO-BASE

# --------------------YOLO-WORLD Models (Use 'yolo_world')--------------------
echo "#################################################"
echo " STARTING YOLO-WORLD OPTIMIZATION RUNS"
echo "#################################################"

# YOLO-World S, single prompt ("olive")
run_optim yolo_world \
    optim_result_YOLO_S_WORLD_V1 \
    --model YOLO-S-WORLD --prompts "olive"

# YOLO-World X, single prompt ("olive")
run_optim yolo_world \
    optim_result_YOLO_X_WORLD_V1 \
    --model YOLO-X-WORLD --prompts "olive"

# YOLO-World S, two prompts ("olive" "single olive")
run_optim yolo_world \
    optim_result_YOLO_S_WORLD_V2 \
    --model YOLO-S-WORLD --prompts "olive" "single olive"

# YOLO-World X, different prompt ("single olive")
# Adjusted output name for consistency with model size X
run_optim yolo_world \
    optim_result_YOLO_X_WORLD_V2 \
    --model YOLO-X-WORLD --prompts "single olive"

echo "================================================="
echo " ALL THRESHOLD OPTIMIZATION RUNS FINISHED SUCCESSFULLY"
echo "================================================="