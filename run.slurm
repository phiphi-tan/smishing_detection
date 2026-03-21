#!/bin/bash
#SBATCH --job-name=speedup         
#SBATCH --output=logs/output_%j.log         
#SBATCH --error=logs/error_%j.log           
#SBATCH --gres=gpu:h100-47:2
#SBATCH --partition=gpu
#SBATCH --time=00:20:00            
#SBATCH --cpus-per-task=4     

mkdir -p logs

source qwen/bin/activate

# MODEL="Qwen/Qwen3-32B-AWQ"
# MODEL="Qwen/Qwen3-30B-A3B-FP8"
# MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# MODEL="Qwen/Qwen3-235B-A22B-GPTQ-Int4"

# MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
# ---------------------------
# 3. Start VLLM HTTP API in the background
# ---------------------------

# for Qwen3 Models
vllm serve $MODEL \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --chat-template ./data/qwen3_nonthinking.jinja \
    --port 8000 &
VLLM_PID=$!

# vllm serve $MODEL \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --max-model-len 4096 \
#     --max-num-seqs 8 \
#     --chat-template ./data/qwen3_nonthinking.jinja \
#     --port 8000 &
# VLLM_PID=$!

# for Qwen3.5 models
# vllm serve $MODEL \
#   --enable-expert-parallel \
#   --language-model-only \
#   --reasoning-parser qwen3 \
#   --enable-prefix-caching \
#   --max-model-len 4096 \
#   --max-num-seqs 8 \
#   --max-num-batched-tokens 4096 \
#   --enforce-eager \
#   --port 8000 &
# VLLM_PID=$!

echo "Started VLLM API with PID $VLLM_PID"

# ---------------------------
# 4. Poll until vLLM API is ready
# ---------------------------
TIMEOUT=400       # max seconds to wait
INTERVAL=30       # seconds between checks
ELAPSED=0

echo "Waiting for vLLM to be ready..."
while ! curl -s -f http://127.0.0.1:8000/v1/models >/dev/null; do
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "checking vLLM server... elapsed time: $ELAPSED"

    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "vLLM did not start within $TIMEOUT seconds. Ending program"
        kill $VLLM_PID
        exit 1
    fi
done
echo "vLLM API is ready!"


# ---------------------------
# 5. Run your LangChain agents
# ---------------------------
python -u ./src/detection.py --model "$MODEL"

# ---------------------------
# 6. Cleanup
# ---------------------------
echo "Stopping VLLM API..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "Job finished."