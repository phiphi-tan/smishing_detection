#!/bin/bash
#SBATCH --job-name=speedup         
#SBATCH --output=logs/output_%j.log         
#SBATCH --error=logs/error_%j.log           
#SBATCH --gres=gpu:h100-47:1
#SBATCH --partition=gpu
#SBATCH --time=00:20:00            
#SBATCH --cpus-per-task=4     

mkdir -p logs

source .venv/bin/activate

# ---------------------------
# 2. Set environment variables for VLLM as OpenAI API
# ---------------------------
# export OPENAI_API_BASE="http://127.0.0.1:8000/v1"

# ---------------------------
# 3. Start VLLM HTTP API in the background
# ---------------------------
# vllm serve Qwen/Qwen3-8B --chat-template ./data/qwen3_nonthinking.jinja --enable-auto-tool-choice --tool-call-parser hermes --port 8000 &
# VLLM_PID=$!

# vllm serve Qwen/Qwen3-14B-FP8 --chat-template ./data/qwen3_nonthinking.jinja --enable-auto-tool-choice --tool-call-parser hermes --port 8000 &
# VLLM_PID=$!

vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --enable-auto-tool-choice --tool-call-parser hermes --port 8000 &
VLLM_PID=$!

# vllm serve Qwen/Qwen3.5-9B --language-model-only --enable-auto-tool-choice --reasoning-parser qwen3 --tool-call-parser qwen3_coder --port 8000 &
# VLLM_PID=$!

echo "Started VLLM API with PID $VLLM_PID"

# ---------------------------
# 4. Poll until vLLM API is ready
# ---------------------------
TIMEOUT=300       # max seconds to wait
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
python -u ./src/detection.py

# ---------------------------
# 6. Cleanup
# ---------------------------
echo "Stopping VLLM API..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
echo "Job finished."