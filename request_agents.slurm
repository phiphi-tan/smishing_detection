#!/bin/bash
#SBATCH --job-name=speedup         
#SBATCH --output=logs/agent_output_%j.log         
#SBATCH --error=logs/agent_error_%j.log           
#SBATCH --partition=normal
#SBATCH --time=00:20:00            
#SBATCH --cpus-per-task=4     

mkdir -p logs

source qwen/bin/activate

# ---------------------------
# Poll until vLLM API is ready
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