from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datasets import load_dataset
from agents.agent1 import Agent1

dataset_dict = load_dataset("csv", data_files="../data/D2.csv")
dataset = dataset_dict['train']
print(dataset)

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

agent1 = Agent1(model, tokenizer)

output_file = "../data/smishing_output.ndjson"
batch_size = 1
for i in tqdm(range(0, len(dataset), batch_size), total=(len(dataset) + batch_size - 1) // batch_size):
    batch = dataset[i:i+batch_size]
    entries = agent1.parse_input(batch)

    with open(output_file, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    break