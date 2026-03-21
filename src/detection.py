import json
import csv
import argparse

from langchain_community.document_loaders.csv_loader import CSVLoader
from agents import create_agents

from qwen_agent.utils.output_beautify import typewriter_print

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-FP8")
args = parser.parse_args()

claim_agent, filter_agent, evidence_agent, judge_agent = create_agents(args.model)


output_file = "./data/test_output.ndjson"
claim_agent_output = './data/claim_agent_test.ndjson'

loader = CSVLoader(
    file_path="./data/D2.csv",
    encoding="utf-8",
    metadata_columns=['scam_id', 'raw_text']
)
data = loader.load() # into a langchain document

i = 0

for row in data:
    if i < -1:
        i += 1
        continue

    phishing = row.metadata.get('raw_text')
    scam_id = row.metadata.get('scam_id')
    print(f'\nScam id: {scam_id}: {phishing}')

    message_1 = [{'role': 'user', 'content': phishing}]
    response = []
    response_plain_text = ''
    for response in claim_agent.run(messages=message_1):
        response_plain_text = typewriter_print(response, response_plain_text)
    response_claim_agent = response
    output_claim = response_claim_agent[-1]['content']
    # print(f"\nclaim output: {output_claim}")

    message_filter_agent = [{'role': 'user', 'content': output_claim}]
    response = []
    response_plain_text = ''
    for response in filter_agent.run(messages=message_filter_agent):
        response_plain_text = typewriter_print(response, response_plain_text)
    response_filter_agent = response
    print(f'response filter agent: {response_filter_agent}')
    output_filter = response_filter_agent[-1]['content']
    # print(f'\nfilter output: {output_filter}')

    try:
        claims = json.loads(output_filter)
        # print(f'\n[response_filter RESPONSE]: \n {json.dumps(claims, indent=4)}')

        with open(claim_agent_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(claims, ensure_ascii=False) + "\n")

        # with open('../data/claim_agent.ndjson') as f:
        #     claims = ndjson.load(f)
        #     claims = claims[-1] # latest row only

        all_claims = []

        for claim in claims:
            print(f"\nClaim: {claim}")
            
            message_evidence_agent = [{"role": "user",
                     "content": f"""
                        Claim: {claim}
                        scam_id: {scam_id}
                        """}]
            response = []
            response_plain_text = ''
            for response in evidence_agent.run(messages=message_evidence_agent):
                response_plain_text = typewriter_print(response, response_plain_text)
            response_evidence_agent = response
            # print(response_evidence_agent)
            output_evidence_agent = response_evidence_agent[-1]['content']
            # print(f'\noutput_evidence_agent: {output_evidence_agent}')
            
            message_judgement_agent = [{'role': 'user', 'content': output_evidence_agent}]
            response = []
            response_plain_text = ''
            for response in judge_agent.run(messages=message_judgement_agent):
                response_plain_text = typewriter_print(response, response_plain_text)
            response_3 = response
            # print(response_3)
            output_3 = response_3[-1]['content']
            # print(f'\noutput_3: {output_3}')

            agent3_claim = json.loads(output_3)
            # print(f'\n[AGENT3 RESPONSE]: \n {json.dumps(agent3_claim, indent=4)}')
            all_claims.append(agent3_claim)

        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(all_claims, ensure_ascii=False) + "\n")

    except json.JSONDecodeError as e:
        print("\nFailed to parse JSON:", e)
    

    i += 1
    if i >= 2:
        break

