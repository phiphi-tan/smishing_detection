import json
from langchain_community.document_loaders.csv_loader import CSVLoader
from agents import agent1, agent2, agent3

output_file = "./data/test_output.ndjson"
agent1_output = './data/agent1_test.ndjson'

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
    print(f'Scam id: {scam_id}: {phishing}')

    response1 = agent1.invoke({
        'messages': [
            {"role": "user", "content": phishing}
        ]
    })

    # print(f"response 1: {response1}")

    agent1_response = response1['messages'][-1].content
    print(f"agent 1: {agent1_response}")

    try:
        claims = json.loads(agent1_response)
        print(f'[AGENT1 RESPONSE]: \n {json.dumps(claims, indent=4)}')

        with open(agent1_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(claims, ensure_ascii=False) + "\n")

        # with open('../data/agent1.ndjson') as f:
        #     claims = ndjson.load(f)
        #     claims = claims[-1] # latest row only

        all_claims = []

        for claim in claims:
            print(claim)
            
            response2 = agent2.invoke({
                'messages': [
                    {"role": "user",
                     "content": f"""
                        Claim: {claim}
                        scam_id: {scam_id}
                        """}
                ]      
            })
            agent2_response = response2['messages'][-1].content 
            # print(f'[AGENT2 RESPONSE]: \n {json.dumps(json.loads(agent2_response), indent=4)}')

            response3 = agent3.invoke({
                'messages': [
                    {"role": "user", "content": agent2_response}
                ]
            })

            agent3_response = response3['messages'][-1].content 
            agent3_claim = json.loads(agent3_response)
            print(f'[AGENT3 RESPONSE]: \n {json.dumps(agent3_claim, indent=4)}')

            all_claims.append(agent3_claim)

        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(all_claims, ensure_ascii=False) + "\n")

    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
    

    i += 1
    if i >= 0:
        break

