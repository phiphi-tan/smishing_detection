import argparse
import json
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders.csv_loader import CSVLoader
from pydantic import BaseModel

from .agents import get_agents
from .schemas import JudgeOutput
from .tools import set_claims_file

DATA_FILE = "./data/D2.csv"
OUTPUT_ROOT = Path("./data/output")
CLAIMS_FILENAME = "claims.ndjson"
OUTPUT_FILENAME = "output.ndjson"
VERIFIABILITY_ORDER = {
    "Publicly-Verifiable": 0,
    "Recipient-Verifiable": 1,
    "Message-Internally-Verifiable": 2,
    "Unverifiable": 3,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5.4-nano")
    parser.add_argument("--model_helper", type=str, default="gpt-5-mini")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based row index to start processing from")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process")
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start-index must be greater than or equal to 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than or equal to 1")

    return args


def load_messages(file_path=DATA_FILE):
    loader = CSVLoader(
        file_path=file_path,
        encoding="utf-8",
        metadata_columns=["scam_id", "raw_text"],
    )
    return loader.load()


def invoke_agent(agent, content):
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": content}
        ]
    })
    return response["messages"][-1].content

# safe conversion to JSON to write to .ndjson files
def to_jsonable(data):
    if isinstance(data, BaseModel):
        return data.model_dump(by_alias=True)
    if isinstance(data, list):
        return [to_jsonable(item) for item in data]
    if isinstance(data, dict):
        return {key: to_jsonable(value) for key, value in data.items()}
    return data


def write_ndjson_line(file_path, data):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(data), ensure_ascii=False) + "\n")

def extract_claims(phishing_text, claim_agent):
    claims_output = claim_agent.invoke(phishing_text)
    claims = claims_output.claims
    print(f"[INITIAL CLAIM RESPONSE]: \n {json.dumps(to_jsonable(claims), indent=4)}")
    return claims


def filter_claims(initial_claims, filter_agent):
    filtered_claims_output = filter_agent.invoke(json.dumps(to_jsonable(initial_claims), ensure_ascii=False))
    filtered_claims = filtered_claims_output.claims
    filtered_claims = sorted(
        filtered_claims,
        key=lambda claim: VERIFIABILITY_ORDER.get(claim.Verifiability, 99),
    )
    print(f"[FILTERED CLAIM RESPONSE]: \n {json.dumps(to_jsonable(filtered_claims), indent=4)}")
    return filtered_claims


def collect_evidence(claim, scam_id, evidence_agent):
    evidence_prompt = f"""
        Claim: {claim}
        scam_id: {scam_id}
        """
    claims_evidence = invoke_agent(evidence_agent, evidence_prompt)
    print(f"[claims_evidence]:\n{claims_evidence}")
    return claims_evidence


def judge_claim(claims_evidence, judge_agent):
    final_claim = judge_agent.invoke(claims_evidence)
    print(f"[final_claim]:\n{json.dumps(to_jsonable(final_claim), indent=4)}")
    return final_claim


def is_unverifiable_claim(claim):
    return claim.Verifiability.lower() == "unverifiable"


def build_skipped_claim_output(claim):
    return JudgeOutput(
        raw_claim=claim.raw_claim,
        parsed_claim=claim.parsed_claim,
        High_Value_Type=claim.High_Value_Type,
        Verifiability=claim.Verifiability,
        evidence_required=None,
        evidence_collected=[],
        Verdict="Skipped",
        **{"Extra Evidence Needed": None},
    )



def iter_selected_rows(data, start_index, limit):
    processed_count = 0

    for row_index, row in enumerate(data):
        if row_index < start_index:
            continue

        yield row

        processed_count += 1
        if limit is not None and processed_count >= limit:
            break


def create_run_output_paths(output_root=OUTPUT_ROOT):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    return {
        "run_dir": run_dir,
        "claims": run_dir / CLAIMS_FILENAME,
        "output": run_dir / OUTPUT_FILENAME,
    }

# Main logic for processing messages
def process_message(row, claim_agent, filter_agent, evidence_agent, judge_agent, output_paths):
    phishing = row.metadata.get("raw_text")
    scam_id = row.metadata.get("scam_id")
    print(f"Scam id: {scam_id}: {phishing}")

    # per-message actions
    initial_claims = extract_claims(phishing, claim_agent)
    filtered_claims = filter_claims(initial_claims, filter_agent)

    write_ndjson_line(output_paths["claims"], filtered_claims)

    # per-claim actions
    claim_outputs = process_claims(filtered_claims, scam_id, evidence_agent, judge_agent)
    write_ndjson_line(output_paths["output"], claim_outputs)

# Main logic for processing claims (downstream task from above process_message)
def process_claims(claims, scam_id, evidence_agent, judge_agent):
    claim_outputs = []

    for claim in claims:
        print(f"[CLAIM]: {claim}")
        if is_unverifiable_claim(claim):
            claim_outputs.append(build_skipped_claim_output(claim))
            continue

        claims_evidence = collect_evidence(claim, scam_id, evidence_agent)
        final_claim = judge_claim(claims_evidence, judge_agent)
        claim_outputs.append(final_claim)
        if final_claim.Verdict == "False":
            break

    return claim_outputs


def main():
    args = parse_args()
    data = load_messages()
    output_paths = create_run_output_paths()
    set_claims_file(output_paths["claims"])
    print(f"Writing run outputs to: {output_paths['run_dir']}")

    claim_agent, filter_agent, evidence_agent, judge_agent = get_agents(
        claim_model=args.model_helper,
        filter_model=args.model_helper,
        evidence_model=args.model,
        judge_model=args.model_helper,
    )

    for row in iter_selected_rows(data, args.start_index, args.limit):
        try:
            process_message(row, claim_agent, filter_agent, evidence_agent, judge_agent, output_paths)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)


if __name__ == "__main__":
    main()
