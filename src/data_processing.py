import argparse
import json
from collections import defaultdict
from pathlib import Path
from itertools import zip_longest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"
CSV_FILE = PROJECT_ROOT / "data" / "D2.csv"
CLAIMS_FILENAME = "claims.ndjson"
OUTPUT_FILENAME = "output.ndjson"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run output directory to summarize. Defaults to the latest timestamped directory in data/output.",
    )
    return parser.parse_args()


def find_latest_run_dir(output_root=OUTPUT_ROOT):
    run_dirs = [
        path for path in output_root.iterdir()
        if path.is_dir() and (path / OUTPUT_FILENAME).exists()
    ]
    if not run_dirs and (output_root / OUTPUT_FILENAME).exists():
        return output_root
    if not run_dirs:
        raise FileNotFoundError(f"No output files found in {output_root}")
    return max(run_dirs, key=lambda path: path.name)


def get_output_file(run_dir):
    output_file = run_dir / OUTPUT_FILENAME
    if not output_file.exists():
        raise FileNotFoundError(f"Output file not found: {output_file}")
    return output_file


def get_claims_file(run_dir):
    claims_file = run_dir / CLAIMS_FILENAME
    if not claims_file.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_file}")
    return claims_file


def get_claim_verifiability(claim):
    return claim.get("Verifiability", "Unknown").strip().lower()


def build_verifiability_lookup(filtered_claims):
    return {
        claim.get("raw_claim"): get_claim_verifiability(claim)
        for claim in filtered_claims
    }


def get_output_claim_verifiability(output_claim, verifiability_lookup):
    verifiability = get_claim_verifiability(output_claim)
    if verifiability != "unknown":
        return verifiability
    return verifiability_lookup.get(output_claim.get("raw_claim"), "unknown")


def load_ndjson_rows(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def calculate_stats(output_file, claims_file, csv_file=CSV_FILE):
    verifiability_stats = defaultdict(lambda: defaultdict(int))
    verifiability_claim_totals = defaultdict(int)
    type_stats = defaultdict(lambda: defaultdict(int))
    verifiability_type_stats = defaultdict(lambda: defaultdict(int))
    message_verifiability_stats = defaultdict(int)
    message_verifiability_claim_totals = defaultdict(int)

    total_claims = 0
    total_messages = 0
    messages_with_false = 0
    messages_with_extra_evidence = 0
    messages_with_true = 0
    other_messages = 0

    df = pd.read_csv(csv_file)

    output_rows = load_ndjson_rows(output_file)
    claims_rows = load_ndjson_rows(claims_file)

    for idx, (claims, filtered_claims) in enumerate(zip_longest(output_rows, claims_rows, fillvalue=[])):
        total_messages += 1
        verdicts = [claim.get("Verdict", "").lower() for claim in claims]
        verifiability_lookup = build_verifiability_lookup(filtered_claims)
        message_verifiability_labels = {
            get_claim_verifiability(claim)
            for claim in filtered_claims
        }

        if any(v == "false" for v in verdicts):
            messages_with_false += 1
        elif verdicts and all(v == "extra evidence needed" for v in verdicts):
            messages_with_extra_evidence += 1
        elif verdicts and all(v == "true" for v in verdicts):
            messages_with_true += 1
        else:
            other_messages += 1

        for verifiability in message_verifiability_labels:
            message_verifiability_stats[verifiability] += 1

        msg_type = df.loc[idx, "message_type"] if idx in df.index else "Unknown"

        for claim in filtered_claims:
            total_claims += 1
            verifiability = get_claim_verifiability(claim)
            verifiability_claim_totals[verifiability] += 1
            verifiability_type_stats[msg_type][verifiability] += 1
            message_verifiability_claim_totals[verifiability] += 1

        for claim in claims:
            verifiability = get_output_claim_verifiability(claim, verifiability_lookup)
            verdict = claim.get("Verdict", "Unknown").strip().lower()
            verifiability_stats[verifiability][verdict] += 1
            type_stats[msg_type][verdict] += 1

    return {
        "verifiability_stats": verifiability_stats,
        "verifiability_claim_totals": verifiability_claim_totals,
        "type_stats": type_stats,
        "verifiability_type_stats": verifiability_type_stats,
        "message_verifiability_stats": message_verifiability_stats,
        "message_verifiability_claim_totals": message_verifiability_claim_totals,
        "total_claims": total_claims,
        "total_messages": total_messages,
        "messages_with_false": messages_with_false,
        "messages_with_extra_evidence": messages_with_extra_evidence,
        "messages_with_true": messages_with_true,
        "other_messages": other_messages,
    }


def print_stats(stats):
    total_messages = stats["total_messages"]
    total_claims = stats["total_claims"]

    print("=== Message-level Stats ===")
    print("Total Messages:", total_messages)
    if total_messages:
        print("Average claims per message:", round(total_claims / total_messages, 2))
    else:
        print("Average claims per message:", 0)
    print('\nMessages with at least 1 "False" claim:', stats["messages_with_false"])
    print('Messages with only "Extra Evidence Needed" claims:', stats["messages_with_extra_evidence"])
    print('Messages with only "True" claims:', stats["messages_with_true"])
    print("Other messages:", stats["other_messages"])
    print("\nMessages containing each verifiability class:")
    for verifiability in [
        "publicly-verifiable",
        "recipient-verifiable",
        "message-internally-verifiable",
        "unverifiable",
    ]:
        print(f"  {verifiability}: {stats['message_verifiability_stats'].get(verifiability, 0)}")

    print("\nAverage claims per message by verifiability:")
    for verifiability in [
        "publicly-verifiable",
        "recipient-verifiable",
        "message-internally-verifiable",
        "unverifiable",
    ]:
        average = (
            stats["message_verifiability_claim_totals"].get(verifiability, 0) / total_messages
            if total_messages > 0 else 0
        )
        print(f"  {verifiability}: {average:.2f}")

    print("\n=== Claim-level Stats ===")
    print(f"Total Claims: {total_claims}")
    for verifiability, verdict_counts in stats["verifiability_stats"].items():
        category_total = stats["verifiability_claim_totals"].get(verifiability, 0)
        print(f"\nVerifiability: {verifiability.capitalize()} (Total claims: {category_total})")
        for verdict, count in verdict_counts.items():
            print(f"  {verdict.capitalize()}: {count}")

    print("\n=== Verdict Percentages Per Type ===")
    for msg_type, verdict_counts in stats["type_stats"].items():
        total = sum(verdict_counts.values())
        print(f"\nType: {msg_type} (Total claims: {total})")
        for verdict in ["false", "extra evidence needed", "true", "unsure", "skipped"]:
            count = verdict_counts.get(verdict, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {verdict}: {percentage:.2f}%")

    print("\n=== Verifiability Distribution Per Type ===")
    for msg_type, verifiability_counts in stats["verifiability_type_stats"].items():
        total = sum(verifiability_counts.values())
        print(f"\nType: {msg_type} (Total claims: {total})")
        for verifiability in [
            "publicly-verifiable",
            "recipient-verifiable",
            "message-internally-verifiable",
            "unverifiable",
        ]:
            count = verifiability_counts.get(verifiability, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {verifiability}: {percentage:.2f}%")


def main():
    args = parse_args()
    run_dir = args.run_dir or find_latest_run_dir()
    output_file = get_output_file(run_dir)
    claims_file = get_claims_file(run_dir)
    print(f"Summarizing run: {run_dir}")
    print_stats(calculate_stats(output_file, claims_file))


if __name__ == "__main__":
    main()
