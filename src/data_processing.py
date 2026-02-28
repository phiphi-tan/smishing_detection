import json
from collections import defaultdict
import pandas as pd

filename = "../data/smishing_output.ndjson"
csv_file = "../data/D2.csv"

claim_stats = defaultdict(lambda: defaultdict(int))
total_claims = 0  

total_messages = 0
messages_with_false = 0
messages_with_extra_evidence = 0
messages_with_true = 0
other_messages = 0


for line in open(filename, "r", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    total_messages += 1
    claims = json.loads(line)
    
    verdicts = [claim.get("Verdict", "").lower() for claim in claims]
    
    # Line-level categories
    if any(v == "false" for v in verdicts):
        messages_with_false += 1
    elif all(v == "extra evidence needed" for v in verdicts):
        messages_with_extra_evidence += 1
    elif all(v == "true" for v in verdicts):
        messages_with_true += 1
    else:
        other_messages += 1
    
    # Claim-level counts
    for claim in claims:
        total_claims += 1
        category = claim.get("Category", "Unknown").lower()
        verdict = claim.get("Verdict", "Unknown").lower()
        claim_stats[category][verdict] += 1

# Print line-level stats
print("=== Message-level Stats ===")
print("Total Messages:", total_messages)
print("Average claims per message:", total_claims // total_messages)
print('\nMessages with at least 1 "False" claim:', messages_with_false)
print('Messages with only "Extra Evidence Needed" claims:', messages_with_extra_evidence)
print('Messages with only "True" claims:', messages_with_true)
print('Other messages:', other_messages)

# Print claim-level stats
print("\n=== Claim-level Stats ===")
print(f"Total Claims: {total_claims}")
for category, verdict_counts in claim_stats.items():
    category_total = sum(verdict_counts.values())
    print(f"\nCategory: {category.capitalize()} (Total claims: {category_total})")
    for verdict, count in verdict_counts.items():
        print(f"  {verdict.capitalize()}: {count}")

print("\n=== Verdict Percentages Per Type ===")

df = pd.read_csv(csv_file)

type_stats = defaultdict(lambda: defaultdict(int))
category_type_stats = defaultdict(lambda: defaultdict(int))

with open(filename, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        
        msg_type = df.loc[idx, "message_type"]
        claims = json.loads(line)
        
        for claim in claims:
            verdict = claim.get("Verdict", "Unknown").strip().lower()
            type_stats[msg_type][verdict] += 1
            category = claim.get("Category", "Unknown").strip().lower()
            category_type_stats[msg_type][category] += 1

# Print percentages
for msg_type, verdict_counts in type_stats.items():
    total = sum(verdict_counts.values())
    
    print(f"\nType: {msg_type} (Total claims: {total})")
    for verdict in ["false", "extra evidence needed", "true", "unsure"]:
        count = verdict_counts.get(verdict, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {verdict}: {percentage:.2f}%")


print("\n=== Category Distribution Per Type ===")

# Print percentages per type
for msg_type, category_counts in category_type_stats.items():
    total = sum(category_counts.values())
    
    print(f"\nType: {msg_type} (Total claims: {total})")
    for category in ["verifiable", "unverifiable"]:
        count = category_counts.get(category, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {category}: {percentage:.2f}%")