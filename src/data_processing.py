import json
from collections import defaultdict

filename = "../data/smishing_output.ndjson"

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