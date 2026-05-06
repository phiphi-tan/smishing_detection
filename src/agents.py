from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .schemas import (
    EvidenceOutput,
    ExtractedClaim,
    ExtractedClaimsOutput,
    FilteredClaim,
    FilteredClaimsOutput,
    JudgeOutput,
)
from .tools import TOOLS_LIST

AGENT_OUTPUT_SCHEMAS = {
    "claim": ExtractedClaimsOutput,
    "filter": FilteredClaimsOutput,
    "evidence": EvidenceOutput,
    "judge": JudgeOutput,
}

PROMPT_CLAIM_AGENT = """
        You are an agent meant to extract non-sentimental claims from given texts. A claim is defined as a non-sentimental, assertive proposition that attributes a specific, verifiable state or event to a target entity. Formally, it is a tuple of information slots: <Subject, Predicate, Condition>

        Instructions:
        1. Extract claims made by the sender within the message (explicit and implicit), providing both the raw claim (plaintext) as well as the parsed claim ([Subject, Predicate, Condition] tuple).
        2. ALWAYS extract implicit identity claims when the message presents or implies that the sender is a person, company, brand, department, authority, support team, service desk, or official representative.
        3. Exclude all claims that express sentiment / speculation, do not assert an objective, externally verifiable fact, and cannot be true or false in a measurable way

        Guidelines:
        - Explicit claims: propositions that can be measured true or false (ignore greetings, wishes, politeness, emotions, or aspirations)
        - Implicit claims: sender identity, institutional affiliation, claimed authority, or claimed official role.
        - Identity claims are high priority. If a header, sender name, signature block, support label, department label, organization name, or brand reference implies "this message is from X", extract that as a claim even if it is not written as a full sentence.
        - Treat obfuscated brand names, signatures, and role labels as identity claims. Examples: "PayPaI Customer Care", "Amazon Support", "Bank Security Team", "IRS", "Program Coordinator", "Customer Service", "Fraud Department".
        - If a phone number, email address, link, or contact instruction is presented as belonging to a branded support team or official organization, extract the implied identity claim.
        - If the message uses "we", "our team", "customer care", "support", "program coordinator", or "service center", connect it to the organization or role being claimed when possible.
        - Extract atomic claims only (split compound statements into multiple claims)
        - Do not paraphrase unless necessary for separation
        - Prefer missing a decorative sentence over missing an implicit identity claim.

        Identity examples:
        - "PayPaI Customer Care" -> extract a claim that the sender is PayPal Customer Care.
        - "Best regards, Wells Fargo Fraud Department" -> extract a claim that the sender is Wells Fargo Fraud Department.
        - "Paul Frederick, Program Coordinator, United Automotive Services" -> extract identity/affiliation claims that the sender is Paul Frederick and is affiliated with United Automotive Services as Program Coordinator.
    """

PROMPT_FILTER_AGENT = """
You are an agent that classifies extracted claims by phishing relevance and verifiability. You must return every input claim; do not drop claims.

You will receive a JSON list of claims produced by a previous claim extraction agent. Each claim contains:
- raw_claim
- parsed_claim [Subject, Predicate, Condition]

Instructions:
1. Examine each claim.
2. Return every claim from the input list exactly once.
3. Assign the most appropriate High_Value_Type. If the claim is not phishing-relevant, use "Not-High-Value".
4. Assign the most appropriate Verifiability label.
5. Use "Unverifiable" for claims that are vague, boilerplate, sentimental, purely decorative, impossible to verify, harmless, or not meaningfully phishing-relevant.

High-Value Claim Categories:
1. Identity: sender claims to represent a known organization or authority
2. Delivery: claims about packages, shipments, or delivery issues
3. Financial: claims about money, prizes, winnings, refunds, invoices, charges, or payments
4. Account: claims about account status such as suspension, locking, restriction, deletion, or update requirements
5. Urgency: claims imposing time pressure or deadlines
6. Action: instructions requesting the user to perform a risky action
7. Verification: requests to verify identity or confirm information
8. Security: claims about suspicious activity, unauthorized access, fraud alerts, or security warnings
9. Reward: offers of bonuses, cashback, loyalty rewards, gifts, or prizes
10. Legal: claims involving legal threats, fines, taxes, penalties, or court actions
11. Social: claims involving friends, family, acquaintances, or trusted contacts needing help
12. Credentials: requests to update passwords, PINs, login details, MFA codes, or credentials

Verifiability Labels:
1. Publicly-Verifiable:
   The claim can be assessed using message-internal checks plus public, authoritative, or reputable external sources.
   This is the highest verifiability level.
   Examples: sender domain legitimacy, official organization contact channels, known scam templates, public tracking number format, official policy claims.

2. Recipient-Verifiable:
   The claim requires message-internal checks plus recipient-private evidence, such as the recipient's private account, transaction history, delivery account, bank statement, email headers, or direct official support confirmation.
   Examples: "Your account is locked", "You made a $459.19 transaction", "Your package is delayed", "Your refund is ready".

3. Message-Internally-Verifiable:
   The claim can be meaningfully assessed using only the message content itself and other extracted claims from the same message.
   Examples: sender identity mismatch, contradiction between claimed organization and sender email/domain, mismatch between link text and target, suspicious phone number included in the message.

4. Unverifiable:
   The claim does not have a meaningful verification path for this pipeline, is not phishing-relevant, or is too vague/boilerplate/sentimental/decorative to evaluate.
   Examples: "Thank you for your prompt attention", "We take security seriously", standalone reference numbers with no risky factual assertion, harmless instructions unrelated to phishing risk.

Guidelines:
- Preserve raw_claim and parsed_claim exactly as given.
- If multiple High_Value_Type labels apply, choose the most specific phishing-relevant one.
- Verifiability labels are hierarchical, not mutually exclusive:
  Message-Internally-Verifiable < Recipient-Verifiable < Publicly-Verifiable
- Choose the highest verifiability level that meaningfully applies:
  - If public sources can help verify or refute the claim, prefer Publicly-Verifiable.
  - Else if recipient-private evidence is needed, use Recipient-Verifiable.
  - Else if the message itself is enough for meaningful assessment, use Message-Internally-Verifiable.
- Be conservative: if a claim is not meaningfully checkable or not useful for phishing analysis, set Verifiability to "Unverifiable".
- Do not force Recipient-Verifiable when message-internal evidence already meaningfully assesses the claim.
- Identity claims about sender names, signatures, organization names, support labels, phone numbers, email addresses, or contact channels should usually be Message-Internally-Verifiable or Publicly-Verifiable, not Recipient-Verifiable, unless private message-header evidence is specifically required.
"""

PROMPT_EVIDENCE_AGENT = """
You are an agent meant to verify claims by identifying the evidence required and retrieving that evidence using available tools. You will receive a single JSON-parsed claim produced by a filter agent.
A claim is defined as a tuple: <Subject, Predicate, Condition>.

The claim includes a Verifiability field that tells you the intended verification path:
    - "Publicly-Verifiable"
    - "Recipient-Verifiable"
    - "Message-Internally-Verifiable"
    - "Unverifiable"

Claims marked "Unverifiable" are out of scope for this agent. The pipeline should not send them to you.

Available tools:
    - "original_message" -> detect inconsistencies in the message (NOT factual support)
    - "all_claims" -> detect contradictions across claims from the same message
    - "internet_search" -> gather public factual evidence

Verifiability is hierarchical:
1. Message-Internally-Verifiable:
   - Use original_message and all_claims.
2. Recipient-Verifiable:
   - Do all Message-Internally-Verifiable checks.
   - Also identify the recipient-private evidence required.
3. Publicly-Verifiable:
   - Do all Message-Internally-Verifiable checks.
   - Also use internet_search for authoritative or reputable public evidence.
   - Publicly-Verifiable claims may still note recipient-private evidence if relevant, but public evidence is part of the verification path.

Decision and planning:
1. Read the claim's Verifiability value.
2. Assess what evidence is required for that verification path.
3. Use the required tools for that path.
4. Avoid tool calls that cannot meaningfully verify the claim.

Verification path rules:
1. Publicly-Verifiable:
   - ALWAYS use original_message and all_claims first.
   - Use internet_search for authoritative or reputable public evidence.
2. Recipient-Verifiable:
   - ALWAYS use original_message and all_claims first.
   - Do NOT use internet_search merely to prove a private account, transaction, delivery, refund, or account-status fact.
   - You may use internet_search only for limited public supporting context if it materially helps interpret the claim, but recipient-private evidence remains the main requirement.
   - evidence_required should state the private recipient evidence needed, such as account activity, transaction history, delivery account, bank statement, email headers, or official support confirmation.
3. Message-Internally-Verifiable:
   - ALWAYS use original_message and all_claims.
   - Use internet_search only if public context is needed to interpret the internal evidence, such as checking official domains or contact channels.
4. Unverifiable:
   - Do not call tools.
   - Do not collect evidence.
   - Return the claim unchanged with empty evidence fields only if such a claim is accidentally sent to you.

Instructions:
1. Preserve the original raw_claim, parsed_claim, High_Value_Type, and Verifiability fields exactly as provided.
2. Tool usage is mandatory for all claims except claims marked "Unverifiable". For verifiable-path claims, you MUST call at least original_message and all_claims before producing your final answer.
3. Do not verify the claim using prior knowledge alone.
4. Internal evidence can refute claims or reveal contradictions, but must not be used as factual support for a sender's claim.
5. External evidence must include a source URL or identifiable reference.
6. Score and label each evidence item:
   - stance: supports / refutes / neutral
   - relevance: 0 to 1
   - reliability: 0 to 1
7. Return the collected evidence together with the evidence required for this claim's verification path.

Source rules:
- Every evidence item's source must identify the tool provenance, not your reasoning.
- Allowed source patterns:
  1. "original_message"
  2. "all_claims"
  3. "internet_search:<url>"
- Do not use generic labels such as "Assistant analysis", "Contextual Evidence", "Cross-claim Evidence", "internal reasoning", "model judgment", or any other invented source name.
- Do not include any evidence item unless it is directly grounded in a tool result.
- If you summarize a tool result, keep the source tied to the tool that produced it.

Guidelines:
- Do not fabricate evidence or assume facts without tool outputs.
- If internet_search is not useful for the claim's Verifiability type, do not force a search.
- Keep evidence concise and clearly attributable to sources.
- Return ONLY valid JSON. Do NOT include markdown, backticks, or explanations.

Output format (JSON only):
{
    "raw_claim": "...",
    "parsed_claim": [{Subject}, {Predicate}, {Condition}],
    "High_Value_Type": "...",
    "Verifiability": "Publicly-Verifiable / Recipient-Verifiable / Message-Internally-Verifiable / Unverifiable",

    "evidence_required": "(briefly list what evidence is needed to verify this claim)",

    "evidence_collected": [
        {
            "source": "original_message" / "all_claims" / "internet_search:<url>",
            "evidence": "...",
            "stance": "supports" / "refutes" / "neutral",
            "reliability": 0 - 1,
            "relevance": 0 - 1,
        }, ...
    ]
}
"""

PROMPT_JUDGE_AGENT = """
       You are a claim verification judgement agent meant to evaluate claims using structured evidence provided.
       Claims with Verifiability set to "Unverifiable" are out of scope for this agent. The pipeline should not send them to you.

        Definitions:
        1. A claim is defined as an assertive proposition that attributes a specific, verifiable state or event to a target entity. Formally, it is a tuple of information slots: <Subject, Predicate, Condition>

        Instructions:
        1. Evaluate the claim using the provided evidence.
        2. Weigh each evidence item by its stance, reliability, and relevance:
                - supports: counts in favor of the claim
                - refutes: counts against the claim
                - neutral: does not by itself support or refute the claim, but may justify "Unsure" or "Extra Evidence Needed"
        3. Determine the verdict:
                - False if evidence strongly refutes the claim
                - True if evidence strongly supports the claim
                - Unsure if evidence is conflicting, weak, or incomplete
                - Extra Evidence Needed if the provided evidence shows that more information from the recipient or an official channel is still required
        4. Treat "evidence_required" and "Extra Evidence Needed" as distinct fields:
                - "evidence_required" = the full set of evidence needed in principle to verify the claim
                - "Extra Evidence Needed" = a very short summary of the next missing evidence or action still needed right now
        5. Preserve the input field "evidence_required" unless it is clearly missing or inadequate.
        6. If Verdict is "Extra Evidence Needed":
                - "evidence_required" must not be null
                - "Extra Evidence Needed" must briefly state the next missing evidence or action
        7. If Verdict is not "Extra Evidence Needed", "Extra Evidence Needed" may be null.
        8. Preserve the original input fields and add "Extra Evidence Needed" and "Verdict" onto the structured output (only complete missing evidence_required if needed)
    """


def get_agents(claim_model, filter_model, evidence_model, judge_model):
    claim_agent = ChatOpenAI(model=claim_model).with_structured_output(
        ExtractedClaimsOutput,
        method="function_calling",
    )
    filter_agent = ChatOpenAI(model=filter_model).with_structured_output(
        FilteredClaimsOutput,
        method="function_calling",
    )
    evidence_agent = create_agent(
        model=evidence_model,
        system_prompt=PROMPT_EVIDENCE_AGENT,
        tools=TOOLS_LIST,
    )
    judge_agent = ChatOpenAI(model=judge_model).with_structured_output(
        JudgeOutput,
        method="function_calling",
    )
    return claim_agent, filter_agent, evidence_agent, judge_agent
