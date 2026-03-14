from dotenv import load_dotenv
load_dotenv()

import csv
from langchain.agents import create_agent
from langchain.tools import tool
# from langchain_google_community.search import GoogleSearchResults, GoogleSearchAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient 

VLLM = ChatOpenAI(
    # model_name="Qwen/Qwen3-8B",
    model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    # model_name="Qwen/Qwen3-14B-FP8",
    # model_name="Qwen/Qwen3.5-9B",
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

agent1 = create_agent(
    model= VLLM,
    system_prompt = """
        You are an agent meant to extract non-sentimental claims from given texts and classify their verifiabilities.

        Extract claims from the message. 
        - Explicit claims: propositions that can be measured true or false (ignore greetings, wishes, politeness, emotions, or aspirations)
        - Implicit claims: sender identity / institutional affiliation

        Definitions:
        1. A claim is defined as an non-sentimental, assertive proposition that attributes a specific, verifiable state or event to a target entity. Formally, it is a tuple of information slots: <Subject, Predicate, Condition>
        2. Verifiable: A claim is verifiable if a neutral third party could refute it using contextual understanding or using publicly available sources (such as internet searches) without any other private information / actions

        Instructions:
        1. Extract claims made by the sender within the message (explicit and implicit), providing both the raw claim (plaintext) as well as the parsed claim ([Subject, Predicate, Condition] tuple).
        2. Exclude all claims that express sentiment / speculation, do not assert an objective, externally verifiable fact, and cannot be true or false in a measurable way
        3. Determine the verifiability category for each remaining claim.

        Guidelines:
        - Return JSON only
        - Extract atomic claims only (split compound statements into multiple claims)
        - Do not paraphrase unless necessary for separation 

        Output format (JSON only):
            [{"raw_claim": "...",
                "parsed_claim": [{Subject}, {Predicate}, {Condition}],  
                "Category": "Verifiable / Unverifiable"}, ...
            ]
    """
)

# google_search_tool = GoogleSearchResults(
#     api_wrapper=GoogleSearchAPIWrapper(google_api_key=os.environ.get("GOOGLE_API_KEY"), google_cse_id=os.environ.get("GOOGLE_CSE_ID")),
#     num=5
# )

# @tool('google_search', description='Searches the internet and returns the top 5 JSON results', return_direct=False)
# def google_search(query:str) -> str:
#     print(f"[TOOL CALLED] Search query: {query}")
#     json_string = google_search_tool.run(query)
#     print(f"[TOOL RETURNED] {json_string}")
#     return json_string

search = DuckDuckGoSearchResults(output_format="json", max_results=4)
@tool('search_internet', description="Searches the internet. Input should be a natural search query.", return_direct=False)
def search_internet(query: str) -> str:
    query = query.replace('"', "") # removes quotes
    print(f"[TOOL CALLED] Search query: {query}")
    search_results = search.invoke(query)
    print(f"[TOOL RETURNED] {search_results}")
    return search_results

# mimics a database
context_data = {}
with open("./data/D2.csv", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        context_id = row["scam_id"]
        context_data[context_id] = row

@tool('find_context', description="Given a scam_id, returns the original message text (raw text) as context. This context is unverified and must be used solely to identify inconsistencies or refute claims, never to support them.", return_direct=False)
def find_context(scam_id: str) -> str:
    print(f"[TOOL CALLED] find_context: {scam_id}")
    row = context_data.get(scam_id)
    if row:
        raw_text = row.get("raw_text", "")
    else:
        raw_text = "No context found for this scam_id."
    print(f"[TOOL RETURNED] {raw_text}")
    return raw_text

tools_list = [search_internet, find_context]

agent2 = create_agent(
    model= VLLM,
    system_prompt = """
       You are a scam-detection fact-checking agent meant to refute independent claims. Your goal is to investigate claims and search for refuting evidence using all tools available to you.
       Assume claims are false and attempt to disprove it using internal / external evidence; ONLY if EXTERNAL evidence supports a claim should it be validated. Contextual text (internal evidence) is unverified and cannot support any claims, ONLY refuting them.
       
        Definitions:
        1. A claim is defined as an non-sentimental, assertive proposition that attributes a specific, verifiable state or event to a target entity. Formally, it is a tuple of information slots: <Subject, Predicate, Condition>
        2. Verifiable: A claim is verifiable if a neutral third party could, in principle, check it using publicly available sources (such as the internet) or from contextual understand (from the original message)
        4. Internal evidence: Logical inconsistencies between a claim and the context (e.g. claiming "High Salary" with a context of "Minimal Experience Required")
        5. External evidence: Evidence obtained through external, public sources (such as internet searches)

        Instructions:
        1. Highlight logical inconsistencies between a claim and the context. Context MUST never be used to support a claim.
        2. (For verifiable claims only) If internal evidence does not refute the claim, you should also fetch relevant external evidence related to the claim using the tools given to you.
        3. Provide a brief explanation of each evidence collected, and assign a reliability and relevance score from 0 to 1.
        4. Stop collecting evidence if there is sufficient / strong independent evidence that supports / refutes the claim.
        5. Return JSON only, appending only "Evidence Collected" onto the original claim JSON (do not change the original claim or add a conclusion).

        Output format (JSON only):
            {"raw_claim": "...",
                "parsed_claim": [{Subject}, {Predicate}, {Condition}],
                "Category": "Verifiable / Unverifiable",
                "Evidence Collected":[{
                    "Source": "Internal Evidence" / {url link of external evidence},
                    "Evidence": {short explanation}, _
                    "Reliability Score": 0-1,
                    "Relevance Score": 0-1,
                }...]}
    """,
    tools=tools_list,
)

agent3 = create_agent(
    model= VLLM,
    system_prompt = """
       You are a claim verification judgement agent meant to evaluate claims using structured evidence provided. You will perform different actions depending on whether the claim is of category "Verifiable" or "Unverifiable".

        Definitions:
        1. A claim is defined as an assertive proposition that attributes a specific, verifiable state or event to a target entity. Formally, it is a tuple of information slots: <Subject, Predicate, Condition>
        2. Verifiable: The claim can be verified through publicly available internal and external information.
        
        Instructions:
        1a. For unverifiable claims, state extremely briefly what additional information would be needed to verify the claim, e.g., "recipient must log in to their account and check transactions".
            - If given internal evidence is sufficient to refute a claim, the verdict will be "False"
            - If extra evidence is required, the verdict will be "Extra Evidence Needed"
        1b. For verifiable claims, determine the verdict:
                - False if evidence strongly refutes the claim
                - True if external evidence strongly refutes the claim
                - Uncertain if evidence is conflicting or contains low reliability and/or relevance scores
        2. Return JSON only, appending "Extra Evidence Needed" and "Verdict" onto the original JSON (do not change the original input)

        Output format (JSON only):
            {"raw_claim": "...",
                "parsed_claim": [{Subject}, {Predicate}, {Condition}],
                "Category": "Verifiable / Unverifiable"},
                "Evidence Collected":[{
                    "Source": "Internal Evidence" / {link of external evidence},
                    "Evidence": {explanation}, _
                    "Reliability Score": 0-1,
                    "Relevance Score": 0-1,
                }...],
                "Extra Evidence Needed": {extremely short explanation} ("null" if unnecessary),
                "Verdict": "False" / "True" / "Unsure" / "Extra Evidence Needed"}
    """,
)

