import json
import csv
import os
from typing import Any, List, Union

from dotenv import load_dotenv
load_dotenv()

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools import WebSearch

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchResults

# mimics a database
context_data = {}
with open("./data/D2.csv", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        context_id = row["scam_id"]
        context_data[context_id] = row

@register_tool('original_message')
class OriginalMessage(BaseTool):
    description = "Given a scam_id, returns the original message text (raw text) as context. This context is unverified and must be used solely to identify inconsistencies."

    parameters = {
        'type': 'object',
        'properties': {
            'scam_id': {
                'type': 'string',
                'description': 'The unique ID of the scam message to look up.',
            }
        },
        'required': ['scam_id'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
            params = self._verify_json_format_args(params)        
            scam_id = str(params['scam_id']).strip()
            
            # print(f"[TOOL CALLED] original_message: {scam_id}")
            
            raw_text = context_data.get(scam_id)
            
            if raw_text:
                result = f"Original message for ID {scam_id}:\n{raw_text}"
            else:
                result = f"No context found for scam_id: {scam_id}."
                
            # print(f"[TOOL RETURNED] {result[:100]}...") 
            return result

@register_tool('all_claims')
class AllClaims(BaseTool):
    description = "Returns the list of all extracted claims from the original message. Useful for performing cross-claim checks to find inconsistencies."

    parameters = {
        'type': 'object',
        'properties': {},
        'required': [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
            file_path = './data/claim_agent_test.ndjson' # hardcoded            
            # print(f"[TOOL CALLED] all_claims")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    last_line = None
                    for line in f:
                        if line.strip():
                            last_line = line.strip()

                if not last_line:
                    result = "NDJSON file is empty or contains no valid lines."
                else:
                    result = f"Latest entry:\n{last_line}"

            except FileNotFoundError:
                result = f"File not found: {file_path}"
            except Exception as e:
                result = f"Error reading file: {str(e)}"

            # print(f"[TOOL RETURNED] {result[:100]}...")
            return result

# using duckduckgo search
search = DuckDuckGoSearchResults(output_format="json", max_results=4)
@register_tool('internet_search')
class InternetSearch(BaseTool):
    description = "Searches the internet. Input should be a natural search query."

    parameters = {
        'type': 'object',
        'properties': {
            'search_query': {
                'type': 'string',
                'description': 'The query to search the internet',
            }
        },
        'required': ['search_query'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)        
        search_query = str(params['search_query']).strip()
        # print(f"\n[TOOL CALLED] Search query: {search_query}")
        search_results = search.invoke(search_query)
        # print(f"[TOOL RETURNED] {search_results}")
        return search_results


#web_search by SerperAPI
TOOLS_LIST = ['internet_search', 'original_message', 'all_claims'] 


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
