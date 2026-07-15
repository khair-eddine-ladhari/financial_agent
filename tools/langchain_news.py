import os
from langchain_core.tools import tool
from tavily import TavilyClient
from tenacity import retry, wait_random_exponential, stop_after_attempt
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
def search_recent_news(query: str) -> str:
    """Search the web for recent news about a company or topic.
    Example input: 'Tesla Q3 2026 earnings'
    """
    try:
        results = tavily_client.search(query=query, max_results=3)
        summaries = [r["content"][:300] for r in results.get("results", [])]

        if not summaries:
            return f"No recent news found for: {query}"

        return "\n\n".join(summaries)
    except Exception as e:
        return f"Error searching news: {str(e)}"