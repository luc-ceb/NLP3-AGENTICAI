from typing import Optional
from httpx import request
from langchain_core.tools import tool
from pydantic import BaseModel
from .models import Summary

class wikipediaSummarySearch(BaseModel):
    query: str

@tool(args_schema=wikipediaSummarySearch)
def search_wikipedia_summary(query: str) -> Optional[Summary]:
    """Search Wikipedia and return the summary of the top result."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}?redirect=false"
    try:
        response = request("GET",url)
        response.raise_for_status()
    except Exception as e:
        return {"error": str(e)}
    return Summary(**response.json())

@tool()
def search_wikipedia_full(query: str) -> str:
    """Return the full content of a Wikipedia page as plain text."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{query}?redirect=false"
    try:
        response = request("GET", url)
        response.raise_for_status()
        data = response.json()
        # Combine all sections into a single text string
        content = ""
        for section in data.get("sections", []):
            content += section.get("text", "") + "\n"
        return content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
@tool()
def search_wikipedia_related(query: str, limit: int = 5) -> list[str]:
    """Return a list of related Wikipedia page titles."""
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch={query}"
    try:
        response = request("GET", url)
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])
        return [r["title"] for r in results[:limit]]
    except Exception as e:
        return [f"Error: {str(e)}"]

if __name__ == "__main__":
    response = search_wikipedia_summary.invoke("Python (programming language)")
    print(response.model_dump())

