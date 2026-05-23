from typing import Optional
from httpx import request
from langchain_core.tools import tool

@tool
def search_arxiv(query: str, max_results: int = 3) -> Optional[dict]:
    """Search arXiv for latest papers on a topic."""
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    try: 
        response = request(
            "GET",
            url
            )
        response.raise_for_status()
    except Exception as e:
        return {"error": str(e)}
    
    # Parse basic info from arXiv ATOM feed
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.text)
    print(root)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        results.append({
            "title": entry.find("atom:title", ns).text,
            "summary": entry.find("atom:summary", ns).text.strip(),
            "published": entry.find("atom:published", ns).text,
            "link": entry.find("atom:id", ns).text,
        })
    return {"papers": results}
