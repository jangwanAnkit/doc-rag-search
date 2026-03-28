#!/usr/bin/env python3
"""
Snippet Extractor - Extract and highlight relevant snippets from chunks.
"""

import re
from typing import List, Optional


def extract_snippet(
    text: str, query: str, window: int = 200, max_snippets: int = 2
) -> List[dict]:
    """
    Extract relevant snippets from text with keyword highlighting.

    Args:
        text: Full chunk text
        query: User query
        window: Character window around match
        max_snippets: Maximum number of snippets to return

    Returns:
        List of snippet dicts with highlighted text and positions
    """
    keywords = [w.lower() for w in query.split() if len(w) > 3]

    if not keywords:
        return [
            {
                "text": text[:window] + "...",
                "start": 0,
                "end": min(window, len(text)),
                "highlighted": text[:window] + "...",
            }
        ]

    snippets = []
    text_lower = text.lower()

    for keyword in keywords[:3]:
        start = 0
        while start < len(text_lower):
            pos = text_lower.find(keyword, start)
            if pos == -1:
                break

            snippet_start = max(0, pos - window // 2)
            snippet_end = min(len(text), pos + len(keyword) + window // 2)

            snippet_text = text[snippet_start:snippet_end]

            highlighted = re.sub(
                rf"\b({re.escape(keyword)})\b",
                r"<mark>\1</mark>",
                snippet_text,
                flags=re.IGNORECASE,
            )

            snippets.append(
                {
                    "text": snippet_text,
                    "start": snippet_start,
                    "end": snippet_end,
                    "highlighted": highlighted,
                    "keyword": keyword,
                }
            )

            start = pos + len(keyword)

            if len(snippets) >= max_snippets:
                break

        if len(snippets) >= max_snippets:
            break

    if not snippets:
        return [
            {
                "text": text[:window] + "...",
                "start": 0,
                "end": min(window, len(text)),
                "highlighted": text[:window] + "...",
            }
        ]

    return snippets[:max_snippets]


if __name__ == "__main__":
    text = """
    This is an action for breach of contract arising from the failure to deliver
    goods under a commercial supply agreement. The plaintiff, Mr. Smith, entered
    into a written contract with the defendant, Jones Pty Ltd, for the supply
    of office equipment worth $150,000.
    
    The defendant failed to deliver the goods by the agreed date of 15 March 2023.
    The plaintiff contends that this delay caused significant financial loss.
    The Court finds in favour of the plaintiff.
    """

    query = "breach of contract damages"

    snippets = extract_snippet(text, query)

    print(f"Query: {query}")
    print(f"\nExtracted {len(snippets)} snippet(s):")
    for i, s in enumerate(snippets, 1):
        print(f"\n--- Snippet {i} (keyword: {s['keyword']}) ---")
        print(s["highlighted"])
