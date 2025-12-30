# results_formatter.py
from __future__ import annotations

from vector_store import SearchResult
from llm_adapter import LLMAdapter


def _label_distance(dist: float) -> str:
    """
    Chroma returns cosine distance (lower = closer / better match).
    These bands are heuristic and mainly for user-friendly interpretation.
    """
    if dist <= 0.20:
        return "Strong match"
    if dist <= 0.40:
        return "Good match"
    if dist <= 0.60:
        return "Moderate match"
    return "Weak match"


def format_results(original_query: str, results: list[SearchResult], llm_adapter: LLMAdapter) -> str:
    """
    Format search results into a user-friendly string using the LLM adapter,
    including cosine distance (lower = better semantic match).
    """
    if not results:
        return "No relevant items found."

    # Build a combined text input for the LLM with distances included
    formatted_items: list[str] = [
        "Original Query: " + (original_query or "").strip(),
        "Search Results (ranked by semantic similarity; lower cosine distance = stronger match):",
    ]

    for idx, result in enumerate(results, start=1):
        title = "Unknown Title"
        item_type = "unknown"
        if result.metadata:
            title = result.metadata.get("name", title)
            item_type = result.metadata.get("item_type", item_type)

        description = (result.document or "No description available.").strip()

        dist = float(result.score)  # this is the Chroma distance (lower = closer)
        dist_str = f"{dist:.3f}"
        dist_label = _label_distance(dist)

        formatted_items.append(
            f"{idx}. {title} ({item_type})\n"
            f"Cosine distance: {dist_str} ({dist_label})\n"
            f"{description}\n"
        )

    combined_text = "\n".join(formatted_items)

    system_prompt = (
        "You are responsible for explaining semantic search results to users in a concise and readable manner.\n"
        "Important:\n"
        "- Each result includes a COSINE DISTANCE score, where LOWER means a STRONGER semantic match.\n"
        "- You MUST include the cosine distance for each item and interpret it briefly (Strong/Good/Moderate/Weak).\n"
        "Formatting rules:\n"
        "- Do NOT use markdown.\n"
        "- Use a numbered list.\n"
        "- For each item, include:\n"
        "  1) Title and item type\n"
        "  2) 'Similarity:' line with the label and the cosine distance value\n"
        "  3) A very brief plot description (1–2 sentences)\n"
        "  4) 'Relation with your query:' explanation (1 sentence)\n"
        "- Keep total output under 300 words.\n"
        "- Always start with one short introductory sentence like:\n"
        "  'Here are some items related to your query, ranked by semantic similarity:'\n"
    )

    user_prompt = f"Please format the following search results:\n\n{combined_text}"

    try:
        formatted_output = llm_adapter.generate(system_prompt, user_prompt)
        return formatted_output
    except Exception as e:
        print(f"LLM adapter failed to format results: {e}")
        # Fallback: return the raw combined text (still includes distances)
        return combined_text