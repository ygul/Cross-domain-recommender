from vector_store import SearchResult
from llm_adapter import LLMAdapter

def format_results(original_query: str,results: list[SearchResult], llm_adapter: LLMAdapter) -> str:
    """Format search results into a user-friendly string using the LLM adapter."""
    if not results:
        return "No relevant items found."

    # Create a list of the query and results to give to the LLM for proper formatting
    formatted_items = ['Original Query: ' + original_query + "\n", "Search Results:\n"]
    for idx, result in enumerate(results, start=1):
        
        # Metadata available: source_id,item_type,name,vote_count,vote_average,source_overview,Year,source_genres,Simplified genre,created_by / director / author
        title = result.metadata.get("name", "Unknown Title") if result.metadata else "Unknown Title"
        description = result.document or "No description available."
        formatted_items.append(f"{idx}. {title}\n{description}\n")

    combined_text = "\n".join(formatted_items)

    system_prompt = (
        "You are responsible for explaining search results to users in a concise and readable manner."
        "Format the results clearly, using bullet points or numbered lists as appropriate."
        "Explain for each item a very brief plot description, and an explanation how it is related to the user's original query."
        "Don't use more than 300 words in total."
        "Don't use markdown formatting."
        "Always start with a short introductory sentence, e.g., 'Here are some items related to your query:'."
    )
    user_prompt = f"Please format the following search results:\n\n{combined_text}"

    try:
        formatted_output = llm_adapter.generate(system_prompt, user_prompt)
        return formatted_output
    except Exception as e:
        print(f"LLM adapter failed to format results: {e}")
        return combined_text