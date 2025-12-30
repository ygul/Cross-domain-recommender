# chat_ui_cli.py
import chat_orchestrator

from preference_elicitor_llm import PreferenceElicitorLLM
from elicitation_logger import ElicitationLogger


def _parse_item_type_filter(q: str):
    """
    User input: 'B', 'S', 'M' (any order, multiple allowed)
    Returns: set of item types or None
    """
    q = (q or "").strip()
    if not q:
        return None

    opts = set()
    for ch in q.upper():
        if ch == "B":
            opts.add("Book")
        elif ch == "S":
            opts.add("TV series")
        elif ch == "M":
            opts.add("movie")

    return opts if opts else None


def _read_multiline(prompt: str, end_hint: str = "Finish with an empty line.") -> str:
    """
    Reads multiple lines from the user until an empty line is entered.
    This prevents 'leftover' pasted lines from being consumed by later input() calls.
    """
    print(prompt)
    print(f"({end_hint})")
    lines = []

    while True:
        line = input("> ")
        # Stop when user enters empty line
        if line == "":
            break
        lines.append(line)

    # Join into a single text block
    return "\n".join(lines).strip()


def _read_singleline(prompt: str) -> str:
    """Normal single-line input."""
    return (input(prompt) or "").strip()


def main():
    orchestrator = chat_orchestrator.ChatOrchestrator(llm_provider="openai")

    # Reuse the same LLM adapter for elicitation (saves cost, consistent config)
    elicitor = PreferenceElicitorLLM(llm=orchestrator.llm_adapter, max_questions=2)

    # Logger for evaluation
    logger = ElicitationLogger(base_dir="logs")

    while True:
        q = _read_singleline(
            "\nPlease let me know if you're looking for a book, TV series, and/or movie "
            "(type B, S, and/or M, multiple options are possible - ENTER for ALL):\n> "
        )
        where_types = _parse_item_type_filter(q)

        # ✅ IMPORTANT: read seed as MULTILINE to avoid leftover stdin lines
        seed = _read_multiline(
            "\nPlease describe what you want (or type 'exit'/'quit' on the first line to quit):",
            end_hint="Paste your text, then press ENTER on an empty line.",
        )

        seed_stripped = seed.strip()
        if seed_stripped == "" or seed_stripped.lower() in {"exit", "quit"}:
            break

        # 1) LLM asks up to 2 questions and returns a final query
        # NOTE: answers are typically short; if you also want multiline answers, see below.
        elicited = elicitor.run(seed)
        final_query = elicited.final_query

        # ✅ DEBUG: see what the elicitation did
        print(f"\n[DEBUG] Elicitation turns used: {len(elicited.turns)}")
        print(f"[DEBUG] Final query (sent to retrieval): {final_query}\n")

        # 2) Log the elicitation session for evaluation
        logger.log(
            user_seed=seed,
            turns=elicited.turns,
            final_query=final_query,
            item_types=where_types,
        )

        # 3) Run vector search + formatting
        print("\nSearching for relevant items...\n")
        print("-" * 40)
        print(orchestrator.run_once(final_query, item_types=where_types))
        print("-" * 40)


if __name__ == "__main__":
    main()