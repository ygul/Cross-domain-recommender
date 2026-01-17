# chat_ui_cli.py
from __future__ import annotations

import chat_orchestrator
from typing import Optional, Set, Literal

ItemType = Literal["Book", "TV series", "movie"]

# ----------------------------------------
# Feature flag: item-type filter in CLI
# ----------------------------------------
# False (default): user only provides a description; system searches cross-domain (ALL)
# True: show B/S/M prompt to optionally filter retrieval by item type(s)
ENABLE_TYPE_FILTER = False
PRINT_SUGGESTION_FROM_ALTERNATIVE_COLLECTION = True


def _parse_item_type_filter(q: str) -> Optional[Set[ItemType]]:
    """
    User input: 'B', 'S', 'M' (any order, multiple allowed)
    Returns: set of item types or None
    """
    if not q:
        return None

    opts: set[ItemType] = set()
    for ch in q.upper():
        if ch == "B":
            opts.add("Book")
        elif ch == "S":
            opts.add("TV series")
        elif ch == "M":
            opts.add("movie")

    return opts if opts else None


def main():
    orchestrator = chat_orchestrator.ChatOrchestrator(
        llm_provider="openai",
        enable_clarify_gate=True,
        clarify_max_questions=2,
    )

    while True:
        where_types: Optional[Set[ItemType]] = None

        # Optional item-type filtering (debug/feature flag)
        if ENABLE_TYPE_FILTER:
            print(
                "\nPlease let me know if you're looking for a book, TV series, and/or movie "
                "(type B, S, and/or M, multiple options are possible - ENTER for ALL):"
            )
            type_input = input("> ").strip()
            where_types = _parse_item_type_filter(type_input)

        print("\nPlease describe what you want (or 'exit' / ENTER to quit):")
        seed = (input("> ") or "").strip()
        if seed == "" or seed.lower() in {"exit", "quit"}:
            break

        print("\nSearching for relevant items...\n")
        print("-" * 40)

        # Single entrypoint: all chat logic is in the orchestrator
        output, elicited = orchestrator.chat(
            seed,
            item_types=where_types,
            input_fn=input,
            print_fn=print,
            use_alternative_collection=False,
        )
        print(output)

        if PRINT_SUGGESTION_FROM_ALTERNATIVE_COLLECTION:
            alt_query = seed
            if elicited is not None and getattr(elicited, "final_query", None):
                alt_query = elicited.final_query

            print("\n" + "=" * 70)
            print("ALTERNATIVE COLLECTION SUGGESTION")
            print("=" * 70)
            alt_output = orchestrator.run_once(
                alt_query,
                item_types=where_types,
                use_alternative_collection=True,
            )
            print(alt_output)
            print("=" * 70 + "\n")

        # Optional debug visibility
        if elicited is not None:
            print("\n[DEBUG]")
            print(f"Elicitation turns used: {len(elicited.turns)}")
            print(f"Final query (sent to retrieval): {elicited.final_query}")
            if ENABLE_TYPE_FILTER:
                print(f"Item type filter: {sorted(where_types) if where_types else ['ALL']}")

        print("-" * 40)


if __name__ == "__main__":
    main()