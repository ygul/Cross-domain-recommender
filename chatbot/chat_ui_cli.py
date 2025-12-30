import chat_orchestrator

from preference_elicitor_llm import PreferenceElicitorLLM
from elicitation_logger import ElicitationLogger


def _read_multiline(prompt: str, end_hint: str = "Paste your text, then finish with an empty line.") -> str:
    """
    Reads multiple lines until an empty line is entered.
    Prevents leftover pasted lines from being consumed by later input() calls.
    """
    print(prompt)
    print(f"({end_hint})")
    lines = []
    while True:
        line = input("> ")
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    orchestrator = chat_orchestrator.ChatOrchestrator(llm_provider="openai")

    # LLM generates up to 2 clarification questions + a final query (may stop after 1)
    elicitor = PreferenceElicitorLLM(llm=orchestrator.llm_adapter, max_questions=2)

    # Logger for evaluation
    logger = ElicitationLogger(base_dir="logs")

    while True:
        # Search across ALL item types by default
        where_types = None

        seed = _read_multiline(
            "\nDescribe what you are looking for (or type 'exit'/'quit' to stop):",
            end_hint="You may paste multiple lines. Press ENTER on an empty line to continue.",
        )

        seed_stripped = seed.strip()
        if seed_stripped == "" or seed_stripped.lower() in {"exit", "quit"}:
            break

        # 1) LLM asks up to 2 questions and returns a final query
        elicited = elicitor.run(seed)
        final_query = elicited.final_query

        # DEBUG
        print(f"\n[DEBUG] Elicitation turns used: {len(elicited.turns)}")
        print(f"[DEBUG] Final query (sent to retrieval): {final_query}\n")

        # 2) Log the elicitation session
        logger.log(
            user_seed=seed,
            turns=elicited.turns,
            final_query=final_query,
            item_types=where_types,
        )

        # 3) Retrieval + formatting
        print("\nSearching for relevant items...\n")
        print("-" * 40)
        print(orchestrator.run_once(final_query, item_types=where_types))
        print("-" * 40)


if __name__ == "__main__":
    main()