import chat_orchestrator
from preference_elicitor_llm import PreferenceElicitorLLM
from elicitation_logger import ElicitationLogger


class ScriptedInput:
    """
    Supplies predefined answers for follow-up questions.
    Falls back to empty string if answers run out.
    """

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.i = 0

    def __call__(self, prompt: str = "") -> str:
        if self.i >= len(self.answers):
            return ""
        ans = self.answers[self.i]
        self.i += 1
        print(ans)  # echo so logs show what was "typed"
        return ans


def main():
    orchestrator = chat_orchestrator.ChatOrchestrator(llm_provider="openai")
    elicitor = PreferenceElicitorLLM(llm=orchestrator.llm_adapter, max_questions=2)

    # Logger for evaluation
    logger = ElicitationLogger(base_dir="logs")

    tests = [
        {
            "label": "Expected 0 questions",
            "seed": (
                "I want a slow, introspective story focused on loneliness and inner reflection, "
                "with a subdued and melancholic emotional tone, and less emphasis on action."
            ),
            "answers": [],
        },
        {
            "label": "Expected 1 question",
            "seed": "I’m interested in stories that explore inner conflict.",
            "answers": [
                "Mainly internal and psychological, and I want it to be darker rather than uplifting."
            ],
        },
        {
            "label": "Expected 2 questions",
            "seed": "I’m looking for something engaging.",
            "answers": [
                "Tense and suspenseful.",
                "Character-driven rather than plot-driven, darker rather than uplifting, with less action and more emotional pressure.",
            ],
        },
    ]

    for t in tests:
        print("\n" + "=" * 90)
        print(t["label"])
        print("- Seed:")
        print(t["seed"])
        print("=" * 90)

        scripted_input = ScriptedInput(t["answers"])

        # 1) Elicit preferences (0/1/2 questions)
        res = elicitor.run(t["seed"], input_fn=scripted_input, print_fn=print)

        print("\n[ELICITATION RESULT]")
        print(f"Questions asked: {len(res.turns)}")

        if res.turns:
            for i, turn in enumerate(res.turns, start=1):
                print(f"\nQ{i}: {turn.question}")
                print(f"A{i}: {turn.answer}")

        print("\nFinal query (sent to retrieval):")
        print(res.final_query)

        # 2) Log elicitation session (CSV + JSONL)
        logger.log(
            user_seed=t["seed"],
            turns=res.turns,
            final_query=res.final_query,
            item_types=None,  # tests are unfiltered
        )

        # 3) Run chatbot response
        print("\n[CHATBOT RESPONSE]")
        print("-" * 90)
        chatbot_output = orchestrator.run_once(res.final_query, item_types=None)
        print(chatbot_output)
        print("-" * 90)

        print("=" * 90)

    print("\nDone. Logs written under: logs/")


if __name__ == "__main__":
    main()