from __future__ import annotations

from dataclasses import dataclass
from typing import List
import json

from llm_adapter import LLMAdapter


SYSTEM_PROMPT = """
You are a conversational preference elicitation assistant for a cross-domain recommender.

Rules:
- Ask at most TWO questions in total.
- Each turn, output ONLY valid JSON (no markdown, no extra text).
- Never ask about genres, ratings, popularity, or specific titles/creators.
- Focus on narrative experience: atmosphere, themes, character dynamics, tension, pacing, worldview.
- Keep questions short and practical.
- If you already have enough information, stop asking questions.

Input you receive:
- user_seed: the user's initial request
- history: prior questions and answers (may be empty)
- remaining_questions: number of questions you are still allowed to ask

Decision:
- If remaining_questions > 0 AND information is insufficient: action="ask" with ONE concise question.
- Otherwise: action="stop" and provide final_query in English, third-person, present tense, synopsis-like,
  using ONLY concepts mentioned by the user in user_seed and history (no new entities, settings, or themes).

JSON schema (must always comply):
{
  "action": "ask" | "stop",
  "question": "string | null",
  "final_query": "string | null"
}
""".strip()


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ElicitationResult:
    final_query: str
    turns: List[Turn]


class PreferenceElicitorLLM:
    def __init__(self, llm: LLMAdapter, max_questions: int = 2) -> None:
        self.llm = llm
        self.max_questions = max_questions

    def run(self, user_seed: str, input_fn=input, print_fn=print) -> ElicitationResult:
        turns: List[Turn] = []
        remaining = self.max_questions

        while True:
            user_prompt = self._build_user_prompt(user_seed, turns, remaining)
            raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)

            data = self._safe_json_load(raw)
            if data is None:
                final_query = self._fallback_final_query(user_seed, turns)
                return ElicitationResult(final_query=final_query, turns=turns)

            action = data.get("action")
            question = data.get("question")
            final_query = data.get("final_query")

            # Ask
            if action == "ask" and remaining > 0 and isinstance(question, str) and question.strip():
                print_fn("\n" + question.strip())
                ans = input_fn("> ")
                ans = (ans or "").strip()

                # If user gives empty answer, stop early (no point burning the 2 turns)
                if ans == "":
                    final_query = self._fallback_final_query(user_seed, turns)
                    return ElicitationResult(final_query=final_query, turns=turns)

                turns.append(Turn(question=question.strip(), answer=ans))
                remaining -= 1
                continue

            # Stop
            if action == "stop" and isinstance(final_query, str) and final_query.strip():
                return ElicitationResult(final_query=final_query.strip(), turns=turns)

            final_query = self._fallback_final_query(user_seed, turns)
            return ElicitationResult(final_query=final_query, turns=turns)

    def _build_user_prompt(self, user_seed: str, turns: List[Turn], remaining: int) -> str:
        if turns:
            history_text = "\n\n".join([f"Q: {t.question}\nA: {t.answer}" for t in turns])
        else:
            history_text = "N/A"

        return (
            f"user_seed:\n{user_seed}\n\n"
            f"history:\n{history_text}\n\n"
            f"remaining_questions:\n{remaining}\n"
        )

    def _safe_json_load(self, raw: str):
        """
        Parse JSON robustly. If the model wraps JSON in extra text,
        try extracting the first {...} block.
        """
        text = (raw or "").strip()
        if not text:
            return None

        # Direct attempt
        try:
            return json.loads(text)
        except Exception:
            pass

        # Extract first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None

        return None

    def _fallback_final_query(self, user_seed: str, turns: List[Turn]) -> str:
        """
        Minimal safe fallback: keep only user text.
        Your QueryEmbedder may rewrite to English anyway.
        """
        parts = [user_seed.strip()]
        for t in turns:
            if t.answer:
                parts.append(t.answer.strip())
        return " ".join([p for p in parts if p]).strip()
