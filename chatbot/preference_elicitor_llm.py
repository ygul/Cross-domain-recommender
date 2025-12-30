from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable
import json
import re

from llm_adapter import LLMAdapter


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ElicitationResult:
    final_query: str
    turns: List[Turn]


# ----------------------------
# Prompting
# ----------------------------
SYSTEM_PROMPT = """
You are a conversational preference elicitation assistant for a cross-domain recommender system.

Goal:
- Ask up to TWO clarification questions, only when needed, to create a strong semantic query for vector search.
- The final output must be a single "final_query" that can be embedded for retrieval.

Constraints:
- Never ask about specific titles, actors, directors, authors, popularity, ratings, or release years.
- Avoid genre labels. Focus on experiential aspects: atmosphere, tone, themes, character dynamics, pacing, tension, emotional impact, worldview.
- Keep questions short, concrete, and easy to answer.
- Ask at most ONE question per turn.
- Ask at most TWO questions total.

Stopping guidance:
- You MAY stop after one question if the user's answer provides sufficient information to generate a meaningful recommendation.
- If uncertainty remains about narrative focus, tone, or direction, you SHOULD ask a second question (if remaining_questions > 0).

Output format:
- Output ONLY valid JSON. No markdown. No extra text.

JSON schema:
{
  "action": "ask" | "stop",
  "question": string | null,
  "final_query": string | null
}

Meaning of fields:
- action="ask": provide a single question, final_query must be null
- action="stop": provide final_query, question must be null

Final query requirements:
- English
- third person, present tense
- compact, synopsis-like description
- preserve the user's intent and emotional tone exactly
- do NOT introduce new entities, settings, themes, or plot facts not present in the user inputs
""".strip()


def _build_user_prompt(user_seed: str, turns: List[Turn], remaining: int) -> str:
    if turns:
        history = "\n\n".join([f"Q: {t.question}\nA: {t.answer}" for t in turns])
    else:
        history = "N/A"

    return (
        "INPUT\n"
        f"user_seed:\n{user_seed}\n\n"
        f"history:\n{history}\n\n"
        f"remaining_questions:\n{remaining}\n\n"
        "TASK\n"
        "Decide whether to ask ONE more clarification question or stop.\n"
        "Return JSON only.\n"
    )


# ----------------------------
# JSON parsing helpers
# ----------------------------
def _safe_json_load(raw: str) -> Optional[dict]:
    """
    Try parsing JSON even if the model wrapped it in extra text.
    Strategy:
    1) direct json.loads
    2) extract first {...} block via regex
    """
    text = (raw or "").strip()
    if not text:
        return None

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) extract a JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0).strip()

    try:
        return json.loads(candidate)
    except Exception:
        return None


def _combined_text(user_seed: str, turns: List[Turn]) -> str:
    parts = [user_seed.strip()]
    for tr in turns:
        if tr.answer:
            parts.append(tr.answer.strip())
    return " ".join([p for p in parts if p]).strip()


# ----------------------------
# Main class
# ----------------------------
class PreferenceElicitorLLM:
    """
    Runs a short elicitation: up to 2 questions.
    This variant leaves the decision entirely to the LLM (no heuristics/guardrails).
    """

    def __init__(self, llm: LLMAdapter, max_questions: int = 2) -> None:
        self.llm = llm
        self.max_questions = max_questions

    def run(
        self,
        user_seed: str,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
    ) -> ElicitationResult:
        turns: List[Turn] = []
        remaining = self.max_questions

        while True:
            decision = self._llm_decide(user_seed=user_seed, turns=turns, remaining=remaining)

            # If LLM output fails: fallback immediately
            if decision is None:
                final_query = self._fallback_final_query(user_seed, turns)
                return ElicitationResult(final_query=final_query, turns=turns)

            action = decision.get("action")
            question = decision.get("question")
            final_query = decision.get("final_query")

            # Ask path
            if action == "ask" and remaining > 0 and isinstance(question, str) and question.strip():
                q = question.strip()
                print_fn("\n" + q)
                ans = (input_fn("> ") or "").strip()

                # Empty answer => stop early
                if ans == "":
                    break

                turns.append(Turn(question=q, answer=ans))
                remaining -= 1
                continue

            # Stop path
            if action == "stop" and isinstance(final_query, str) and final_query.strip():
                return ElicitationResult(final_query=final_query.strip(), turns=turns)

            # Unexpected output or no remaining questions
            break

        # If we exit without a valid final_query, synthesize it once from gathered text
        synthesized = self._llm_synthesize_final_query(user_seed, turns)
        if synthesized:
            return ElicitationResult(final_query=synthesized, turns=turns)

        # Last resort fallback
        return ElicitationResult(final_query=self._fallback_final_query(user_seed, turns), turns=turns)

    # ----------------------------
    # LLM calls
    # ----------------------------
    def _llm_decide(self, user_seed: str, turns: List[Turn], remaining: int) -> Optional[dict]:
        user_prompt = _build_user_prompt(user_seed, turns, remaining)
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        return _safe_json_load(raw)

    def _llm_synthesize_final_query(self, user_seed: str, turns: List[Turn]) -> Optional[str]:
        """
        One final call that outputs ONLY rewritten query text (not JSON).
        Used when the stop decision was not properly returned as JSON.
        """
        combined = _combined_text(user_seed, turns)
        system = (
            "You rewrite user preference text into a compact semantic query for vector search over plot synopses.\n"
            "Rules:\n"
            "- English, third person, present tense\n"
            "- Preserve intent and emotional tone exactly\n"
            "- Do not add new entities, settings, themes, or plot facts\n"
            "- Output ONLY the rewritten query text\n"
        )
        out = (self.llm.generate(system, combined) or "").strip()
        return out or None

    # ----------------------------
    # Fallback
    # ----------------------------
    def _fallback_final_query(self, user_seed: str, turns: List[Turn]) -> str:
        """
        Minimal safe fallback. Your QueryEmbedder may rewrite anyway.
        """
        return _combined_text(user_seed, turns)
