from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import json
import re

from llm_adapter import LLMAdapter


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ElicitationResult:
    final_query: str
    turns: List[Turn]


SYSTEM_PROMPT = """
You are a conversational preference elicitation assistant for a cross-domain recommender system.

Your job:
- Decide whether the user's current information is sufficient to form a strong semantic search query.
- If insufficient, ask up to TWO clarification questions (max 2 total).
- If sufficient, stop immediately and return the final_query.

Hard constraints:
- Never ask about specific titles, actors, directors, authors, popularity, ratings, or release years.
- Avoid genre labels. Focus on experiential preferences only:
  atmosphere/tone, themes/ideas, character dynamics, pacing/energy, tension/conflict, emotional intensity, narrative focus.
- Ask at most ONE question per turn.
- Ask at most TWO questions total.
- Output ONLY valid JSON. No markdown. No extra text.

Sufficiency checklist:
You may STOP (action="stop") if the inputs clearly provide BOTH:
A) One dominant experiential preference (choose at least one):
   - atmosphere/tone (quiet, tense, melancholic, hopeful, etc.)
   - narrative focus (character-driven, world-driven, theme-driven, plot-driven)
   - emotional intensity (restrained vs intense)
   - pacing/energy (slow vs fast)
   - type of conflict (internal vs external)
AND
B) One direction/constraint (choose at least one):
   - more/less of something (more introspection, less action, etc.)
   - similar vs contrasting direction
   - an avoidance constraint (not lighthearted, not action-heavy, etc.)
   - explicit emphasis (atmosphere over events, characters over plot, etc.)

Question selection (very important):
- Ask questions in order of what is missing.
- If A is missing, ask a question that elicits A.
- If A is already present in the seed/history (e.g., "inner conflict" already implies type of conflict),
  then ask for B next (direction/constraint) rather than refining A.
- After one answer, re-check A and B:
  - If both are now clear, STOP (do not ask a second question).
  - Only ask a second question if either A or B is still unclear AND remaining_questions > 0.

Do not waste questions:
- Do NOT ask about both pacing and intensity in two separate questions unless the user's first answer is vague
  (e.g., "not sure", "either", "it depends").

Question UX requirements:
- Each question MUST include 2–4 short example options as bullet points.
- You MUST explicitly state the options are only examples and the user may answer differently.
- Use exactly this line before the bullets:
  "Examples (feel free to answer differently):"

JSON schema:
{
  "action": "ask" | "stop",
  "question": string | null,
  "final_query": string | null
}

Meaning of fields:
- action="ask": provide a single question (with examples), final_query must be null
- action="stop": provide final_query, question must be null

Final query requirements:
- English
- third person, present tense
- compact, synopsis-like description for vector search over plot synopses
- preserve the user's intent and emotional tone exactly
- do NOT introduce new entities, settings, themes, or plot facts not present in the user inputs
""".strip()


def _build_user_prompt(user_seed: str, turns: list[Turn], remaining: int) -> str:
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
        "Return JSON only. Decide ask vs stop using the sufficiency checklist and question selection rules.\n"
    )


def _safe_json_load(raw: str) -> Optional[dict]:
    text = (raw or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(0).strip())
    except Exception:
        return None


def _combined_text(user_seed: str, turns: list[Turn]) -> str:
    parts = [user_seed.strip()]
    for t in turns:
        if t.answer:
            parts.append(t.answer.strip())
    return " ".join([p for p in parts if p]).strip()


class ClarifyGate:
    """
    Optional logic module (v2) that can ask 0, 1, or 2 questions.
    All decisions are LLM-driven based on the prompt rules.
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
        turns: list[Turn] = []
        remaining = self.max_questions

        while True:
            decision = self._llm_decide(user_seed=user_seed, turns=turns, remaining=remaining)

            if decision is None:
                # fallback: no more clarification
                return ElicitationResult(final_query=self._fallback_final_query(user_seed, turns), turns=turns)

            action = decision.get("action")
            question = decision.get("question")
            final_query = decision.get("final_query")

            if action == "ask" and remaining > 0 and isinstance(question, str) and question.strip():
                q = self._ensure_examples_block(question.strip())
                
                # --- FIX START: Correct handling of input_fn for Simulated User ---
                print_fn("\n" + q)
                
                if input_fn == input:
                    # Als het een mens is (standaard CLI), toon prompt-pijltje
                    ans = (input_fn("> ") or "").strip()
                else:
                    # Als het SimulatedUser is, geef de VRAAG mee als argument!
                    ans = (input_fn(q) or "").strip()
                # --- FIX END ---

                if ans == "":
                    # user declines further input -> stop
                    break

                turns.append(Turn(question=q, answer=ans))
                remaining -= 1
                continue

            if action == "stop" and isinstance(final_query, str) and final_query.strip():
                return ElicitationResult(final_query=final_query.strip(), turns=turns)

            break

        # synthesize final query from seed + turns
        synthesized = self._llm_synthesize_final_query(user_seed, turns)
        if synthesized:
            return ElicitationResult(final_query=synthesized, turns=turns)

        return ElicitationResult(final_query=self._fallback_final_query(user_seed, turns), turns=turns)

    def _ensure_examples_block(self, text: str) -> str:
        t = text.strip()
        has_examples_line = "examples (feel free to answer differently):" in t.lower()
        has_bullets = ("\n- " in t) or ("\n• " in t) or ("\n* " in t)

        if has_examples_line and has_bullets:
            return t

        suffix = [
            "Examples (feel free to answer differently):",
            "- more introspective and subtle",
            "- darker and heavier",
            "- lighter and hopeful",
            "- slower and contemplative",
        ]
        return t + "\n" + "\n".join(suffix)

    def _llm_decide(self, user_seed: str, turns: list[Turn], remaining: int) -> Optional[dict]:
        user_prompt = _build_user_prompt(user_seed, turns, remaining)
        raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        return _safe_json_load(raw)

    def _llm_synthesize_final_query(self, user_seed: str, turns: list[Turn]) -> Optional[str]:
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

    def _fallback_final_query(self, user_seed: str, turns: list[Turn]) -> str:
        return _combined_text(user_seed, turns)