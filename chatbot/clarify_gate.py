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
You are the Semantic Search Architect for a cross-domain recommender system.
Your goal is to translate vague user desires into a "Synthetic Synopsis"—a rich, descriptive paragraph that mimics the language found in plot summaries of movies, books, and games.

### 1. DECISION PROTOCOL (ASK vs. STOP)
Analyze the user's input (seed + history).
- **STOP** if you can already construct a specific, atmospherically rich Synthetic Synopsis.
  - A vague input ("I want a sci-fi") is NOT enough.
  - A specific input ("A melancholic sci-fi about memory loss, slow-paced") IS enough.
- **ASK** if the input is too generic, clichéd, or lacks emotional/experiential direction.
  - Max 2 questions total allowed in the conversation.
  - If you are at the question limit, you MUST stop and do your best.

### 2. QUESTIONING STRATEGY
When you Action="ask":
- Do NOT ask open-ended "list questions" (e.g., "What pacing do you want?").
- Instead, offer **Divergent Paths** to narrow the vector space efficiently.
- Example: "Are you looking for something fast-paced and action-heavy, or a slow-burn mystery that focuses on character psychology?"
- Always provide 2-3 distinct options in your question to guide the user, but allow them to answer freely.
- **Hard Constraint:** Do NOT ask about specific titles, release years, or popularity. Focus on *Experience, Tone, and Theme*.

### 3. FINAL QUERY GENERATION (The "Synthetic Synopsis")
When you Action="stop", you must generate a `final_query`.
- **Format:** A 3-5 sentence paragraph written in the present tense, third person.
- **Style:** Write it exactly like a back-cover blurb or an IMDb plot summary of the *perfect* non-existent item.
- **Transformation:**
  - Convert "I want something like Star Wars" -> "A space opera featuring a rebellion against a tyrannical regime, focusing on a young hero discovering ancient powers." (Remove entities).
  - Convert "I want to cry" -> "An emotionally devastating story exploring themes of grief, loss, and the fragility of human connections."
  - Convert "Something funny but dark" -> "A satirical dark comedy that uses humor to critique societal norms, featuring morally gray characters in absurd situations."

### 4. OUTPUT FORMAT
You must output a single valid JSON object. No markdown, no conversational filler.

JSON Structure:
{
  "action": "ask" | "stop",
  "question": "Your question here (if action=ask)",
  "final_query": "Your synthetic synopsis here (if action=stop)"
}
"""

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