## Python libraries #########################################################################################################################################
#

import os
import re
from pathlib import Path
from llm_adapter import create_llm_adapter 

## Setup ####################################################################################################################################################
#

class SimulatedUser:
    def __init__(self, hidden_intent):
        self.hidden_intent = hidden_intent
        self.history = set()
        self.llm = create_llm_adapter()

## Main #####################################################################################################################################################
#

    def answer_question(self, question_text):
        """
        Generate a short answer to chatbot questions (max 5 words)
        """
        clean_text = re.sub(r'[^a-zA-Z0-9\s\?\.,]', '', question_text).strip()
        history_str = ", ".join(self.history) if self.history else "None"
        
        system_prompt = (
            "You are a cooperative user searching for a movie, a book or a series.\n"
            "Answer the chatbot's question naturally based on your Hidden Intent.\n"
            "RULES:\n"
            "1. Keep it short (max 5 words).\n"
            "2. Do NOT repeat exact answers from History.\n"
            "3. Never reveal the exact Title of the item.\n"
            "4. Aim to get closer to the Hidden Intent.\n"
            "5. Ignore the examples given. Your focus is on the Hidden Intent only.\n"
        )
        
        user_prompt = (
            f"HIDDEN INTENT: {self.hidden_intent}\n"
            f"HISTORY: {history_str}\n"
            f"QUESTION: {clean_text}\n\n"
            "ANSWER:"
        )

        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        final_answer = response.strip().strip(" .-\"'")
        
        if ":" in final_answer:
            final_answer = final_answer.split(":")[-1].strip()

        self.history.add(final_answer)
        
        return final_answer