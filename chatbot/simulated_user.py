## Python libraries #########################################################################################################################################
#

import os
import re
from pathlib import Path
from working_adapter import WorkingAdapter 

## Setup ####################################################################################################################################################
#

class SimulatedUser:
    def __init__(self, hidden_intent):
        self.hidden_intent = hidden_intent
        self.history = set()
        
        current_dir = Path(__file__).resolve().parent
        config_path = current_dir.parent / 'config.ini' 
        if not config_path.exists():
            config_path = current_dir / 'config.ini'

        self.llm = WorkingAdapter(config_path=str(config_path))

## Main #####################################################################################################################################################
#

    def answer_question(self, question_text):
        """
	Generate answer to chatbot questions
        """
        
        # Keep only the bullet points
        clean_text = question_text.replace("Examples (feel free to answer differently):", "") # Sanitizing prompt to prevent LLM to take to much freedom
        clean_text = clean_text.replace("feel free to answer differently", "")
        
        bullets = re.findall(r"[-*•]\s+([^\n]+)", clean_text)
        
        history_str = ", ".join(self.history) if self.history else "None" # Preventing repeating the answers on the chatbot questions
        
  
        current_temp = 0.0 # Limit llm freedom when generating answers

        if bullets: # Bullet points as possible answers given to llm

            current_temp = 0.0 # Limit llm freedom to 0 when generating answers
            
            menu_str = "\n".join([f"- {opt.strip()}" for opt in bullets])
            
            system_prompt = (
                "You are a Selection Mechanism. You are NOT a chat assistant.\n"
                "TASK: Output EXACTLY one line from the Fixed Menu that matches the Hidden Intent.\n"
                "RULES:\n"
                "1. Output ONLY the word/phrase from the menu.\n"
                "2. Do NOT invent new words."
            )
            
            user_prompt = (
                f"HIDDEN INTENT: {self.hidden_intent}\n"
                f"PREVIOUSLY CHOSEN: {history_str}\n\n"
                f"--- FIXED MENU ---\n"
                f"{menu_str}\n"
                f"------------------\n\n"
                "SELECTION:"
            )
            
        else: # No bullet points as possible answers given to llm
            
            current_temp = 0.3 # Limit llm freedom when generating answers
            
            system_prompt = (
                "You are a cooperative user searching for a movie or book.\n"
                "Answer the chatbot's question naturally based on your Hidden Intent.\n"
                "RULES:\n"
                "1. Keep it short (max 5 words).\n"
                "2. Do NOT repeat exact answers from History.\n"
                "3. Never reveal the exact Title of the item."
            )
            
            user_prompt = (
                f"HIDDEN INTENT: {self.hidden_intent}\n"
                f"HISTORY: {history_str}\n"
                f"QUESTION: {clean_text}\n\n"
                "ANSWER:"
            )

        # Connectiong to llm
        response = self.llm.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=40, 
            temperature=current_temp
        )
        
        final_answer = response.strip().strip(" .-\"'")
        
        if bullets:
            if ":" in final_answer: 
                final_answer = final_answer.split(":")[-1].strip()
            
            # Check if llm choose an option in the bullet point list
            match_found = False
            for b in bullets:
                if b.strip().lower() in final_answer.lower():
                    final_answer = b.strip()
                    match_found = True
                    break
            
            # fallback: force first option
            if not match_found and bullets:
                final_answer = bullets[0].strip()

        self.history.add(final_answer)
        
        return final_answer