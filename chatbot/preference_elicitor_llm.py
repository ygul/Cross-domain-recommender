import re

class ElicitationResult:
    def __init__(self, conversation_history, final_query):
        self.conversation_history = conversation_history
        self.final_query = final_query

class PreferenceElicitorLLM:
    def __init__(self, llm, max_questions=2):
        self.llm = llm
        self.max_questions = max_questions
        # Systeem prompt om de AI te dwingen vragen te stellen of samen te vatten
        self.system_prompt_ask = (
            "You are a movie/book recommendation assistant. "
            "Your goal is to understand the user's specific taste. "
            "Ask a SHORT, clarifying question based on their input. "
            "Do NOT recommend items yet."
        )
        self.system_prompt_summarize = (
            "You are a search query generator. "
            "Based on the conversation history, generate a single, descriptive search query "
            "that captures the user's specific intent, genre, and mood. "
            "Output ONLY the query."
        )

    def run(self, user_seed, input_fn, print_fn=print):
        """
        Voert de dialoog in stappen:
        1. Start met seed
        2. Loop 'max_questions' keer om vragen te stellen
        3. Genereer uiteindelijke zoekvraag
        """
        history = [f"User: {user_seed}"]
        context_str = f"User: {user_seed}"

        # --- FASE 1: Vragen stellen (De Loop) ---
        for i in range(self.max_questions):
            # 1. AI genereert een vraag
            prompt = f"CONVERSATION:\n{context_str}\n\nTASK: Ask a follow-up question."
            ai_question = self.llm.generate(self.system_prompt_ask, prompt)
            
            print_fn(f"[Bot Q{i+1}]: {ai_question}")
            history.append(f"AI: {ai_question}")
            context_str += f"\nAI: {ai_question}"

            # 2. Gebruiker (of SimUser) geeft antwoord
            # We voegen hier de vraag toe zodat input_fn weet wat er gevraagd werd
            user_answer = input_fn(f"{ai_question}") 
            
            history.append(f"User: {user_answer}")
            context_str += f"\nUser: {user_answer}"

        # --- FASE 2: Samenvatten (De Final Query) ---
        summary_prompt = f"CONVERSATION:\n{context_str}\n\nTASK: Generate the final search query."
        final_query = self.llm.generate(self.system_prompt_summarize, summary_prompt)
        
        # Schoonmaak (verwijder quotes of 'Search query:' tekst)
        final_query = re.sub(r'^["\']|["\']$', '', final_query.strip())
        final_query = re.sub(r'^(Search query:|Query:)\s*', '', final_query, flags=re.IGNORECASE)

        return ElicitationResult(history, final_query)