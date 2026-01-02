def run_dynamic_session(orchestrator, sim_user, seed_text)
    
    Laat de ChatOrchestrator (en dus ClarifyGate) bepalen hoe het gesprek loopt.
    
    history = []
    current_input = seed_text
    
    # Maximaal 5 beurten om oneindige loops te voorkomen
    for _ in range(5)
        # 1. Stuur input naar de ECHTE orchestrator logic
        # (Dit activeert intern de ClarifyGate)
        response = orchestrator.chat(user_message=current_input, history=history)
        
        # 2. Check Is het een vraag of een aanbeveling
        # (Afhankelijk van hoe je chat() output eruit ziet, vaak is dat een dict of object)
        
        # SCENARIO HET IS EEN AANBEVELING (EINDE)
        if recommendations in response 
            return response[final_query], response[recommendations]
            
        # SCENARIO HET IS EEN VRAAG (DOORGAAN)
        else
            bot_question = response[response_text]
            print(f[Dynamic Bot] {bot_question})
            
            # Laat de SimUser antwoorden op basis van Hidden Intent
            user_answer = sim_user.answer_question(bot_question)
            print(f[Dynamic User] {user_answer})
            
            # Update history en input voor de volgende ronde
            history.append({role user, content current_input})
            history.append({role assistant, content bot_question})
            current_input = user_answer

    return current_input, [] # Fallback als het te lang duurt