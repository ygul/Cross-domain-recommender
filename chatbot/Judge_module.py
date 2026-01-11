## Python libraries #########################################################################################################################################
#

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from chat_orchestrator import ChatOrchestrator
from simulated_user import SimulatedUser
from query_embedder import QueryEmbedder
from llm_adapter import create_llm_adapter
from metrics import evaluate_usr, calculate_average_similarity

## Setup ####################################################################################################################################################
#


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


SCENARIO_FILE = BASE_DIR / "test_scenarios.txt"

## Helper fucntions #########################################################################################################################################
#

## loading scenarios from test_scenarios.txt
def load_scenarios(filepath):
    scenarios = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    scenarios.append({
                        "id": parts[0].strip(),
                        "max_questions": int(parts[1].strip()),
                        "seed": parts[2].strip(),
                        "hidden_intent": parts[3].strip()
                    })

    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
    return scenarios

## calculate cosine similarity
def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    
    a = np.array(vec_a).flatten()
    b = np.array(vec_b).flatten()
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a, b) / (norm_a * norm_b)


## Judge ####################################################################################################################################################
#

def main():

    print("\n Running chatbot evaluation using LLM-as-a-Judge")
    
    ## Initialization
    orchestrator = ChatOrchestrator() 
    metrics_embedder = QueryEmbedder() 
    judge_adapter = create_llm_adapter(temperature=0.0) # Low temperature for deterministic judging
    
    ## Get scenarios
    scenarios = load_scenarios(SCENARIO_FILE)
    print(f"Loaded {len(scenarios)} scenarios from {SCENARIO_FILE}")

    results_data = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO {scenario['id']}: {scenario['seed']}")
        print(f"HIDDEN INTENT: {scenario['hidden_intent']}")
        print(f"{'='*60}")
        
        # Create simulated user
        sim_user = SimulatedUser(scenario['hidden_intent'])
        
        ## Single turn answering

        print("\n Single turn mode")
        
        # De orchestrator voert 1 keer de zoekslag uit
        single_turn_response = orchestrator.run_once(scenario['seed'])
        
        # Get the actual vector search results that were used
        search_results_singleturn = orchestrator.get_last_search_results()
        print(f"[Vector DB]: Found {len(search_results_singleturn)} items with embeddings: {[r.embedding[:5] if r.embedding else None for r in search_results_singleturn]}")
                
        print(f"[Bot final answer]: {single_turn_response}")
        
        ## Multi turn answering

        print("\n Multi turn mode")
        
        # Hook function: Connecting orchestrator to simulated User
        def automated_input_hook(bot_question):
            instruction = "(Answer in max 5 words)" # Answer in max 5 words
            full_prompt = f"{bot_question}\n{instruction}"
            
            answer = sim_user.answer_question(full_prompt)
            print(f"[Simulated user answer]: {answer}")
            return answer

        ## Start conversation        
        multi_turn_response_text, result_obj = orchestrator.chat(
            seed=scenario['seed'], 
            input_fn=automated_input_hook,
            print_fn=print # Show chatbot questions in console
        )

        # Get the actual vector search results that were used
        search_results_multiturn = orchestrator.get_last_search_results()
        print(f"[Vector DB]: Found {len(search_results_multiturn)} items with embeddings: {[r.embedding[:5] if r.embedding else None for r in search_results_multiturn]}")
        
        print(f"\n[Bot Final Answer (multi-turn)]: {multi_turn_response_text}")
        
        ## Calculate metrics

        print("\n[Calculating Metrics]")
        
        # Embeddings - calculate average similarity with all found items
        cos_sim_intent_single = calculate_average_similarity(scenario['hidden_intent'], search_results_singleturn, metrics_embedder)
        cos_sim_intent_multi = calculate_average_similarity(scenario['hidden_intent'], search_results_multiturn, metrics_embedder)
        
        # Calculate RRI based on intent similarity
        rri_cos_sim = cos_sim_intent_multi - cos_sim_intent_single

        # Judge
        score_single, reason_single = evaluate_usr(
            scenario['hidden_intent'], 
            single_turn_response, 
            judge_adapter
        )
        
        score_multi, reason_multi = evaluate_usr(
            scenario['hidden_intent'], 
            multi_turn_response_text, 
            judge_adapter
        )
        
        rri_judge = score_multi - score_single

        print(f"- Judge scores: Single {score_single}/5 | Multi {score_multi}/5 | RRI: {rri_judge}")
        print(f"- Intent similarity  	: Single {cos_sim_intent_single:.3f} | Multi {cos_sim_intent_multi:.3f}")


        ## Save results

        results_data.append({
            "Scenario ID": scenario['id'],
            "Seed": scenario['seed'],
            "Hidden Intent": scenario['hidden_intent'],
            
            # Single-turn data
            "Single Turn Response": single_turn_response[:200] + "...",
            "Score (Single)": score_single,
            "Reason (Single)": reason_single,
            "Intent Sim (Single)": round(cos_sim_intent_single, 4),
            
            # Multi-turn data
            "Multi Turn Response": multi_turn_response_text[:200] + "...",
            "Score (Multi)": score_multi,
            "Reason (Multi)": reason_multi,
            "Intent Sim (Multi)": round(cos_sim_intent_multi, 4),
            
            # RRI values
            "RRI (Judge)": rri_judge,
            "RRI (Intent Sim)": round(rri_cos_sim, 4),
            "Turns Used": len(result_obj.turns) if result_obj else 0
        })
        
        time.sleep(1) # Rate limit

    ## Export

    if results_data:
        df = pd.DataFrame(results_data)
        output_file = BASE_DIR / "rag_evaluation_final.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\nEvaluation Complete. Results saved to {output_file}")

    else:
        print("No results generated.")

if __name__ == "__main__":
    main()