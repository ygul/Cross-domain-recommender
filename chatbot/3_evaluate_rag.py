## Python libraries #########################################################################################################################################
#

import time
import pandas as pd
import re
import configparser
from pathlib import Path
from huggingface_hub import InferenceClient

import chat_orchestrator
from preference_elicitor_llm import PreferenceElicitorLLM
from query_embedder import QueryEmbedder

## Setup ####################################################################################################################################################
#

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# Judge setup (HuggingFace)
JUDGE_MODEL = config["AI"]["model_judge"]
HF_TOKEN = config["AI"]["hf_token"]
client_judge = InferenceClient(token=HF_TOKEN)

# System-to-test setup
print("Initializing Chat Orchestrator")
# 1. Deze regel MOET er staan (anders krijg je NameError: orchestrator not defined)
orchestrator = chat_orchestrator.ChatOrchestrator(llm_provider="openai", max_items=3)

# 2. Deze regel heb je net toegevoegd voor de embedder fix
print("Initializing Independent Embedder")
test_embedder = QueryEmbedder(model_name=config["AI"]["model_embedding"])

## Test Data #################################################################################################################################################
#

TEST_SCENARIOS = [
    {
        "id": 1, # Scenario 1
        "max_questions": 1,  # Direct answer, evaulate single shot performance
        "seed": "I want a psychological thriller.",
        "hidden_intent": "I want a book about isolation and madness, specifically The Shining style. Not too modern.",
    },
    {
        "id": 2, # Scenario 2
        "max_questions": 2,  # Evaulate 2 shot performance
        "seed": "Recommend something funny about work.",
        "hidden_intent": "I like satire about corporate life, awkward situations, mockumentary style like The Office.",
    },
    {
        "id": 3, # Scenario 3
        "max_questions": 4,  # Evaulate long conversation performance
        "seed": "I want a dark story about technology.",
        "hidden_intent": "I am interested in dystopian futures where tech ruins society, like Black Mirror or Ex Machina. I prefer slow-paced, philosophical stories over action.",
    },
    {
        "id": 4, # Scenario 4
        "max_questions": 3,  # Evaulate long conversation performance
        "seed": "Stories about survival competitions.",
        "hidden_intent": "I loved Hunger Games but I want the darker, older version like The Long Walk or Battle Royale. Focus on psychological trauma.",
    },
    {
        "id": 5, # Scenario 5
        "max_questions": 2,  # Evaulate 2 shot performance
        "seed": "I am looking for a love story.",
        "hidden_intent": "I do NOT want a happy rom-com. I want a tragic, realistic romance with emotional depth, like Blue Valentine or Normal People.",
    },
    {
        "id": 6, # Scenario 6
        "max_questions": 4,  # Evaulate long conversation performance
        "seed": "Something with space travel.",
        "hidden_intent": "I want a gritty, political space opera involving complex factions and war, not just action-adventure. Like The Expanse or Dune.",
    },
    {
        "id": 7, # Scenario 7
        "max_questions": 1, # Zou direct duidelijk moeten zijn
        "seed": "A mystery story.",
        "hidden_intent": "I want a 'cozy mystery' set in a small village or old house, focused on puzzle solving, not violent police procedurals.",
    },
    {
        "id": 8, # Scenario 8
        "max_questions": 3,  # Evaulate long conversation performance
        "seed": "I like stories with magic.",
        "hidden_intent": "I prefer 'Magical Realism' or low fantasy where magic is subtle and mysterious, set in the real world. Like Pan's Labyrinth.",
    },
    {
        "id": 9, # Scenario 9
        "max_questions": 2,  # Evaulate 2 shot performance
        "seed": "I want to watch something historical.",
        "hidden_intent": "I want an immersive period drama about gangsters or crime in the 1920s/30s. Like Peaky Blinders.",
    },
    {
        "id": 10, # Scenario 10
        "max_questions": 2,  # Evaulate 2 shot performance
        "seed": "A story about growing up.",
        "hidden_intent": "I want a nostalgic coming-of-age story set in the 80s, focusing on friendship and maybe a bit of mystery. Like Stranger Things.",
    }
]

## Helper functies ###########################################################################################################################################
#

# Extract score
def extract_score(text):
    if not text: return 0
    match = re.search(r'Score:\s*([1-5])', text, re.IGNORECASE)
    if not match:
        match = re.search(r'\b([1-5])\b', text)
    return int(match.group(1)) if match else 0

# Extract reasoning
def extract_reason(text):
    if not text: return "No response"
    match = re.search(r'Reason:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r'Score:\s*\d+', '', text, flags=re.IGNORECASE).strip()
    return cleaned

# Initialize Judge
def ask_judge(system_role, user_content):
    example_input = "CONTEXT: movie A is fun.\nANSWER: movie A is great.\nTASK: Check Factuality."
    example_output = "Score: 5\nReason: The answer matches the context perfectly."

    messages = [
        {
            "role": "system", 
            "content": (
                f"{system_role}\n"
                "You are an automated scoring engine. "
                "You must ONLY output the Score (1-5) and a one-sentence Reason. "
                "Do NOT write stories. Do NOT chat."
            )
        },
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": user_content}
    ]

    try:
        response = client_judge.chat_completion(
            messages=messages,
            model=JUDGE_MODEL,
            max_tokens=100,
            temperature=0.1, 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return ""

class SimulatedUser:
    def __init__(self, intent):
        self.intent = intent

    def answer_question(self, question):
        sys_prompt = (
            "You are a user participating in a test."
            "Your Goal: Answer the chatbot's question based on your HIDDEN INTENT."
            "CONSTRAINT 1: Do NOT suggest specific books/movies yourself."
            "CONSTRAINT 2: Only describe your feelings, preferences, or what you are looking for."
            "CONSTRAINT 3: Keep it short (max 2 sentences)."
        )
        user_prompt = f"HIDDEN INTENT: {self.intent}\n\nCHATBOT QUESTION: {question}\n\nYour Answer:"
        return orchestrator.llm_adapter.generate(sys_prompt, user_prompt)

## Performance metrics #######################################################################################################################################
#

# Evaluate precision
def evaluate_precision(query, context_docs, answer):
    short_context = context_docs[:1500].replace("\n", " ")
    prompt = f"""
	TASK: Rate FACTUALITY (1-5).
	CONTEXT: {short_context}
	ANSWER: {answer}

	INSTRUCTIONS:
	- Score 5: Answer uses ONLY the Context.
	- Score 1: Answer makes up facts or cites items NOT in context.
	- Output format: 'Score: X' then 'Reason: ...'
	"""
    response = ask_judge("You are a strict Hallucination Checker.", prompt)
    score = extract_score(response)
    if score == 0 and response:
        print(f"⚠️⚠️⚠️ Judge failed at precision: {response[:50]}...")
    return score, response

# Evaluate user satisfaction
def evaluate_usr(user_intent, answer):
    prompt = f"""
	TASK: Rate SATISFACTION (1-5).
	USER INTENT: {user_intent}
	ANSWER: {answer}

	INSTRUCTIONS:
	- Score 5: Perfect recommendation matching the intent.
	- Score 1: Completely irrelevant recommendation.
	- Output format: 'Score: X' then 'Reason: ...'
	"""
    response = ask_judge("You are a Quality Assurance bot.", prompt)
    score = extract_score(response)
    if score == 0 and response:
        print(f"⚠️⚠️⚠️ Judge failed at user satisfaction: {response[:50]}...")
    return score, response

## Main script ###############################################################################################################################################
#

results_data = []

print(f"\n =================== Starting automated evaluation of ({len(TEST_SCENARIOS)} Scenarios)=================== \n")

for scenario in TEST_SCENARIOS:
    max_q = scenario.get("max_questions", 2)
    print(f"Scenario {scenario['id']}: {scenario['seed']} (Max no. questions: {max_q})")
    
    # New elicitor with limit for current scenario
    elicitor = PreferenceElicitorLLM(llm=orchestrator.llm_adapter, max_questions=max_q)
    
## Step 1: Single shot
    baseline_output = orchestrator.run_once(scenario['seed'])
    
    # Context
    emb = test_embedder.embed(scenario['seed'])
    raw_results = orchestrator.vector_store.similarity_search(emb, k=3)
    context_text = "\n".join([r.document for r in raw_results if r.document])
    
    # Judge
    pr_score_1, pr_raw_1 = evaluate_precision(scenario['seed'], context_text, baseline_output)
    usr_score_1, usr_raw_1 = evaluate_usr(scenario['hidden_intent'], baseline_output)
    
    print(f"   -> Baseline USR: {usr_score_1}/5 | PR: {pr_score_1}/5")

## Step 2: Dialogue
    sim_user = SimulatedUser(scenario['hidden_intent'])
    
    def automated_input(prompt_text):
        display_q = prompt_text.split('\n')[0] 
        print(f"      [Bot]: {display_q[:60]}...")
        ans = sim_user.answer_question(prompt_text)
        print(f"      [User]: {ans}")
        return ans

    elicitation_result = elicitor.run(
        user_seed=scenario['seed'],
        input_fn=automated_input,
        print_fn=lambda x: None
    )
    
    final_query = elicitation_result.final_query
    print(f"      [Final Query]: {final_query}")
    
    final_output = orchestrator.run_once(final_query)
    
    emb_final = test_embedder.embed(final_query)
    raw_results_final = orchestrator.vector_store.similarity_search(emb_final, k=3)
    context_text_final = "\n".join([r.document for r in raw_results_final if r.document])

    # Judge
    pr_score_2, pr_raw_2 = evaluate_precision(final_query, context_text_final, final_output)
    usr_score_2, usr_raw_2 = evaluate_usr(scenario['hidden_intent'], final_output)
    
    print(f"   -> Final USR: {usr_score_2}/5 | PR: {pr_score_2}/5")

## Step 3: RRI
    rri_score = usr_score_2 - usr_score_1
    print(f"   -> RRI: {rri_score}")

    # Results appending
    results_data.append({
        "Scenario ID": scenario['id'],
        "Max Questions": max_q,
        "Start question": scenario['seed'],
        "User intent (hidden)": scenario['hidden_intent'],
        "USR (Single shot)": usr_score_1,
        "PR (Single shot)": pr_score_1,
        "USR (Final)": usr_score_2,
        "PR (Final)": pr_score_2,
        "RRI": rri_score,
        "Judge Reasoning": extract_reason(usr_raw_2)
    })
    
    time.sleep(1)

## Exporting functions #######################################################################################################################################
#
if results_data:
    df = pd.DataFrame(results_data)
    output_file = BASE_DIR / "rag_evaluatie_metrics.xlsx"
    
    print(f"REPORT SUMMARY")
    print(f"Average Precision (PR): {df['PR (Final)'].mean():.2f}")
    print(f"Average Satisfaction (USR): {df['USR (Final)'].mean():.2f}")
    print(f"Average Improvement (RRI): {df['RRI'].mean():.2f}")
    print("="*40)

    try:
        df.to_excel(output_file, index=False)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving Excel: {e}")
else:
    print("No results to save.")