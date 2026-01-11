"""
Metrics module for evaluating recommender system performance.

This module provides evaluation metrics and LLM-as-a-Judge functionality
for assessing recommendation quality in the cross-domain recommender system.
"""

from typing import Tuple
from llm_adapter import LLMAdapter, create_llm_adapter

###############################################################################################################
# USR (User Satisfaction Rating) Rubric for LLM Judge
###############################################################################################################
USR_RUBRIC = """
- Score 1 (Bad): The response is completely irrelevant, hallucinates facts, or ignores the user's hidden intent entirely.
- Score 2 (Poor): The response touches on the topic but recommends the wrong items or misses the core mood/style of the hidden intent.
- Score 3 (Average): The response is acceptable and relevant, but generic. It fits the broad category but lacks specific nuance from the hidden intent.
- Score 4 (Good): The response matches the hidden intent well and recommends items that fit the specific description (e.g. isolation/madness).
- Score 5 (Excellent): The response perfectly captures the hidden intent. The recommendation is exactly what the user implicitly wanted.
"""

def evaluate_usr(hidden_intent: str, model_response: str, llm_adapter: LLMAdapter) -> Tuple[int, str]:
    """
    Evaluate User Satisfaction Rating (USR) using an LLM as a judge.
    
    This function uses an LLM to evaluate how well a recommender system's response
    satisfies a user's hidden intent, following the USR rubric (1-5 scale).
    
    Args:
        hidden_intent: The user's underlying need or preference that wasn't explicitly stated
        model_response: The recommender system's response to evaluate
        llm_adapter: An LLMAdapter instance (e.g., OpenAIAdapter) with generate() method
        
    Returns:
        tuple: (score, reason) where:
            - score (int): USR score from 1 (Bad) to 5 (Excellent)
            - reason (str): Explanation for the score from the judge
            
    Examples:
        >>> from llm_adapter import create_llm_adapter
        >>> adapter = create_llm_adapter(provider='openai', model='gpt-4o-mini', temperature=0.0)
        >>> score, reason = evaluate_usr(
        ...     "Books about isolation and madness",
        ...     "I recommend 'The Shining' by Stephen King",
        ...     adapter
        ... )
        >>> print(f"Score: {score}/5 - {reason}")
        Score: 5/5 - Perfect match for isolation/madness theme
    """
    system_prompt = f"""You are an expert judge evaluating a recommender system.
    
    RUBRIC FOR SCORING:
    {USR_RUBRIC}
    
    Output format:
    SCORE: [1-5]
    REASON: [Short explanation]
    """
    
    user_prompt = f"""
    User Hidden Intent: "{hidden_intent}"
    Chatbot Response: "{model_response}"
    
    Evaluate how well the response satisfies the hidden intent based on the rubric.
    """
    
    raw_output = llm_adapter.generate(system_prompt, user_prompt)
    
    # Scoring + motivation
    score = 1
    reason = "Parse error"
    
    try:
        lines = raw_output.split('\n')
        for line in lines:
            if "SCORE:" in line:
                score_str = line.split("SCORE:")[1].strip() # Select first character if starts with a number ('4 (Good)')
                score = int(score_str[0])
 
            if "REASON:" in line:
                reason = line.split("REASON:")[1].strip()
        
        # Fallback als REASON niet op een nieuwe regel staat
        if reason == "Parse error" and len(lines) > 0:
            reason = raw_output.replace('\n', ' ')
            
    except Exception as e:
        print(f"Judge parse warning: {raw_output} - Error: {e}")
        
    return score, reason


###############################################################################################################
# Smoke Tests
###############################################################################################################

# Test data for smoke tests
EVALUATE_USR_TEST_CASES = [
    {
        "name": "Perfect Match (Expected Score: 5)",
        "hidden_intent": "Dark psychological thrillers about isolation and madness",
        "model_response": "Based on your interest in psychological depth, I recommend: 1) 'The Shining' by Stephen King - a masterpiece about isolation-induced madness in a remote hotel, 2) 'Black Swan' (film) - exploring psychological breakdown in the competitive ballet world, 3) 'True Detective' Season 1 - dark psychological crime series examining human darkness and mental deterioration.",
        "expected_score_range": (4, 5)
    },
    {
        "name": "Good Match (Expected Score: 4)", 
        "hidden_intent": "Uplifting stories about friendship and personal growth",
        "model_response": "Here are some great recommendations: 1) 'The Pursuit of Happyness' - inspiring story about perseverance and father-son bond, 2) 'Anne of Green Gables' by L.M. Montgomery - heartwarming tale of friendship and growing up, 3) 'Ted Lasso' series - comedy-drama about kindness, friendship, and positive leadership transforming people's lives.",
        "expected_score_range": (3, 5)
    },
    {
        "name": "Average Match (Expected Score: 3)",
        "hidden_intent": "Historical fiction set during World War II",
        "model_response": "I can suggest these titles: 1) 'The Book Thief' by Markus Zusak - set in Nazi Germany, 2) 'Saving Private Ryan' - World War II film about soldiers, 3) 'Band of Brothers' - miniseries following Easy Company through the war. These are all well-regarded historical works.",
        "expected_score_range": (2, 4)
    },
    {
        "name": "Poor Match (Expected Score: 2)",
        "hidden_intent": "Romantic comedies with strong female leads",
        "model_response": "Here are some recommendations: 1) 'The Godfather' - classic crime drama about family loyalty, 2) 'Breaking Bad' - intense series about a chemistry teacher turned drug dealer, 3) 'Gone Girl' by Gillian Flynn - psychological thriller about a troubled marriage. These are all highly acclaimed works.",
        "expected_score_range": (1, 3)
    },
    {
        "name": "Terrible Match (Expected Score: 1)",
        "hidden_intent": "Light children's fantasy books for bedtime reading",
        "model_response": "I suggest these intense options: 1) 'American Psycho' by Bret Easton Ellis - disturbing psychological horror novel, 2) 'Saw' film series - graphic horror about life-or-death games, 3) 'The Walking Dead' - zombie apocalypse series with extreme violence. These will definitely keep you engaged.",
        "expected_score_range": (1, 2)
    }
]


def run_evaluate_usr_smoke_test():
    """
    Run smoke tests for the evaluate_usr function using structured test cases.
    
    Tests the function with 5 scenarios ranging from perfect matches to terrible matches,
    using real OpenAI API calls to verify that the scoring logic works correctly.
    """
    print("🧪 Running evaluate_usr smoke tests with real OpenAI API...")
    print("=" * 80)
    
    try:
        # Create real OpenAI adapter for judge evaluation
        judge_adapter = create_llm_adapter(temperature=0.0)
        print("✅ OpenAI adapter initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI adapter: {e}")
        print("   Make sure OPENAI_API_KEY is set in your .env file")
        return False
    all_passed = True
    
    for i, test_case in enumerate(EVALUATE_USR_TEST_CASES, 1):
        print(f"\n📋 Test {i}/5: {test_case['name']}")
        print(f"📝 Hidden Intent: {test_case['hidden_intent']}")
        print(f"🤖 Simulated Model Response: {test_case['model_response'][:500]}...")
        
        try:
            # Run the evaluation with real OpenAI API
            score, reason = evaluate_usr(
                test_case['hidden_intent'],
                test_case['model_response'], 
                judge_adapter
            )
            
            # Check if score is in expected range
            min_score, max_score = test_case['expected_score_range']
            score_in_range = min_score <= score <= max_score
            
            # Print results
            status = "✅ PASS" if score_in_range else "❌ FAIL"
            print(f"📊 Result: Score {score}/5 - {reason}")
            print(f"🎯 Expected Range: {min_score}-{max_score} | {status}")
            
            if not score_in_range:
                all_passed = False
                print(f"   ⚠️  Score {score} outside expected range {min_score}-{max_score}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All evaluate_usr smoke tests passed! ✅")
    else:
        print("⚠️  Some evaluate_usr smoke tests failed! ❌")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    run_evaluate_usr_smoke_test()
