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


def calculate_average_similarity(hidden_intent_text, search_results, embedder):
    """
    Calculate the average cosine similarity between hidden intent and all search result embeddings.
    
    Args:
        hidden_intent_text: The hidden intent text to embed and compare against
        search_results: List of SearchResult objects with embedding vectors
        embedder: QueryEmbedder instance to create embeddings
        
    Returns:
        float: Average similarity score, or 0.0 if no valid embeddings
        
    Examples:
        >>> from query_embedder import QueryEmbedder
        >>> embedder = QueryEmbedder()
        >>> avg_sim = calculate_average_similarity("science fiction", search_results, embedder)
        >>> print(f"Average similarity: {avg_sim:.3f}")
    """
    if not search_results or not hidden_intent_text:
        return 0.0
    
    # Embed the hidden intent text
    intent_vector = embedder.embed(hidden_intent_text)
    if not intent_vector:
        return 0.0
    
    similarities = []
    for result in search_results:
        if result.embedding is not None:
            # Use cosine_similarity from numpy calculations
            similarity = cosine_similarity(intent_vector, result.embedding)
            similarities.append(similarity)
    
    if not similarities:
        return 0.0
    
    return sum(similarities) / len(similarities)


def cosine_similarity(vec_a, vec_b):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    if vec_a is None or vec_b is None:
        return 0.0
    
    import numpy as np
    a = np.array(vec_a).flatten()
    b = np.array(vec_b).flatten()
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))


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


class MockSearchResult:
    """Mock SearchResult for testing similarity calculations."""
    def __init__(self, title: str, description: str, embedding: list[float] | None = None):
        self.id = title.lower().replace(' ', '_')
        self.score = 0.0  # Not used in similarity calculation
        self.document = description
        self.metadata = {'title': title}
        self.embedding = embedding


# Test data for calculate_average_similarity smoke tests
SIMILARITY_TEST_CASES = [
    {
        "name": "Perfect Match - Science Fiction",
        "hidden_intent": "In a galaxy still reeling from the collapse of the Galactic Empire, chaos reigns across the outer reaches, where lawlessness has become a way of life. Amidst the remnants of a once-mighty regime, a solitary gunfighter navigates the treacherous landscape as a bounty hunter, forging a path through a universe filled with danger and intrigue. Each mission presents not only a test of skill but also a moral quandary, as the line between right and wrong blurs in a world where survival often comes at a steep price. As he traverses desolate planets and bustling spaceports, the bounty hunter encounters a diverse array of characters—some allies, others adversaries—each with their own agendas and secrets. The weight of his past looms large, shaping his choices and challenging his resolve. In this unforgiving environment, the hunter must grapple with themes of identity, loyalty, and the quest for redemption, all while evading the shadows of his own making.The atmosphere is thick with tension, as the hunter's journey unfolds against a backdrop of stunning vistas and gritty underbellies, where every corner turned could lead to fortune or peril. With each encounter, the stakes rise, and the hunter's resolve is tested, forcing him to confront not only the dangers of the galaxy but also the deeper conflicts within himself. In this relentless pursuit of purpose, the hunter discovers that sometimes, the greatest battles are fought not against others, but within one's own soul.",
        "search_items": [
            {"title": "The Mandalorian", "description": "In a galaxy still reeling from the collapse of the Galactic Empire, chaos reigns across the outer reaches, where lawlessness has become a way of life. Amidst the remnants of a once-mighty regime, a solitary gunfighter navigates the treacherous landscape as a bounty hunter, forging a path through a universe filled with danger and intrigue. Each mission presents not only a test of skill but also a moral quandary, as the line between right and wrong blurs in a world where survival often comes at a steep price. As he traverses desolate planets and bustling spaceports, the bounty hunter encounters a diverse array of characters—some allies, others adversaries—each with their own agendas and secrets. The weight of his past looms large, shaping his choices and challenging his resolve. In this unforgiving environment, the hunter must grapple with themes of identity, loyalty, and the quest for redemption, all while evading the shadows of his own making.The atmosphere is thick with tension, as the hunter's journey unfolds against a backdrop of stunning vistas and gritty underbellies, where every corner turned could lead to fortune or peril. With each encounter, the stakes rise, and the hunter's resolve is tested, forcing him to confront not only the dangers of the galaxy but also the deeper conflicts within himself. In this relentless pursuit of purpose, the hunter discovers that sometimes, the greatest battles are fought not against others, but within one's own soul."},
            {"title": "The Mandalorian", "description": "In a galaxy still reeling from the collapse of the Galactic Empire, chaos reigns across the outer reaches, where lawlessness has become a way of life. Amidst the remnants of a once-mighty regime, a solitary gunfighter navigates the treacherous landscape as a bounty hunter, forging a path through a universe filled with danger and intrigue. Each mission presents not only a test of skill but also a moral quandary, as the line between right and wrong blurs in a world where survival often comes at a steep price. As he traverses desolate planets and bustling spaceports, the bounty hunter encounters a diverse array of characters—some allies, others adversaries—each with their own agendas and secrets. The weight of his past looms large, shaping his choices and challenging his resolve. In this unforgiving environment, the hunter must grapple with themes of identity, loyalty, and the quest for redemption, all while evading the shadows of his own making.The atmosphere is thick with tension, as the hunter's journey unfolds against a backdrop of stunning vistas and gritty underbellies, where every corner turned could lead to fortune or peril. With each encounter, the stakes rise, and the hunter's resolve is tested, forcing him to confront not only the dangers of the galaxy but also the deeper conflicts within himself. In this relentless pursuit of purpose, the hunter discovers that sometimes, the greatest battles are fought not against others, but within one's own soul."},
            {"title": "The Mandalorian", "description": "In a galaxy still reeling from the collapse of the Galactic Empire, chaos reigns across the outer reaches, where lawlessness has become a way of life. Amidst the remnants of a once-mighty regime, a solitary gunfighter navigates the treacherous landscape as a bounty hunter, forging a path through a universe filled with danger and intrigue. Each mission presents not only a test of skill but also a moral quandary, as the line between right and wrong blurs in a world where survival often comes at a steep price. As he traverses desolate planets and bustling spaceports, the bounty hunter encounters a diverse array of characters—some allies, others adversaries—each with their own agendas and secrets. The weight of his past looms large, shaping his choices and challenging his resolve. In this unforgiving environment, the hunter must grapple with themes of identity, loyalty, and the quest for redemption, all while evading the shadows of his own making.The atmosphere is thick with tension, as the hunter's journey unfolds against a backdrop of stunning vistas and gritty underbellies, where every corner turned could lead to fortune or peril. With each encounter, the stakes rise, and the hunter's resolve is tested, forcing him to confront not only the dangers of the galaxy but also the deeper conflicts within himself. In this relentless pursuit of purpose, the hunter discovers that sometimes, the greatest battles are fought not against others, but within one's own soul."}
        ],
        "expected_similarity_range": (0.7, 1.0)
    },
    {
        "name": "Good Match - Mystery with Romance",
        "hidden_intent": "Mystery novels with romantic subplots and strong detective work", 
        "search_items": [
            {"title": "Gone Girl", "description": "Psychological thriller about a marriage gone wrong with mystery and twisted romance"},
            {"title": "The Thursday Murder Club", "description": "Cozy mystery about retired people solving cold cases in their community"},
            {"title": "Pride and Prejudice", "description": "Classic romance novel about Elizabeth Bennet and Mr. Darcy in Regency England"}
        ],
        "expected_similarity_range": (0.4, 0.7)
    },
    {
        "name": "Average Match - Horror/Thriller Mix",
        "hidden_intent": "Horror novels with supernatural elements and psychological terror",
        "search_items": [
            {"title": "The Shining", "description": "Horror novel about a haunted hotel driving a caretaker to madness during winter isolation"},
            {"title": "Gone Girl", "description": "Psychological thriller about a marriage gone wrong with mystery and twisted romance"},
            {"title": "The Girl with the Dragon Tattoo", "description": "Crime thriller about a journalist and hacker investigating a wealthy family's dark secrets"}
        ],
        "expected_similarity_range": (0.3, 0.6)
    },
    {
        "name": "Poor Match - Children's vs Adult",
        "hidden_intent": "Light children's fantasy books with magical adventures suitable for bedtime",
        "search_items": [
            {"title": "Harry Potter", "description": "Young wizard's magical adventures at Hogwarts School of Witchcraft and Wizardry"},
            {"title": "American Psycho", "description": "Disturbing psychological horror about a wealthy investment banker's violent double life"},
            {"title": "The Godfather", "description": "Crime saga about an Italian-American mafia family's power struggles and violence"}
        ],
        "expected_similarity_range": (0.1, 0.4)
    },
    {
        "name": "Terrible Match - Cooking vs Technology",
        "hidden_intent": "Cookbooks with healthy vegetarian recipes and meal planning tips",
        "search_items": [
            {"title": "Advanced Calculus", "description": "Mathematical textbook covering differential equations, vector analysis, and complex functions"},
            {"title": "Auto Repair Manual", "description": "Technical guide for maintaining and repairing car engines, transmissions, and electrical systems"},
            {"title": "Medieval Warfare", "description": "Historical analysis of military tactics, siege warfare, and weapon technology in medieval Europe"}
        ],
        "expected_similarity_range": (0.0, 0.2)
    }
]


def run_similarity_smoke_test():
    """
    Run smoke tests for the calculate_average_similarity function using structured test cases.
    
    Tests the function with 5 scenarios ranging from perfect matches to terrible matches,
    creating mock search results with real embeddings to verify similarity calculations.
    Uses ChatOrchestrator to get the embedder (primary or alternative model).
    """
    print("🧪 Running calculate_average_similarity smoke tests...")
    print("=" * 80)
    
    try:
        # Import here to avoid circular imports
        from chat_orchestrator import ChatOrchestrator
        orchestrator = ChatOrchestrator(enable_clarify_gate=False, enable_logging=False)
        print("✅ ChatOrchestrator initialized successfully")
        
        # Ask user which embedding model to use
        print("\nWhich embedding model would you like to use?")
        print("   1. Primary model")
        print("   2. Alternative model")
        choice = input("Enter 1 or 2 [default: 1]: ").strip()
        
        if choice == "2":
            embedder = orchestrator.get_alternative_query_embedder()
            model_name = embedder._ef.model_name
            print(f"✅ Using alternative embedding model: {model_name}")
        else:
            embedder = orchestrator.get_primary_query_embedder()
            model_name = embedder._ef.model_name
            print(f"✅ Using primary embedding model: {model_name}")
            
    except Exception as e:
        print(f"❌ Failed to initialize embedder: {e}")
        return False
    
    all_passed = True
    
    for i, test_case in enumerate(SIMILARITY_TEST_CASES, 1):
        print(f"\n📋 Test {i}/5: {test_case['name']}")
        print(f"📝 Hidden Intent: {test_case['hidden_intent']}")
        print(f"📚 Search Items: {[item['title'] for item in test_case['search_items']]}")
        
        try:
            # Create mock search results with real embeddings
            mock_search_results = []
            for item in test_case['search_items']:
                # Embed the item description
                item_embedding = embedder.embed(item['description'])
                mock_result = MockSearchResult(
                    title=item['title'],
                    description=item['description'], 
                    embedding=item_embedding
                )
                mock_search_results.append(mock_result)
            
            # Calculate average similarity
            avg_similarity = calculate_average_similarity(
                test_case['hidden_intent'],
                mock_search_results,
                embedder
            )
            
            # Check if similarity is in expected range
            min_sim, max_sim = test_case['expected_similarity_range']
            similarity_in_range = min_sim <= avg_similarity <= max_sim
            
            # Print results
            status = "✅ PASS" if similarity_in_range else "❌ FAIL"
            print(f"📊 Result: Average Similarity {avg_similarity:.3f}")
            print(f"🎯 Expected Range: {min_sim:.1f}-{max_sim:.1f} | {status}")
            
            # Show individual similarities for debugging
            intent_vector = embedder.embed(test_case['hidden_intent'])
            individual_sims = []
            for result in mock_search_results:
                sim = cosine_similarity(intent_vector, result.embedding)
                individual_sims.append(sim)
                print(f"   • {result.metadata['title']}: {sim:.3f}")
            
            if not similarity_in_range:
                all_passed = False
                print(f"   ⚠️  Similarity {avg_similarity:.3f} outside expected range {min_sim:.1f}-{max_sim:.1f}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All calculate_average_similarity smoke tests passed! ✅")
    else:
        print("⚠️  Some calculate_average_similarity smoke tests failed! ❌")
    print("=" * 80)
    
    return all_passed


def run_all_smoke_tests():
    """Run all smoke tests in the metrics module."""
    print("🚀 Running all metrics module smoke tests...")
    print("=" * 100)
    
    usr_passed = run_evaluate_usr_smoke_test()
    print()
    similarity_passed = run_similarity_smoke_test()
    
    print("\n" + "=" * 100)
    if usr_passed and similarity_passed:
        print("🎉 ALL METRICS MODULE SMOKE TESTS PASSED! ✅")
    else:
        print("⚠️  Some metrics module smoke tests failed! ❌")
        print(f"   - evaluate_usr: {'✅ PASS' if usr_passed else '❌ FAIL'}")
        print(f"   - calculate_average_similarity: {'✅ PASS' if similarity_passed else '❌ FAIL'}")
    print("=" * 100)
    
    return usr_passed and similarity_passed


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
        print(f"🤖 Model Response: {test_case['model_response'][:100]}...")
        
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
    run_all_smoke_tests()
