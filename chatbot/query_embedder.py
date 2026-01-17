from chromadb.utils import embedding_functions
import llm_adapter as llm_module
from llm_adapter import LLMAdapter

SYSTEM_PROMPT = """ You rewrite user search queries to be optimal for semantic vector search. 
                    Preserve the original intent.
                    Do not add new concepts, opinions, or recommendations.
                    Return only the rewritten query text in English; no introduction text etc."""

SYSTEM_PROMPT = \
"""You rewrite user search queries to be optimal for semantic vector search over plot synopses.
Preserve the user's intent and emotional tone exactly and do not introduce new entities, settings, themes, or genres.
If the user expresses tone constraints (e.g. dark, serious, not light), reflect them explicitly.
Convert the query into a compact "synopsis-like" description using the user's own concepts only.
Use English, third person, present tense.
Return only the rewritten text."""

class QueryEmbedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", llm_adapter: LLMAdapter | None = None) -> None:
        """
        Initialize the QueryEmbedder.

        Args:
            model_name (str, optional): Name of the sentence transformer model to use for embeddings.
                Defaults to "paraphrase-multilingual-MiniLM-L12-v2".
            llm_adapter (llm_adapter.LLMAdapter | None, optional): An optional LLM adapter instance.
                If not provided, a default adapter will be created (OpenAI).

        Returns:
            None
        """
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        # Use provided adapter or create a default one if possible
        if llm_adapter is not None:
            self._llm_adapter = llm_adapter
        else:
            try:
                self._llm_adapter = llm_module.create_llm_adapter(provider="openai")
                print("Default LLM adapter (OpenAI) created for QueryEmbedder.")
            except Exception as e:
                print(f"Warning: default LLM adapter could not be created: {e}")
                self._llm_adapter = None

    def embed(self, query: str) -> list[float]:
        """Embed a single query string into a vector. Use the LLM adapter if provided during initialization to improve the query first."""
        if self._llm_adapter:
            query = self.improve_query_with_llm(query)
        
        return self._ef([query])[0].tolist()

    def improve_query_with_llm(self, query):
        system_prompt = SYSTEM_PROMPT
        try:
            if not self._llm_adapter:
                raise ValueError("LLM adapter is not provided.")
            improved_query = self._llm_adapter.generate(system_prompt, query)
            query = improved_query
        except Exception as e:
            print(f"LLM adapter failed to improve query: {e}")
        return query

# Smoke test
if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import configparser
    from pathlib import Path
    
    # Load config
    BASE_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = BASE_DIR / "config.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    
    # Test cases: (user_query, item_description)
    test_cases = [
        (
            "I want to see something that contains pirates and treasure",
            "A swashbuckling adventure on the high seas where a band of pirates search for legendary treasure, navigating treacherous waters and dangerous rivals. The story combines action, mystery, and the allure of riches beyond imagination."
        ),
        (
            "Dark psychological thriller with murder mystery",
            "In a small town, a detective must unravel a complex murder case that reveals disturbing psychological depths. Each clue leads deeper into the twisted minds of the inhabitants, where nothing is as it seems."
        ),
        (
            "Fantasy adventure with magic systems and epic quests",
            "A young hero discovers an ancient magic system and embarks on an epic quest to save the kingdom from dark forces. Along the way, they master powerful spells and forge alliances with mythical creatures in a world of endless wonder."
        ),
        (
            "Contemporary romance in a small town",
            "Two neighbors reconnect after years apart in their quiet hometown, discovering that their past feelings never truly faded. As they navigate family expectations and personal dreams, love blooms in the most unexpected places."
        ),
        (
            "Bounty hunter in the Star Wars universe",
            "In a galaxy still reeling from the collapse of the Galactic Empire, chaos reigns across the outer reaches, where lawlessness has become a way of life. Amidst the remnants of a once-mighty regime, a solitary gunfighter navigates the treacherous landscape as a bounty hunter, forging a path through a universe filled with danger and intrigue."
        ),
        (
            "Historical fiction about World War II",
            "During the darkest days of World War II, a group of resistance fighters risk everything to sabotage Nazi operations. Based on true events, this gripping narrative showcases the courage and sacrifice of ordinary people fighting against tyranny."
        ),
        (
            "Cozy mystery in a bookstore",
            "A bookstore owner with a knack for solving mysteries finds herself caught up in a murder case when a regular customer is found dead. With help from quirky friends and her trusty cat, she must uncover the truth hidden among the pages."
        ),
        (
            "Coming of age story about teenagers",
            "A group of high school friends navigate the complexities of growing up, dealing with first loves, family struggles, and the pressure to figure out their futures. Together, they discover that friendship is the greatest adventure of all."
        ),
        (
            "Dystopian future with rebellion",
            "In a totalitarian future where free thought is forbidden, a young rebel discovers forbidden knowledge and ignites a movement for freedom. As the rebellion grows, they must confront powerful forces determined to maintain control."
        ),
        (
            "Comedy about workplace relationships",
            "In a chaotic advertising agency, a quirky group of coworkers navigate romantic entanglements, career ambitions, and absurd client demands. Hilarity ensues as they discover that sometimes the best moments happen when nothing goes according to plan."
        )
    ]

    print("Testing query-item embedding matching stability with both models...")
    print("=" * 100)

    # Initialize embedders for both models
    llm_adapter_instance = llm_module.create_llm_adapter(provider="openai")
    
    model_primary = config["AI"]["model_embedding"]
    model_alternative = config["AI"]["model_embedding_alternative"]
    
    print(f"🔤 Primary model: {model_primary}")
    print(f"🔤 Alternative model: {model_alternative}")
    print("=" * 100)
    
    embedder_primary = QueryEmbedder(model_name=model_primary, llm_adapter=llm_adapter_instance)
    embedder_alternative = QueryEmbedder(model_name=model_alternative, llm_adapter=llm_adapter_instance)
    
    # Store results for both models
    similarities_primary = []
    similarities_alternative = []
    
    def test_query_item_matching(embedder, model_name, test_cases):
        """
        Test query-item matching quality.
        Queries are improved via LLM, items are embedded directly (no LLM).
        Measures the similarity score for each query-item pair.
        """
        print(f"\n📊 Testing {model_name}")
        print("-" * 95)
        
        similarities = []
        
        for i, (user_query, item_description) in enumerate(test_cases, 1):
            print(f"Case {i}: {user_query[:60]}{'...' if len(user_query) > 60 else ''}")
            
            # Item embedding is constant (no LLM improvement)
            item_embedding = embedder._ef([item_description])[0].tolist()
            
            # Improve query via LLM and embed
            improved_query = embedder.improve_query_with_llm(user_query)
            query_embedding = embedder._ef([improved_query])[0].tolist()
            
            # Calculate query-item similarity
            similarity = cosine_similarity([query_embedding], [item_embedding])[0][0]
            similarities.append(similarity)
            
            print(f"  Improved query: {improved_query}")
            print(f"  Query-Item similarity: {similarity:.4f}")
            
        return similarities
    
    # Test both models
    similarities_primary = test_query_item_matching(embedder_primary, "Primary", test_cases)
    similarities_alternative = test_query_item_matching(embedder_alternative, "Alternative", test_cases)
    
    # Analysis and comparison
    print("\n" + "=" * 100)
    print("QUERY-ITEM MATCHING QUALITY ANALYSIS:")
    print("=" * 100)
    
    def get_statistics(similarities, model_name):
        """Calculate and return statistics for a model"""
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        if mean_sim > 0.85:
            quality_rating = "Excellent"
        elif mean_sim > 0.75:
            quality_rating = "Very Good"
        elif mean_sim > 0.65:
            quality_rating = "Good"
        elif mean_sim > 0.55:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
            
        return {
            'name': model_name,
            'mean': mean_sim,
            'std': std_sim,
            'min': min_sim,
            'max': max_sim,
            'rating': quality_rating
        }
    
    stats_primary = get_statistics(similarities_primary, "Primary")
    stats_alternative = get_statistics(similarities_alternative, "Alternative")
    
    # Print comparison table
    print(f"\n{'Model':<15} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Rating':<15}")
    print("-" * 75)
    print(f"{stats_primary['name']:<15} {stats_primary['mean']:<8.4f} {stats_primary['std']:<8.4f} {stats_primary['min']:<8.4f} {stats_primary['max']:<8.4f} {stats_primary['rating']:<15}")
    print(f"{stats_alternative['name']:<15} {stats_alternative['mean']:<8.4f} {stats_alternative['std']:<8.4f} {stats_alternative['min']:<8.4f} {stats_alternative['max']:<8.4f} {stats_alternative['rating']:<15}")
    
    # Determine which model has better average matching
    if stats_primary['mean'] > stats_alternative['mean']:
        better_model = stats_primary['name']
        difference = stats_primary['mean'] - stats_alternative['mean']
    else:
        better_model = stats_alternative['name']
        difference = stats_alternative['mean'] - stats_primary['mean']
    
    print(f"\n🏆 Best matching model: {better_model} (difference: {difference:.4f})")
    
    if stats_primary['std'] < stats_alternative['std']:
        more_consistent = stats_primary['name']
        std_difference = stats_alternative['std'] - stats_primary['std']
    else:
        more_consistent = stats_alternative['name']
        std_difference = stats_primary['std'] - stats_alternative['std']
    
    print(f"📊 Most consistent model: {more_consistent} (std difference: {std_difference:.4f})")
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Boxplot comparison
    plt.subplot(2, 2, 1)
    plt.boxplot([similarities_primary, similarities_alternative], 
                tick_labels=[f"Primary\n({model_primary})", f"Alternative\n({model_alternative})"])
    plt.title("Query-Item Matching Quality - Boxplot", fontsize=12, fontweight='bold')
    plt.ylabel("Similarity Score")
    plt.grid(True, alpha=0.3)
    
    # Histogram comparison
    plt.subplot(2, 2, 2)
    plt.hist(similarities_primary, alpha=0.7, label=f"Primary", bins=8, color='blue')
    plt.hist(similarities_alternative, alpha=0.7, label=f"Alternative", bins=8, color='orange')
    plt.title("Distribution of Matching Scores", fontsize=12, fontweight='bold')
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Line plot showing per-case comparison
    plt.subplot(2, 2, 3)
    x_pos = range(1, len(test_cases) + 1)
    plt.plot(x_pos, similarities_primary, 'o-', label=f"Primary", linewidth=2, markersize=8, color='blue')
    plt.plot(x_pos, similarities_alternative, 's-', label=f"Alternative", linewidth=2, markersize=8, color='orange')
    plt.title("Per-Query Matching Score Comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Test Case Number")
    plt.ylabel("Similarity Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics bar chart
    plt.subplot(2, 2, 4)
    metrics = ['Mean', 'Std', 'Min', 'Max']
    primary_values = [stats_primary['mean'], stats_primary['std'], stats_primary['min'], stats_primary['max']]
    alternative_values = [stats_alternative['mean'], stats_alternative['std'], stats_alternative['min'], stats_alternative['max']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, primary_values, width, label=f"Primary", alpha=0.8, color='blue')
    plt.bar(x + width/2, alternative_values, width, label=f"Alternative", alpha=0.8, color='orange')
    
    plt.title("Statistical Comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Visualization saved as 'embedding_stability_comparison.png'")
    print("=" * 100)