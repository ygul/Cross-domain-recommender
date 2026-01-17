"""
Report Generator module for Cross-domain Recommender evaluation results.

This module generates visualizations and summary statistics from the Judge_module.py 
evaluation results, including boxplots, bar charts, and console summaries.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class EvaluationReportGenerator:
    """Generates comprehensive reports from evaluation results."""
    
    def __init__(self, excel_file_path: str = None):
        """
        Initialize the report generator.
        
        Args:
            excel_file_path: Path to the evaluation results Excel file.
                           If None, uses default path from Judge_module.py output.
        """
        if excel_file_path is None:
            base_dir = Path(__file__).resolve().parent
            excel_file_path = base_dir.parent / "data" / "output" / "rag_evaluation_final.xlsx"

        self.excel_file_path = Path(excel_file_path)
        self.charts_dir = self.excel_file_path.parent / "charts"
        self.df = None
        
        # Create charts directory if it doesn't exist
        self.charts_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib and seaborn styling
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load evaluation results from Excel file."""
        if not self.excel_file_path.exists():
            raise FileNotFoundError(f"Evaluation results not found: {self.excel_file_path}")
        
        self.df = pd.read_excel(self.excel_file_path)
        print(f"✅ Loaded {len(self.df)} evaluation scenarios from {self.excel_file_path.name}")
        
    def generate_score_comparison_boxplots(self):
        """Generate boxplots comparing Single vs Multi turn scores."""
        if self.df is None:
            self.load_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Score comparison boxplot
        score_data = pd.DataFrame({
            'Single Turn': self.df['Score (Single)'],
            'Multi Turn': self.df['Score (Multi)']
        })
        
        sns.boxplot(data=score_data, ax=ax1, showmeans=True, 
                   meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 8})
        ax1.set_title('Judge Scores: Single vs Multi Turn', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score (1-5)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Intent Similarity comparison boxplot  
        intent_data = pd.DataFrame({
            'Single Turn': self.df['Intent Sim (Single)'],
            'Multi Turn': self.df['Intent Sim (Multi)']
        })
        
        sns.boxplot(data=intent_data, ax=ax2, showmeans=True,
                   meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 8})
        ax2.set_title('Intent Similarity: Single vs Multi Turn', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cosine Similarity (0-1)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.charts_dir / "score_comparison_boxplots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Generated: {output_path.name}")
        
        plt.show()
        
    def generate_scenario_comparison_charts(self):
        """Generate grouped bar charts comparing metrics per scenario."""
        if self.df is None:
            self.load_data()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Prepare data for grouped bar chart
        scenarios = self.df['Scenario ID']
        x_pos = np.arange(len(scenarios))
        width = 0.35
        
        # Judge Scores comparison
        single_scores = self.df['Score (Single)']
        multi_scores = self.df['Score (Multi)']
        
        bars1 = ax1.bar(x_pos - width/2, single_scores, width, label='Single Turn', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, multi_scores, width, label='Multi Turn', alpha=0.8)
        
        ax1.set_xlabel('Scenario ID', fontsize=12)
        ax1.set_ylabel('Judge Score (1-5)', fontsize=12)
        ax1.set_title('Judge Scores per Scenario: Single vs Multi Turn', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Intent Similarity comparison
        single_intent = self.df['Intent Sim (Single)']
        multi_intent = self.df['Intent Sim (Multi)']
        
        bars3 = ax2.bar(x_pos - width/2, single_intent, width, label='Single Turn', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, multi_intent, width, label='Multi Turn', alpha=0.8)
        
        ax2.set_xlabel('Scenario ID', fontsize=12)
        ax2.set_ylabel('Intent Similarity (0-1)', fontsize=12)
        ax2.set_title('Intent Similarity per Scenario: Single vs Multi Turn', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.charts_dir / "scenario_comparison_charts.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Generated: {output_path.name}")
        
        plt.show()
        
    def generate_rri_improvement_chart(self):
        """Generate RRI improvement bar chart with color coding."""
        if self.df is None:
            self.load_data()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenarios = self.df['Scenario ID']
        rri_values = self.df['RRI (Judge)']
        
        # Color bars based on positive/negative values
        colors = ['green' if val >= 0 else 'red' for val in rri_values]
        
        bars = ax.bar(scenarios, rri_values, color=colors, alpha=0.7)
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel('Scenario ID', fontsize=12)
        ax.set_ylabel('RRI (Judge Score Improvement)', fontsize=12)
        ax.set_title('Relative Recommendation Improvement (RRI) per Scenario', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, rri_values):
            height = bar.get_height()
            label_y = height + 0.05 if height >= 0 else height - 0.1
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=10, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Improvement'),
                          Patch(facecolor='red', alpha=0.7, label='Degradation')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = self.charts_dir / "rri_improvement_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Generated: {output_path.name}")
        
        plt.show()
        
    def generate_rri_similarity_chart(self):
        """Generate RRI improvement bar chart for Intent Similarity with color coding."""
        if self.df is None:
            self.load_data()
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenarios = self.df['Scenario ID']
        rri_values = self.df['RRI (Intent Sim)']
        
        # Color bars based on positive/negative values
        colors = ['green' if val >= 0 else 'red' for val in rri_values]
        
        bars = ax.bar(scenarios, rri_values, color=colors, alpha=0.7)
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel('Scenario ID', fontsize=12)
        ax.set_ylabel('RRI (Intent Similarity Improvement)', fontsize=12)
        ax.set_title('Relative Recommendation Improvement (RRI) - Intent Similarity per Scenario', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, rri_values):
            height = bar.get_height()
            label_y = height + 0.01 if height >= 0 else height - 0.02
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=10, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Improvement'),
                          Patch(facecolor='red', alpha=0.7, label='Degradation')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = self.charts_dir / "rri_similarity_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Generated: {output_path.name}")
        
        plt.show()
        
    def print_summary_statistics(self):
        """Print comprehensive summary statistics to console."""
        if self.df is None:
            self.load_data()
            
        print("\n" + "="*80)
        print("📋 EVALUATION SUMMARY STATISTICS")
        print("="*80)
        
        # Judge Score Statistics
        single_mean = self.df['Score (Single)'].mean()
        multi_mean = self.df['Score (Multi)'].mean()
        rri_judge_mean = self.df['RRI (Judge)'].mean()
        
        print(f"\n🏆 JUDGE SCORES:")
        print(f"   • Single Turn Average:  {single_mean:.2f}/5")
        print(f"   • Multi Turn Average:   {multi_mean:.2f}/5")
        print(f"   • Average Improvement:  {rri_judge_mean:+.2f} points")
        
        improvement_pct = ((multi_mean - single_mean) / single_mean) * 100
        print(f"   • Improvement Percentage: {improvement_pct:+.1f}%")
        
        # Intent Similarity Statistics
        single_intent_mean = self.df['Intent Sim (Single)'].mean()
        multi_intent_mean = self.df['Intent Sim (Multi)'].mean()
        rri_intent_mean = self.df['RRI (Intent Sim)'].mean()
        
        print(f"\n🎯 INTENT SIMILARITY:")
        print(f"   • Single Turn Average:  {single_intent_mean:.3f}")
        print(f"   • Multi Turn Average:   {multi_intent_mean:.3f}")
        print(f"   • Average Improvement:  {rri_intent_mean:+.3f}")
        
        intent_improvement_pct = ((multi_intent_mean - single_intent_mean) / single_intent_mean) * 100
        print(f"   • Improvement Percentage: {intent_improvement_pct:+.1f}%")
        
        # Scenario Performance
        print(f"\n📊 SCENARIO BREAKDOWN:")
        improved_scenarios = len(self.df[self.df['RRI (Judge)'] > 0])
        degraded_scenarios = len(self.df[self.df['RRI (Judge)'] < 0])
        unchanged_scenarios = len(self.df[self.df['RRI (Judge)'] == 0])
        
        print(f"   • Scenarios with Improvement: {improved_scenarios}/{len(self.df)} ({improved_scenarios/len(self.df)*100:.1f}%)")
        print(f"   • Scenarios with Degradation: {degraded_scenarios}/{len(self.df)} ({degraded_scenarios/len(self.df)*100:.1f}%)")
        print(f"   • Scenarios Unchanged:        {unchanged_scenarios}/{len(self.df)} ({unchanged_scenarios/len(self.df)*100:.1f}%)")
        
        # Turns Usage
        avg_turns = self.df['Turns Used'].mean()
        print(f"\n💬 CONVERSATION DYNAMICS:")
        print(f"   • Average Turns Used: {avg_turns:.1f}")
        print(f"   • Max Turns Used:     {self.df['Turns Used'].max()}")
        print(f"   • Min Turns Used:     {self.df['Turns Used'].min()}")
        
        print("\n" + "="*80)
        
    def generate_all_reports(self):
        """Generate all reports and visualizations."""
        print("\n🚀 Generating comprehensive evaluation reports...")
        print("="*60)
        
        try:
            self.load_data()
            
            # Generate all visualizations
            self.generate_score_comparison_boxplots()
            self.generate_scenario_comparison_charts() 
            self.generate_rri_improvement_chart()
            self.generate_rri_similarity_chart()
            
            # Print summary statistics
            self.print_summary_statistics()
            
            print(f"\n✅ All reports generated successfully!")
            print(f"📁 Charts saved to: {self.charts_dir}")
            print(f"📊 Data saved to: {self.excel_file_path.parent}")
            
        except Exception as e:
            print(f"❌ Error generating reports: {e}")
            raise


def main():
    """Main function to run report generation."""
    generator = EvaluationReportGenerator()
    generator.generate_all_reports()


if __name__ == "__main__":
    main()