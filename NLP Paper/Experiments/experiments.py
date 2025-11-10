import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import json
from scipy.stats import binomtest, kendalltau

class RankingResult:
    def __init__(self, query_doi, rank, tot, tie_start, tie_end): 
        if not isinstance(query_doi, str) or query_doi.strip() == "":
            raise ValueError('query_doi must be a non-empty string')
        self.query_doi = query_doi
        self.rank = rank
        self.tot = tot
        self.tie_start = tie_start
        self.tie_end = tie_end

    def __repr__(self):
        return f'RankingResult(query_doi={self.query_doi}, rank={self.rank})'

    @staticmethod
    def from_ans_and_ranked_df(query_doi: str, ans_doi: str, ranked_df: pd.DataFrame):
        """
            query_doi: the query DOI for this ranking result (must start with https://doi.org/)
            ans_doi: ground truth (which is found somewhere in the ranked_df, the lower the rank, the better)
            ranked_df: should contain 'doi' and 'score' columns. Will be sorted by score (descending).
        """
        if query_doi is None:
            raise ValueError('`query_doi` cannot be None')
        if ans_doi is None:
            raise ValueError('`ans_doi` cannot be None')
        
        # Sort by score descending (highest scores first)
        sorted_df = ranked_df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Find the answer DOI
        ans_mask = sorted_df['doi'].str.lower() == ans_doi.lower()
        if not ans_mask.any():
            raise ValueError('`ans_doi` not found in `ranked_df`')
        
        # Get the score of the answer DOI
        ans_score = sorted_df.loc[ans_mask, 'score'].iloc[0]
        
        # Find all rows with the same score (tie group)
        tie_mask = sorted_df['score'] == ans_score
        tie_indices = sorted_df[tie_mask].index
        
        # Calculate mid-rank: average of first and last position in tie group
        first_rank = tie_indices[0] + 1  # 1-based
        last_rank = tie_indices[-1] + 1   # 1-based
        mid_rank = (first_rank + last_rank) // 2
        
        return RankingResult(
            query_doi=query_doi,
            rank=mid_rank,
            tot=len(sorted_df),
            tie_start=first_rank,
            tie_end=last_rank
        )
    
class ExperimentResults:
    def __init__(self, ranking_results): # list[RankingResult]
        if not ranking_results:
            raise ValueError('`ranking_results` cannot be empty')    
        self.ranking_results = ranking_results
        

    def plot_rank_hist(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(self.get_ranks(), bins=20)
        ax.set_title("Ranking Results Histogram")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        plt.show()

    def _calculate_top_n_accuracy(self, top_n):
        """Calculate tie-aware accuracy for top-n ranking."""
        total_score = 0
        for result in self.ranking_results:
            if result.tie_end <= top_n:
                # Entire tie block is within top-n
                total_score += 1
            elif result.tie_start <= top_n < result.tie_end:
                # Tie block partially overlaps with top-n
                overlap = top_n - result.tie_start + 1
                tie_size = result.tie_end - result.tie_start + 1
                total_score += overlap / tie_size
            # If tie_start > top_n, no contribution (score = 0)
        
        return total_score / len(self.ranking_results) if self.ranking_results else 0
    
    def _calculate_top_p_accuracy(self, top_p):
        """Calculate tie-aware accuracy for top-p percentage ranking."""
        if not self.ranking_results:
            return 0

        top_n = int(self.ranking_results[0].tot * top_p)
        return self._calculate_top_n_accuracy(top_n)
        
    def show_accuracy(self):
        for top_n in [5, 10, 50, 100]:
            accuracy = self._calculate_top_n_accuracy(top_n)
            print(f"Top-{top_n} Accuracy: {accuracy:.2%}")

        for top_p in [0.01, 0.05, 0.1]:
            accuracy = self._calculate_top_p_accuracy(top_p)
            print(f"Top-{top_p:.2%} Accuracy: {accuracy:.2%}")

    def get_ranks(self):
        return [result.rank for result in self.ranking_results]
    
    def get_reciprocal_ranks(self):
        return [1 / result.rank for result in self.ranking_results]
    
    def rr_stats(self):
        rrs = self.get_reciprocal_ranks()
        mrr = sum(rrs) / len(rrs) if rrs else 0
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        sdt_rr = pd.Series(rrs).std() if rrs else 0
        print(f"Standard Deviation of Reciprocal Ranks: {sdt_rr:.4f}")

    def all_in_one(self):
        self.plot_rank_hist()
        self.show_accuracy()
        self.rr_stats()

    def save_to_json(self, filepath: str):
        """Save ExperimentResults to a JSON file."""
        data = {
            'ranking_results': []
        }
        
        for result in self.ranking_results:
            result_data = {
                'query_doi': result.query_doi,
                'rank': int(result.rank),
                'tot': int(result.tot),
                'tie_start': int(result.tie_start),
                'tie_end': int(result.tie_end)
            }
            data['ranking_results'].append(result_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str):
        """Load ExperimentResults from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        ranking_results = []
        for result_data in data['ranking_results']:
            result = RankingResult(
                query_doi=result_data['query_doi'],
                rank=result_data['rank'],
                tot=result_data['tot'],
                tie_start=result_data['tie_start'],
                tie_end=result_data['tie_end']
            )
            ranking_results.append(result)
        
        return cls(ranking_results)


class ExperimentComparator:
    def __init__(self, exp1: "ExperimentResults", exp2: "ExperimentResults"):
        if exp1.ranking_results[0].tot != exp2.ranking_results[0].tot:
            raise ValueError('Both ExperimentResults must have the same `tot` value in their RankingResults.')
        
        if len(exp1.ranking_results) != len(exp2.ranking_results):
            raise ValueError('This is a paired result comparator. Both ExperimentResults must have the same number of RankingResults.')
        
        # Ensure both experiments have the same query DOIs in the same order
        exp1_dois = [result.query_doi for result in exp1.ranking_results]
        exp2_dois = [result.query_doi for result in exp2.ranking_results]
        
        if exp1_dois != exp2_dois:
            raise ValueError('Both ExperimentResults must have the same query DOIs in the same order.')
        
        self.exp1 = exp1
        self.exp2 = exp2

    def binaomial_test(self):
        """
        Perform a binomial test to compare which experiment performs better.
        Counts how many times exp1 has better (lower) rank than exp2.
        """
        exp1_better = 0
        total_comparisons = 0
        
        for r1, r2 in zip(self.exp1.ranking_results, self.exp2.ranking_results):
            if r1.rank == r2.rank:
                continue  # Tie, do not count
            if r1.rank < r2.rank:
                exp1_better += 1
            total_comparisons += 1

        if total_comparisons == 0:
            print("Ranks are identical across all comparisons. No differences to test.")
            return None

        # Binomial test: null hypothesis is that both experiments are equally good (p=0.5)
        result = binomtest(exp1_better, total_comparisons, p=0.5)
        
        print(f"Binomial Test Results:")
        print(f"  Exp1 better than Exp2: {exp1_better}/{total_comparisons} ({exp1_better/total_comparisons:.2%})")
        print(f"  p-value: {result.pvalue:.4f}")
        print(f"  95% CI: [{result.proportion_ci().low:.3f}, {result.proportion_ci().high:.3f}]")
        
        if result.pvalue < 0.05:
            if exp1_better > total_comparisons / 2:
                print(f"  Result: Exp1 significantly better than Exp2 (p < 0.05)")
            else:
                print(f"  Result: Exp2 significantly better than Exp1 (p < 0.05)")
        else:
            print(f"  Result: No significant difference between experiments (p >= 0.05)")
        
        return result

    def kendall_tau(self):
        """
        Calculate Kendall's tau correlation coefficient between the ranks of the two experiments.
        Kendall's tau is more appropriate for rank data as it's a non-parametric measure.
        """
        ranks1 = self.exp1.get_ranks()
        ranks2 = self.exp2.get_ranks()
        
        correlation, p_value = kendalltau(ranks1, ranks2)
        
        print(f"Kendall's Tau Correlation Results:")
        
        if np.isnan(correlation):
            print(f"  Correlation coefficient (τ): undefined (all ranks identical)")
            print(f"  p-value: undefined")
            print(f"  Interpretation: No variation in ranks to correlate")
        else:
            print(f"  Correlation coefficient (τ): {correlation:.4f}")
            print(f"  p-value: {p_value:.4f}")
            
            if abs(correlation) < 0.1:
                strength = "negligible"
            elif abs(correlation) < 0.3:
                strength = "weak"
            elif abs(correlation) < 0.5:
                strength = "moderate"
            elif abs(correlation) < 0.7:
                strength = "strong"
            else:
                strength = "very strong"
                
            direction = "positive" if correlation > 0 else "negative"
            print(f"  Interpretation: {strength} {direction} correlation")
            
            if p_value < 0.05:
                print(f"  Result: Correlation is statistically significant (p < 0.05)")
            else:
                print(f"  Result: Correlation is not statistically significant (p >= 0.05)")
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(ranks1, ranks2, alpha=0.6)
        ax.set_xlabel("Experiment 1 Ranks")
        ax.set_ylabel("Experiment 2 Ranks")
        ax.set_title(f"Rank Correlation Scatter Plot\nτ = {correlation:.4f}, p = {p_value:.4f}")
        
        # Add diagonal line for perfect correlation
        min_rank = min(min(ranks1), min(ranks2))
        max_rank = max(max(ranks1), max(ranks2))
        ax.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', alpha=0.5, label='Perfect correlation')
        ax.legend()
        
        # Add trend line
        z = np.polyfit(ranks1, ranks2, 1)
        p = np.poly1d(z)
        ax.plot(ranks1, p(ranks1), "b--", alpha=0.8, label='Trend line')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return correlation, p_value

    def all_in_one(self):
        self.binaomial_test()
        self.kendall_tau()
        
