import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.
    
    Args:
        group1 (array-like): First group of observations
        group2 (array-like): Second group of observations
    
    Returns:
        float: Cohen's d value
    """
    group1, group2 = np.asarray(group1), np.asarray(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

def overlap_coefficient(group1, group2, bins=100):
    """
    Calculate the overlap coefficient between two distributions using histograms.
    
    Args:
        group1 (array-like): First group of observations
        group2 (array-like): Second group of observations
        bins (int): Number of bins for histogram
    
    Returns:
        float: Overlap coefficient (0 to 1)
    """
    group1, group2 = np.asarray(group1), np.asarray(group2)
    min_val, max_val = min(group1.min(), group2.min()), max(group1.max(), group2.max())
    
    if min_val == max_val:
        return 1.0
    
    hist1, edges = np.histogram(group1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(group2, bins=bins, range=(min_val, max_val), density=True)
    bin_width = edges[1] - edges[0]
    overlap = np.minimum(hist1, hist2).sum() * bin_width
    return min(overlap, 1.0)

def kl_divergence(p, q, bins=100):
    """
    Calculate Kullback-Leibler divergence from p to q using histograms.
    
    Args:
        p (array-like): First distribution
        q (array-like): Second distribution
        bins (int): Number of bins for histogram
    
    Returns:
        float: KL divergence value
    """
    p, q = np.asarray(p), np.asarray(q)
    min_val, max_val = min(p.min(), q.min()), max(p.max(), q.max())
    
    if min_val == max_val:
        return 0.0
    
    hist_p, edges = np.histogram(p, bins=bins, range=(min_val, max_val), density=True)
    hist_q, _ = np.histogram(q, bins=bins, range=(min_val, max_val), density=True)
    
    # Add small constant to avoid log(0)
    hist_p = hist_p + 1e-10
    hist_q = hist_q + 1e-10
    
    return stats.entropy(hist_p, hist_q)

def js_divergence(p, q, bins=100):
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    Args:
        p (array-like): First distribution
        q (array-like): Second distribution
        bins (int): Number of bins for histogram
    
    Returns:
        float: JS divergence value
    """
    p, q = np.asarray(p), np.asarray(q)
    min_val, max_val = min(p.min(), q.min()), max(p.max(), q.max())
    
    if min_val == max_val:
        return 0.0
    
    hist_p, edges = np.histogram(p, bins=bins, range=(min_val, max_val), density=True)
    hist_q, _ = np.histogram(q, bins=bins, range=(min_val, max_val), density=True)
    
    # Mixture distribution
    m = 0.5 * (hist_p + hist_q)
    
    # Add small constant to avoid log(0)
    hist_p = hist_p + 1e-10
    hist_q = hist_q + 1e-10
    m = m + 1e-10
    
    return 0.5 * (stats.entropy(hist_p, m) + stats.entropy(hist_q, m))

def mutual_information(scores, labels):
    """
    Calculate mutual information between continuous scores and discrete labels.
    
    Args:
        scores (array-like): Continuous scores
        labels (array-like): Discrete labels
    
    Returns:
        float: Mutual information value
    """
    scores = np.asarray(scores).reshape(-1, 1)
    labels = np.asarray(labels)
    return mutual_info_classif(scores, labels, discrete_features=False)[0]

def statistical_test(group1, group2, alpha=0.05):
    """
    Perform a statistical test (t-test or Mann-Whitney U) based on normality.
    
    Args:
        group1 (array-like): First group of observations
        group2 (array-like): Second group of observations
        alpha (float): Significance level for normality test
    
    Returns:
        tuple: (test_statistic, p_value, test_name)
    """
    group1, group2 = np.asarray(group1), np.asarray(group2)
    n1, n2 = len(group1), len(group2)
    
    # Check normality with normaltest (requires at least 8 samples)
    normal1 = stats.normaltest(group1)[1] > alpha if n1 >= 8 else False
    normal2 = stats.normaltest(group2)[1] > alpha if n2 >= 8 else False
    
    if normal1 and normal2:
        # Use Welch's t-test (unequal variances)
        stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
        test_name = "ttest"
    else:
        # Use Mann-Whitney U test
        stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "mann_whitney_u"
    
    return stat, pval, test_name

def pairwise_accuracy(simple_scores, expert_scores, pair_ids=None):
    """Calculate the percentage of pairs where expert texts are correctly scored higher than simple texts.
    
    Parameters:
    -----------
    simple_scores : array-like
        Model scores for simple texts
    expert_scores : array-like
        Model scores for expert texts
    pair_ids : array-like, optional
        IDs that identify which simple and expert texts are paired.
        If None, assumes the arrays are already paired by index.
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'pairwise_accuracy': Percentage of correctly ordered pairs (0-100)
        - 'correct_pairs': Number of correctly ordered pairs
        - 'total_pairs': Total number of pairs
        - 'tied_pairs': Number of pairs with equal scores
    """
    import numpy as np
    
    # Convert inputs to numpy arrays
    simple_scores = np.asarray(simple_scores)
    expert_scores = np.asarray(expert_scores)
    
    # Handle index-paired case
    if pair_ids is None:
        if len(simple_scores) != len(expert_scores):
            raise ValueError("simple_scores and expert_scores must have the same length when pair_ids is None")
            
        correct_count = np.sum(expert_scores > simple_scores)
        tied_count = np.sum(expert_scores == simple_scores)
        total_pairs = len(simple_scores)
        
        return {
            'pairwise_accuracy': (correct_count / total_pairs) * 100 if total_pairs > 0 else 0,
            'correct_pairs': int(correct_count),
            'total_pairs': total_pairs,
            'tied_pairs': int(tied_count)
        }
    
    # Handle pair_ids case
    pair_ids = np.asarray(pair_ids)
    unique_pairs = np.unique(pair_ids)
    
    correct_count = 0
    tied_count = 0
    valid_pairs = 0
    
    for pair_id in unique_pairs:
        simple_indices = np.where((pair_ids == pair_id) & (np.isfinite(simple_scores)))[0]
        expert_indices = np.where((pair_ids == pair_id) & (np.isfinite(expert_scores)))[0]
        
        # Skip pairs with missing data
        if len(simple_indices) == 0 or len(expert_indices) == 0:
            continue
            
        # Use the first occurrence if multiple exist
        simple_score = simple_scores[simple_indices[0]]
        expert_score = expert_scores[expert_indices[0]]
        
        valid_pairs += 1
        
        if expert_score > simple_score:
            correct_count += 1
        elif expert_score == simple_score:
            tied_count += 1
    
    # Return results with proper handling of zero division
    if valid_pairs == 0:
        return {
            'pairwise_accuracy': 0,
            'correct_pairs': 0,
            'total_pairs': 0,
            'tied_pairs': 0
        }
    
    return {
        'pairwise_accuracy': (correct_count / valid_pairs) * 100,
        'correct_pairs': correct_count,
        'total_pairs': valid_pairs,
        'tied_pairs': tied_count
    }
    
# Example usage
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 1000)
    group2 = np.random.normal(1, 1.5, 1000)
    labels = np.random.randint(0, 2, 1000)
    
    print(f"Cohen's d: {cohens_d(group1, group2):.4f}")
    print(f"Overlap Coefficient: {overlap_coefficient(group1, group2):.4f}")
    print(f"KL Divergence: {kl_divergence(group1, group2):.4f}")
    print(f"JS Divergence: {js_divergence(group1, group2):.4f}")
    print(f"Mutual Information: {mutual_information(group1, labels):.4f}")
    stat, pval, test = statistical_test(group1, group2)
    print(f"{test}: stat={stat:.4f}, p-value={pval:.4e}")