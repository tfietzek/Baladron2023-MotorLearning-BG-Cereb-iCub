import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, entropy, ks_2samp
from scipy.signal import correlate
import numpy.ma as ma


def calculate_distribution_similarities(distributions: dict,
                                        methods: list = ['wasserstein', 'kl_divergence', 'ks_stat',
                                                         'correlation']) -> dict:
    """
    Calculate similarity matrices between distributions using multiple methods.

    Args:
        distributions: Dictionary with angles as keys and distribution arrays as values
        methods: List of similarity measures to calculate

    Returns:
        Dictionary containing similarity matrices for each method
    """
    angles = sorted(distributions.keys())
    n_angles = len(angles)
    results = {}

    def safe_kl_divergence(p, q):
        # Add small constant to avoid division by zero
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        return entropy(p, q)

    def histogram_correlation(p, q):
        # Normalize distributions
        p = p / np.sum(p)
        q = q / np.sum(q)
        # Calculate correlation
        corr = correlate(p, q, mode='full')
        return np.max(corr)

    for method in methods:
        matrix = np.zeros((n_angles, n_angles))

        for i, angle1 in enumerate(angles):
            for j, angle2 in enumerate(angles):
                dist1 = distributions[angle1]
                dist2 = distributions[angle2]

                if method == 'wasserstein':
                    # Earth Mover's Distance
                    matrix[i, j] = wasserstein_distance(
                        np.arange(len(dist1)), np.arange(len(dist2)),
                        dist1, dist2
                    )
                elif method == 'kl_divergence':
                    # Symmetric KL divergence
                    kl1 = safe_kl_divergence(dist1, dist2)
                    kl2 = safe_kl_divergence(dist2, dist1)
                    matrix[i, j] = (kl1 + kl2) / 2
                elif method == 'ks_stat':
                    # Kolmogorov-Smirnov statistic
                    matrix[i, j] = ks_2samp(dist1, dist2).statistic
                elif method == 'correlation':
                    # Correlation-based similarity
                    matrix[i, j] = histogram_correlation(dist1, dist2)

        results[method] = matrix

    return results


def plot_similarity_matrices(similarity_matrices: dict, figsize=(20, 5)):
    """Plot heatmaps of similarity matrices."""
    n_methods = len(similarity_matrices)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    for ax, (method, matrix) in zip(axes, similarity_matrices.items()):
        sns.heatmap(matrix, ax=ax, cmap='viridis')
        ax.set_title(f'{method.replace("_", " ").title()}')
        ax.set_xlabel('Angle Index')
        ax.set_ylabel('Angle Index')

    plt.tight_layout()
    return fig


def analyze_distribution_separability(similarity_matrices: dict) -> pd.DataFrame:
    """
    Analyze how well-separated the distributions are based on similarity matrices.

    Returns DataFrame with statistics about distribution separability.
    """
    stats = []

    for method, matrix in similarity_matrices.items():
        # Mask diagonal elements
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diagonal = matrix[mask]

        stats.append({
            'method': method,
            'mean_similarity': np.mean(off_diagonal),
            'std_similarity': np.std(off_diagonal),
            'min_similarity': np.min(off_diagonal),
            'max_similarity': np.max(off_diagonal),
            'separability_score': np.mean(off_diagonal) / np.std(off_diagonal)
        })

    return pd.DataFrame(stats)


def identify_similar_distributions(similarity_matrices: dict,
                                   threshold_percentile: float = 90) -> list:
    """
    Identify pairs of distributions that are particularly similar.

    Args:
        similarity_matrices: Dictionary of similarity matrices
        threshold_percentile: Percentile threshold for considering distributions similar

    Returns:
        List of tuples containing similar angle pairs
    """
    similar_pairs = []
    angles = list(range(similarity_matrices[list(similarity_matrices.keys())[0]].shape[0]))

    for method, matrix in similarity_matrices.items():
        # Determine threshold based on percentile
        threshold = np.percentile(matrix[~np.eye(matrix.shape[0], dtype=bool)],
                                  threshold_percentile)

        # Find similar pairs
        if method in ['correlation']:  # Methods where higher values mean more similar
            pairs = np.where((matrix >= threshold) & ~np.eye(matrix.shape[0], dtype=bool))
        else:  # Methods where lower values mean more similar
            pairs = np.where((matrix <= threshold) & ~np.eye(matrix.shape[0], dtype=bool))

        for idx1, idx2 in zip(*pairs):
            similar_pairs.append((angles[idx1], angles[idx2], method))

    return similar_pairs


def analyze_angle_index_frequencies(df: pd.DataFrame) -> dict:
    # Get unique angles
    unique_angles = df['theta'].unique()
    unique_angles = np.sort(unique_angles)

    # Dictionary to store index frequencies for each angle
    index_frequencies = {}

    for ang in unique_angles:
        # Get r_output arrays for this angle
        r_arrays = df.loc[df['theta'] == ang, 'r_output'].tolist()
        r_arrays = np.array(r_arrays)

        # Count the number of non-zero elements in each array
        n_r = r_arrays.shape[0]
        norm = np.max(r_arrays, axis=1)
        freq = np.zeros(r_arrays.shape[1])
        for i in range(n_r):
            freq += r_arrays[i]/norm[i]

        index_frequencies[ang] = freq/n_r

    return index_frequencies


# Example usage:
if __name__ == "__main__":
    df = pd.read_parquet("results/RHI_j12_sigma4/network_inverse_kinematic/RHI_j12_sigma4_training.parquet")
    freq_dist = analyze_angle_index_frequencies(df)

    # Plot frequency distributions
    fig = plt.figure(figsize=(10, 6))
    for i, key in enumerate(freq_dist):
        ax = fig.add_subplot(5, 7, i + 1)
        ax.plot(freq_dist[key])
        ax.set_ylim(0, 1)
        ax.set_title(f'Angle: {key}')
        ax.set_xticks(np.arange(0, 50, 1, dtype=int), [None if i % 10 != 0 else str(i) for i in range(50)])
    plt.tight_layout()
    plt.savefig("angle_index_frequencies.pdf", bbox_inches='tight',  pad_inches = 0.25)
    plt.show()


    # Calculate similarities
    similarity_matrices = calculate_distribution_similarities(freq_dist)

    # Plot similarity matrices
    fig = plot_similarity_matrices(similarity_matrices)
    plt.show()

    # Analyze separability
    separability_stats = analyze_distribution_separability(similarity_matrices)
    print("\nDistribution Separability Statistics:")
    print(separability_stats)

    # Identify similar distributions
    similar_pairs = identify_similar_distributions(similarity_matrices)
    print("\nMost Similar Distribution Pairs:")
    for index1, index2, method in similar_pairs:
        print(f"Index {index1} and {index2} are similar according to {method}")