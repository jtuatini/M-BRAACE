import numpy as np
import pandas as pd

def get_permutation(weights: df, sigma: float, seed = None):

    np.random.seed(seed)

    weights_permuted = np.random.normal(weights["Mass"], sigma * weights["Mass"])
    datum_permuted = np.random.normal(weights["datum"], sigma * np.abs(weights["datum"]))
    
    permuted_cg = np.sum(weights_permuted * datum_permuted) / np.sum(weights_permuted)
    
    return permuted_cg

def main():

    import matplotlib.pyplot as plt

    df = pd.read_csv("component_weights.csv", nrows=23)
    df = df.dropna(subset=["Mass", "datum"])

    neutral_point = 0.788
    MAC = 0.6276

    n_permutations = 10000
    n_stable = 0

    sm_permuted = []
    for i in range(n_permutations):
        cg_perm = get_permutation(df, 0.05)
        sm_perm = (neutral_point - cg_perm) / MAC
        if sm_perm >= 0.05 and sm_perm <= 0.25:
            n_stable += 1
        sm_permuted.append(sm_perm * 100)
    
    prob_stable = n_stable / n_permutations
    mean_sm = np.mean(sm_permuted)

    plt.figure(figsize=(10, 6))

    plt.hist(sm_permuted, bins=40, color='#1f77b4', edgecolor='black', alpha=0.8)

    plt.axvspan(5, 25, color='green', alpha=0.1, label='Stable Region (0.05 <= SM <= 0.25)')
    plt.axvspan(0, 5, color='red', alpha=0.1)
    plt.axvspan(25, 30, color='red', alpha=0.1)
    plt.xlim(0, 30)

    plt.axvline(5, color='red', ls='--', lw=2, label='Stability Threshold (SM = 0.05)')
    plt.axvline(25, color='red', ls='--', lw=2, label='Control Threshold (SM = 0.25)')
    plt.axvline(mean_sm, color='black', ls='--', lw=2, label=f'Mean SM')

    footer_text = (
        f"Simulations: {n_permutations:,}   |   "
        f"Stable Probability: {prob_stable * 100:.2f}%   |   "
        f"Mean SM: {mean_sm:.2f}%"
    )
    
    plt.figtext(0.5, 0.04, footer_text, ha="center", fontsize=12)
    plt.title("Monte Carlo Simulation", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Static Margin [%]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    plt.legend(loc='upper right', framealpha=1)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    plt.savefig("static_margin_distribution.png")
    plt.close()
    
if __name__ == "__main__":
    main()
