import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_epss_distribution():
    data_path = Path('Data_Files/May/all_features_02_06.csv')

    print(f"Loading data from: {data_path}")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    epss_scores = df['epss_score'].dropna()

    quantiles = epss_scores.quantile([0.25, 0.50, 0.75])
    p25 = quantiles[0.25]
    p50 = quantiles[0.50] 
    p75 = quantiles[0.75]

    print("\n--- EPSS Score Distribution ---")
    print(f"25th Percentile: {p25:.4f}")
    print(f"50th Percentile (Median): {p50:.4f}")
    print(f"75th Percentile: {p75:.4f}")
    print("--------------------------------------------------\n")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(epss_scores, bins=100, alpha=0.7, label='EPSS Score Distribution')
    ax.set_xscale('log') 

    ax.axvline(p25, color='orange', linestyle='--', linewidth=2, label=f'25th Percentile ({p25:.4f})')
    ax.axvline(p50, color='red', linestyle='--', linewidth=2, label=f'50th Percentile ({p50:.4f})')
    ax.axvline(p75, color='purple', linestyle='--', linewidth=2, label=f'75th Percentile ({p75:.4f})')

    ax.set_title('Distribution of EPSS Scores (Log Scale)')
    ax.set_xlabel('EPSS Score (log scale)')
    ax.set_ylabel('Frequency')
    ax.grid(True, which="both", ls="--", c='0.7')
    ax.legend()


    plt.savefig('epss_score_distribution.png', dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyze_epss_distribution()