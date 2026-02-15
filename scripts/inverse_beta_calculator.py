import argparse
from math import ceil
from scipy.stats import beta

def main():
    # 1. Set up command line arguments
    parser = argparse.ArgumentParser(description="Calculate PAC-Conformal Prediction Quantile")
    
    # Define arguments for n, alpha (a), and beta (b)
    parser.add_argument('-n', type=int, required=True, help="Calibration set size (e.g., 300)")
    parser.add_argument('-a', '--alpha', type=float, required=True, help="Target error rate (e.g., 0.1 for 90%% coverage)")
    parser.add_argument('-b', '--beta', type=float, required=True, help="Confidence failure rate (e.g., 0.01 for 99%% confidence)")
    
    args = parser.parse_args()
    
    n = args.n
    a = args.alpha
    b = args.beta

    # 2. Initial Standard Conformal k (Mean coverage >= 1-alpha)
    # k is the rank (number of items to include)
    # Start searching from the standard expectation
    k = ceil((1 - a) * (n + 1))
    
    # 3. PAC Search Loop
    # We want P(Coverage < 1 - alpha) <= beta
    # In scipy, beta.cdf(x, a, b) is P(X <= x)
    # Here X ~ Beta(k, n+1-k). We want P(X <= 1-a) <= b.
    # While the risk (CDF) is still > b, we increase k (include more items).
    while (k < n) and (beta.cdf(1 - a, k, n + 1 - k) > b):
        k += 1
        
    # Check if we hit the ceiling (k=n means include everything)
    if k == n and beta.cdf(1 - a, k, n + 1 - k) > b:
        print(f"WARNING: Even including all {n} items does not satisfy the {1-b:.3%} confidence requirement.")
        print("You need more calibration data.")
    
    # 4. Convert rank k to quantile q
    # If we include the top k items, we are looking for the (n-k)-th item from the bottom.
    # q is the fraction of items to EXCLUDE (the bottom portion).
    # q = (n - k) / n  is equivalent to  1 - (k / n)
    q = 1 - (k / n)
    
    # 5. Output
    print("-" * 30)
    print(f"Parameters: n={n}, alpha={a}, beta={b}")
    print(f"Standard k (mean coverage): {ceil((1 - a) * (n + 1))}")
    print(f"PAC k (high confidence):    {k}")
    print("-" * 30)
    print(f"RESULT: Use q = {q:.4f}")
    print(f"Logic:  np.quantile(scores, {q:.4f})")
    print("-" * 30)

if __name__ == "__main__":
    main()