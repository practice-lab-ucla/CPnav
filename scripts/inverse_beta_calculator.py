from math import ceil
from scipy.stats import beta


n = 300
a = 0.25
b = 0.01


# P(C < 1 - alpha) < beta
k = ceil((1 - a) * (n + 1))
high_k = n

while (k < n) and (beta.cdf(1-a, k, n+1-k) > b):
    k += 1

q = 1-(k / n)
print(f"Use {q:.3f} th quantile from below")
