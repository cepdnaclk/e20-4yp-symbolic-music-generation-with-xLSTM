import os
from collections import Counter

TOKEN_DIR = "notebooks/xLSTM-3/results/xlstm_chunked_eval/tokens/"
files = sorted([f for f in os.listdir(TOKEN_DIR) if f.endswith(".txt")])

print(f"{'Filename':<30} | {'Total Tokens':<12} | {'Unique Tokens':<15} | {'Repetition Rate (%)':<20} | {'Header Freq (s-9 t-38)':<25}")
print("-" * 110)

for f in files:
    with open(os.path.join(TOKEN_DIR, f), 'r') as file:
        content = file.read()
        tokens = content.split()
        total = len(tokens)
        if total == 0: continue
        
        unique = len(set(tokens))
        
        # Check for the repetition of the specific sequence "s-9 o-0 t-38"
        headers = content.count("s-9 o-0 t-38")
        
        # Simple repetition measure: compression ratio or 4-gram repetition
        n = 8
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if len(ngrams) > 0:
            ngram_counts = Counter(ngrams)
            most_common_n, count = ngram_counts.most_common(1)[0]
            rep_rate = (count * n / total) * 100
        else:
            rep_rate = 0
            
        print(f"{f:<30} | {total:<12} | {unique:<15} | {rep_rate:<20.2f} | {headers:<25}")
