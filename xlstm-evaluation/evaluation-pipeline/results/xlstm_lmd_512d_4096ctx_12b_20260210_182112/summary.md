# Perplexity Evaluation Summary

**Model**: xlstm_lmd_512d_4096ctx_12b
**Date**: 2026-02-10 18:21

## Model Configuration
- Embedding Dim: 512
- Num Blocks: 12
- Training Context: 4096
- Vocab Size: 675

## Best Checkpoint
- **Checkpoint**: checkpoint-66000
- **PPL at Training Context**: 1.6427

## Checkpoint Selection Results
| checkpoint             |   step |   context_length |   perplexity |
|:-----------------------|-------:|-----------------:|-------------:|
| checkpoint-2000        |   2000 |             4096 |      4.16436 |
| checkpoint-34000       |  34000 |             4096 |      1.74122 |
| checkpoint-66000       |  66000 |             4096 |      1.64274 |
| checkpoint-98000       |  98000 |             4096 |      1.64723 |
| checkpoint-130000      | 130000 |             4096 |      1.74423 |
| checkpoint-158760-last | 158760 |             4096 |      1.82436 |

## Test Results (All Context Lengths)
| checkpoint       |   step |   context_length |   perplexity |
|:-----------------|-------:|-----------------:|-------------:|
| checkpoint-66000 |  66000 |             1024 |      1.86828 |
| checkpoint-66000 |  66000 |             2048 |      1.71541 |
| checkpoint-66000 |  66000 |             3072 |      1.65771 |
| checkpoint-66000 |  66000 |             4096 |      1.62978 |
| checkpoint-66000 |  66000 |             5120 |      1.62202 |
| checkpoint-66000 |  66000 |            10240 |      1.9757  |
