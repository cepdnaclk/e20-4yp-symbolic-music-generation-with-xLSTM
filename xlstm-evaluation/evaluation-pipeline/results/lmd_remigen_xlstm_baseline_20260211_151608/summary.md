# Perplexity Evaluation Summary

**Model**: lmd_remigen_xlstm_baseline
**Date**: 2026-02-11 15:16

## Model Configuration
- Embedding Dim: 256
- Num Blocks: 12
- Training Context: 2048
- Vocab Size: 675

## Best Checkpoint
- **Checkpoint**: checkpoint-13000
- **PPL at Training Context**: 1.8779

## Checkpoint Selection Results
| checkpoint            |   step |   context_length |   perplexity |
|:----------------------|-------:|-----------------:|-------------:|
| checkpoint-1000       |   1000 |             2048 |      3.19317 |
| checkpoint-4000       |   4000 |             2048 |      2.10973 |
| checkpoint-7000       |   7000 |             2048 |      1.95518 |
| checkpoint-10000      |  10000 |             2048 |      1.89487 |
| checkpoint-13000      |  13000 |             2048 |      1.87789 |
| checkpoint-14895-last |  14895 |             2048 |      1.87865 |

## Test Results (All Context Lengths)
| checkpoint       |   step |   context_length |   perplexity |
|:-----------------|-------:|-----------------:|-------------:|
| checkpoint-13000 |  13000 |             1024 |      1.99315 |
| checkpoint-13000 |  13000 |             2048 |      1.86784 |
| checkpoint-13000 |  13000 |             3072 |      3.36191 |
| checkpoint-13000 |  13000 |             4096 |     19.4676  |
| checkpoint-13000 |  13000 |             5120 |    113.648   |
| checkpoint-13000 |  13000 |            10240 |  20252.9     |
