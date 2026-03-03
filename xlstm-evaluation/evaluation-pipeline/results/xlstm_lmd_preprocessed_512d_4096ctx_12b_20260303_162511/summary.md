# Perplexity Evaluation Summary

**Model**: xlstm_lmd_preprocessed_512d_4096ctx_12b
**Date**: 2026-03-03 16:25

## Model Configuration
- Embedding Dim: 512
- Num Blocks: 12
- Training Context: 4096
- Vocab Size: 435

## Best Checkpoint
- **Checkpoint**: checkpoint-46000
- **PPL at Training Context**: 1.6406

## Checkpoint Selection Results
| checkpoint        |   step |   context_length |   perplexity |
|:------------------|-------:|-----------------:|-------------:|
| checkpoint-2000   |   2000 |             4096 |      2.95047 |
| checkpoint-24000  |  24000 |             4096 |      1.75288 |
| checkpoint-46000  |  46000 |             4096 |      1.64064 |
| checkpoint-68000  |  68000 |             4096 |      1.69085 |
| checkpoint-90000  |  90000 |             4096 |      1.86286 |
| checkpoint-116000 | 116000 |             4096 |      1.80987 |

## Test Results (All Context Lengths)
| checkpoint       |   step |   context_length |   perplexity |
|:-----------------|-------:|-----------------:|-------------:|
| checkpoint-46000 |  46000 |             1024 |      1.82238 |
| checkpoint-46000 |  46000 |             2048 |      1.72064 |
| checkpoint-46000 |  46000 |             3072 |      1.67702 |
| checkpoint-46000 |  46000 |             4096 |      1.6557  |
| checkpoint-46000 |  46000 |             5120 |      1.64668 |
| checkpoint-46000 |  46000 |            10240 |      1.88624 |
