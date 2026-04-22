# Results

This folder contains the main evaluation outputs for the Sielo Robotics personalized ASR project.

## Core Metrics
The project compares four stages on a fixed final evaluation set using:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Exact-Match Accuracy

## Stage-wise Results

| Stage | Exact Match | CER | WER |
|---|---:|---:|---:|
| Baseline | 0.195 | 0.495 | 0.750 |
| Stage 1 | 0.188 | 0.478 | 0.715 |
| Stage 2 | 0.245 | 0.518 | 0.693 |
| Stage 3 | 0.422 | 0.264 | 0.390 |

## Key Takeaway
Stage 3 achieved the strongest overall performance, reducing WER by 48% and CER by 47% relative to the baseline, while improving exact-match accuracy by more than 2×.
