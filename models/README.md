# Models

This folder documents the model stages used in this project.

## Model Versions
- **Baseline:** Pre-trained Whisper (`whisper-small.en`)
- **Stage 1:** Fine-tuned on one-word commands
- **Stage 2:** Fine-tuned on clean full-sentence speech
- **Stage 3:** Fine-tuned on noisy full-sentence speech

## Notes
Full model checkpoints are not included in this repository due to size limitations.

The project uses a multi-stage fine-tuning strategy with LoRA / PEFT.
