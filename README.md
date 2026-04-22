# Evaluating Multi-Stage Personalized Models for Speech Impairment Recognition in Clean and Noisy Conditions

A research-driven applied project developed for **Sielo Robotics** to improve automatic speech recognition (ASR) for **dysarthric and slurred speech** in assistive robotics contexts.

## Overview

Off-the-shelf ASR systems often perform poorly when processing impaired speech, especially in real-world environments with background noise. This project investigates whether a **multi-stage personalization strategy** can progressively improve transcription quality and command recognition for dysarthric speech.

The project uses a staged fine-tuning pipeline built on **OpenAI Whisper (`whisper-small.en`)** and evaluates performance across clean and noisy conditions using standard ASR metrics.

## Client Context

**Client:** Sielo Robotics
**Domain:** Assistive robotics / voice-controlled systems
**Problem:** Commercial ASR systems are not sufficiently reliable for impaired speech, which limits accessibility and usability in robot control scenarios.

## Project Objective

To design and evaluate a **multi-stage ASR personalization framework** that improves recognition performance for dysarthric speech, while generating actionable recommendations for future integration into assistive robotic systems.

## Research Question

How effective is a multi-stage fine-tuning strategy in improving ASR performance for dysarthric speech under both clean and noisy conditions?

## Deliverables

### D1. Multi-Stage ASR Prototype

A staged speech recognition prototype fine-tuned for dysarthric speech using progressively more realistic data conditions:

* **Baseline model**: pre-trained Whisper model
* **Stage 1**: fine-tuned on one-word commands
* **Stage 2**: fine-tuned on clean full-sentence dysarthric-style speech
* **Stage 3**: fine-tuned on noisy full-sentence dysarthric-style speech

### D2. Experimental Evaluation Package

A structured evaluation framework comparing all stages on a fixed test set using:

* **WER** (Word Error Rate)
* **CER** (Character Error Rate)
* **Exact Match Accuracy**

### D3. Client-Focused Insights and Recommendations

A set of practical recommendations for Sielo Robotics, including:

* implications of stage-wise improvements
* deployment considerations for assistive robotics
* future evaluation metrics such as command success rate
* guidance for more realistic dataset expansion and system integration

## Methodology

This project follows a **comparative experimental design**:

1. Start with a pre-trained Whisper baseline.
2. Fine-tune the model in multiple stages with progressively harder and more realistic speech data.
3. Evaluate each stage on a **consistent held-out test set**.
4. Compare results to determine whether staged personalization improves performance.
5. Translate technical findings into practical guidance for the client.

## Model and Technical Stack

* **Backbone Model:** OpenAI Whisper (`whisper-small.en`)
* **Fine-Tuning:** LoRA / PEFT
* **Language:** Python
* **Environment:** Google Colab
* **Libraries:** Transformers, PEFT, PyTorch, Pandas, NumPy, JiWER, Torchaudio

## Dataset Design

The project uses synthetic dysarthric-style speech data generated to simulate progressively more challenging usage conditions:

* **Stage 1:** short spoken commands
* **Stage 2:** clean full-sentence speech
* **Stage 3:** noisy full-sentence speech with environmental interference

The dataset design supports controlled experimentation while approximating real-world speech variability relevant to assistive robotics.

## Evaluation Metrics

The following metrics are used across all model stages:

* **WER (Word Error Rate):** lower is better
* **CER (Character Error Rate):** lower is better
* **Exact Match Accuracy:** higher is better

These metrics help assess both transcription quality and command-level usability.

## Example Early Result

In an early stage-specific evaluation on a test subset:

| Model              | WER  | CER   | Exact Match Accuracy |
| ------------------ | ---- | ----- | -------------------- |
| Baseline Whisper   | 0.72 | 0.245 | 0.30                 |
| Stage 1 Fine-Tuned | 0.29 | 0.124 | 0.73                 |

This early improvement supports the hypothesis that task-specific personalization can significantly enhance ASR performance for impaired speech.

## Repository Structure

```text
.
├── data/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   └── evaluation/
├── notebooks/
│   ├── data_generation/
│   ├── training/
│   └── evaluation/
├── scripts/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── train_stage3.py
│   └── evaluate_all_stages.py
├── results/
│   ├── tables/
│   ├── figures/
│   └── summaries/
├── models/
│   ├── baseline/
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
├── report/
│   └── project_report.pdf
└── README.md
```

## Key Contributions

* Demonstrates a **multi-stage learning strategy** for dysarthric ASR personalization
* Evaluates model robustness in both **clean and noisy environments**
* Bridges **technical experimentation** with **client-oriented recommendations**
* Supports accessibility-focused innovation in **assistive robotics**

## Practical Relevance

This work is relevant for:

* assistive robotics developers
* accessible human-computer interaction researchers
* speech AI practitioners working with non-standard speech
* teams exploring domain adaptation for specialized ASR use cases

## Limitations

* Dataset is primarily synthetic and may not fully capture real dysarthric variability
* Results should be interpreted as a research prototype, not a production-ready deployment
* Real-user testing and robotics integration remain future work

## Future Work

* expand with real dysarthric speech samples
* test integration with live robotic command systems
* evaluate real-world command success rate
* improve robustness under mixed noise and speaker variability
* compare staged fine-tuning with alternative adaptation methods

## Author

**Jeshwanth Premkumar**
M.Eng. in Technology Innovation Management
Carleton University

## Acknowledgements

This project was developed as part of a graduate applied research initiative in collaboration with **Sielo Robotics**.

## License

This repository is intended for academic and research demonstration purposes.
