from pathlib import Path
import re
import argparse

import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
from jiwer import wer, cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def set_generation_defaults(model):
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    return model


def load_base_model(model_name: str, device: str):
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    return set_generation_defaults(model)


def load_adapter_model(model_name: str, adapter_dir: Path, device: str):
    base = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    base = set_generation_defaults(base)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    return set_generation_defaults(model)


@torch.no_grad()
def transcribe_dataset(model, processor, df_eval, model_label, device, max_new_tokens=32):
    model.eval()

    refs, preds = [], []
    rows = []

    for _, row in df_eval.iterrows():
        wav_path = row["file_path"]
        ref = row["text_clean"]

        y, _ = librosa.load(wav_path, sr=16000)
        inp = processor(y, sampling_rate=16000, return_tensors="pt").to(device)

        gen = model.generate(
            inp.input_features,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )

        pred = processor.batch_decode(gen, skip_special_tokens=True)[0]
        pred = normalize_text(pred)

        refs.append(ref)
        preds.append(pred)

        rows.append(
            {
                "model": model_label,
                "file_path": wav_path,
                "reference": ref,
                "prediction": pred,
            }
        )

    summary = {
        "model": model_label,
        "wer": wer(refs, preds),
        "cer": cer(refs, preds),
        "accuracy": sum(p == r for p, r in zip(preds, refs)) / len(refs),
        "n_clips": len(refs),
    }

    return summary, pd.DataFrame(rows)


def save_plots(summary_df: pd.DataFrame, out_dir: Path):
    stage_order = ["baseline", "stage1", "stage2", "stage3"]
    plot_df = summary_df.copy()
    plot_df["stage_order"] = plot_df["model"].apply(lambda x: stage_order.index(x))
    plot_df = plot_df.sort_values("stage_order")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["model"], plot_df["wer"], marker="o")
    plt.title("WER Progression Across Stages")
    plt.xlabel("Model")
    plt.ylabel("WER")
    plt.tight_layout()
    plt.savefig(out_dir / "wer_progression.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["model"], plot_df["cer"], marker="o")
    plt.title("CER Progression Across Stages")
    plt.xlabel("Model")
    plt.ylabel("CER")
    plt.tight_layout()
    plt.savefig(out_dir / "cer_progression.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["model"], plot_df["accuracy"], marker="o")
    plt.title("Accuracy Progression Across Stages")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_progression.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", type=str, required=True, help="CSV with file_path,text")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small.en")
    parser.add_argument("--stage1_adapter", type=str, required=True)
    parser.add_argument("--stage2_adapter", type=str, required=True)
    parser.add_argument("--stage3_adapter", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.eval_csv)
    if not {"file_path", "text"}.issubset(df.columns):
        raise ValueError("Evaluation CSV must contain file_path and text columns.")

    df["file_path"] = df["file_path"].astype(str)
    df = df[df["file_path"].apply(lambda p: Path(p).exists())].copy().reset_index(drop=True)
    df["text_clean"] = df["text"].apply(normalize_text)

    processor = WhisperProcessor.from_pretrained(args.model_name)

    models = {
        "baseline": load_base_model(args.model_name, device),
        "stage1": load_adapter_model(args.model_name, Path(args.stage1_adapter), device),
        "stage2": load_adapter_model(args.model_name, Path(args.stage2_adapter), device),
        "stage3": load_adapter_model(args.model_name, Path(args.stage3_adapter), device),
    }

    summary_rows = []
    prediction_frames = []

    for name, model in models.items():
        print(f"Running inference for: {name}")
        summary, pred_df = transcribe_dataset(model, processor, df, name, device)
        summary_rows.append(summary)
        prediction_frames.append(pred_df)

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    summary_df.to_csv(out_dir / "mixed_eval_summary.csv", index=False)
    predictions_df.to_csv(out_dir / "mixed_eval_predictions.csv", index=False)
    save_plots(summary_df, out_dir)

    print("\nSaved:")
    print(out_dir / "mixed_eval_summary.csv")
    print(out_dir / "mixed_eval_predictions.csv")
    print(out_dir / "wer_progression.png")
    print(out_dir / "cer_progression.png")
    print(out_dir / "accuracy_progression.png")
    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()
