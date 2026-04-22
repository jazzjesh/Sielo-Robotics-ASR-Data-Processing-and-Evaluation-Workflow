from pathlib import Path
import argparse
import json
from datetime import datetime

import pandas as pd
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel


def set_generation_defaults(model):
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    return model


def load_model(stage: str, model_name: str, adapter_dir: str | None, device: str):
    if stage == "baseline":
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        return set_generation_defaults(model)

    if not adapter_dir:
        raise ValueError(f"adapter_dir is required for stage '{stage}'")

    base = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    base = set_generation_defaults(base)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    return set_generation_defaults(model)


@torch.no_grad()
def transcribe_file(model, processor, wav_path: str, device: str, max_new_tokens: int = 32):
    y, _ = librosa.load(wav_path, sr=16000)
    inp = processor(y, sampling_rate=16000, return_tensors="pt").to(device)

    gen = model.generate(
        inp.input_features,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
    )
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def collect_audio_files(input_path: Path):
    if input_path.is_file():
        return [input_path]

    exts = {".wav", ".mp3", ".flac", ".m4a"}
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baseline", "stage1", "stage2", "stage3"], default="stage3")
    parser.add_argument("--input_path", required=True, help="Audio file or folder")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model_name", default="openai/whisper-small.en")
    parser.add_argument("--adapter_dir", default=None, help="Required for stage1/stage2/stage3")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}_{args.stage}"
    run_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = load_model(args.stage, args.model_name, args.adapter_dir, device)

    audio_files = collect_audio_files(Path(args.input_path))
    if not audio_files:
        raise FileNotFoundError("No audio files found in input_path")

    rows = []
    for wav_path in audio_files:
        pred = transcribe_file(model, processor, str(wav_path), device)
        rows.append(
            {
                "audio_path": str(wav_path),
                "hyp_text": pred,
                "model_stage": args.stage,
                "timestamp": timestamp,
            }
        )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved predictions to: {run_dir / 'predictions.csv'}")
    print(f"Saved config to: {run_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
