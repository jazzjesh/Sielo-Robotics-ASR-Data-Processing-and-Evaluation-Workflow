from pathlib import Path
import re

import pandas as pd
import torch
from datasets import Dataset, Audio
from evaluate import load
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import PeftModel, LoraConfig, get_peft_model


MODEL_NAME = "openai/whisper-small.en"
CSV_PATH = Path("data/stage2/data.csv")
STAGE1_ADAPTER_DIR = Path("models/stage1_adapter")
OUTPUT_DIR = Path("models/stage2_adapter")


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WhisperOnTheFlyCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        audios = [f["audio"]["array"] for f in features]
        feats = self.processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")

        labels = self.processor.tokenizer(
            [f["text"] for f in features],
            padding=True,
            return_tensors="pt",
        ).input_ids
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        return {"input_features": feats.input_features, "labels": labels}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(CSV_PATH)
    df["file_path"] = df["file_path"].astype(str)
    df = df[df["file_path"].apply(lambda p: Path(p).exists())].copy()
    df["text_clean"] = df["text"].apply(clean_text)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    wer_metric = load("wer")
    cer_metric = load("cer")

    def to_dataset(df_part):
        ds = Dataset.from_dict(
            {
                "audio": df_part["file_path"].tolist(),
                "text": df_part["text_clean"].tolist(),
            }
        )
        return ds.cast_column("audio", Audio(sampling_rate=16000))

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)

    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    base_model.generation_config.forced_decoder_ids = None
    base_model.generation_config.suppress_tokens = []

    stage1_model = PeftModel.from_pretrained(base_model, str(STAGE1_ADAPTER_DIR)).to(device)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(stage1_model, lora_config)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [s.lower().strip() for s in pred_str]
        label_str = [s.lower().strip() for s in label_str]

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4 if device == "cuda" else 2,
        per_device_eval_batch_size=4 if device == "cuda" else 2,
        gradient_accumulation_steps=4 if device == "cuda" else 8,
        learning_rate=1e-5,
        num_train_epochs=6,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=32,
        logging_steps=25,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=WhisperOnTheFlyCollator(processor),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved Stage 2 adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
