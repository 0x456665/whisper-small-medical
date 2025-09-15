import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, DatasetDict, Audio
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Model configuration
# model_id = "openai/whisper-large-v3"
model_id = "openai/whisper-small"

# Load processor (combines feature extractor and tokenizer)
processor = WhisperProcessor.from_pretrained(model_id, language="en", task="transcribe")

# Load model
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# Configure generation
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Process audio inputs
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Process text labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch):
    """Prepare dataset for training"""
    # Load and resample audio
    audio = batch["path"]

    # Extract features using processor
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode text using processor
    batch["labels"] = processor.tokenizer(
        batch["sentence"], return_tensors="pt"
    ).input_ids[0]

    return batch


def compute_metrics(pred):
    """Compute WER metric"""
    metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    # Load and prepare dataset
    tune_dataset = DatasetDict()
    tune_dataset["train"] = load_dataset(
        "yashtiwari/PaulMooney-Medical-ASR-Data", split="train+validation"
    )
    tune_dataset["test"] = load_dataset(
        "yashtiwari/PaulMooney-Medical-ASR-Data", split="test"
    )

    # Resample audio to 16kHz
    tune_dataset = tune_dataset.cast_column("path", Audio(sampling_rate=16000))

    # Prepare dataset
    tune_dataset = tune_dataset.map(
        prepare_dataset, remove_columns=tune_dataset.column_names["train"], num_proc=2
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Disable cache explicitly (important for Whisper fine-tuning)
    model.config.use_cache = False

    # Training arguments (tuned for low VRAM)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-medical",
        per_device_train_batch_size=1,  # smallest batch
        gradient_accumulation_steps=16,  # effective batch = 16
        learning_rate=5e-5,
        warmup_steps=100,
        max_steps=1000,
        gradient_checkpointing=False,  # <--- turned off
        fp16=False,  # no mixed precision (safe for CPU / iGPU)
        eval_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Create trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=tune_dataset["train"],
        eval_dataset=tune_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,  # important fix
    )

    trainer.train()

    # Save final model
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
