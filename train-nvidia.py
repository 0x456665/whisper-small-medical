import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model

# ----------------------------
# 1. Load dataset
# ----------------------------
# Replace with your dataset path or Hugging Face dataset
dataset = load_dataset("path/to/your/dataset")

# Resample audio to 16kHz (required by Whisper)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 80/10/10 split
splits = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_valid = splits["test"].train_test_split(test_size=0.5, seed=42)
train_dataset = splits["train"]
eval_dataset = test_valid["train"]
test_dataset = test_valid["test"]

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Test: {len(test_dataset)}")

# ----------------------------
# 2. Load processor & model
# ----------------------------
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Apply LoRA (parameter-efficient fine-tuning)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)

# ----------------------------
# 3. Preprocessing function
# ----------------------------
def prepare_dataset(batch):
    audio = batch["audio"]
    input_features = processor(
        audio["array"], sampling_rate=16000, return_tensors="pt"
    ).input_features[0]

    labels = processor.tokenizer(
        batch["text"], return_tensors="pt", padding="longest"
    ).input_ids[0]

    batch["input_features"] = input_features
    batch["labels"] = labels
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)

# ----------------------------
# 4. Training Arguments
# ----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=10,
    learning_rate=1e-4,
    warmup_steps=500,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=225,
    fp16=True,
    push_to_hub=False,
    report_to="none",
)

# ----------------------------
# 5. Trainer
# ----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor,
)

# ----------------------------
# 6. Train
# ----------------------------
trainer.train()

# ----------------------------
# 7. Evaluate
# ----------------------------
metrics = trainer.evaluate(test_dataset)
print(metrics)

# Save final model
model.save_pretrained("./whisper-base-lora-final")
processor.save_pretrained("./whisper-base-lora-final")
