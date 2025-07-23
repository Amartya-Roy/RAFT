import argparse
import os
import torch
from datasets import load_from_disk, Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)


def prepare_datasets(path: str):
    ds = load_from_disk(path)
    contexts = []
    for row in ds:
        if row.get("oracle_context"):
            contexts.append(row["oracle_context"])
        else:
            contexts.append(" ".join(row["context"]["sentences"]))
    
    # Create input text that includes context
    def format_input(example, context):
        return f"Context: {context}\n\nQuestion: {example['question']}"
    
    formatted_ds = []
    for i, example in enumerate(ds):
        formatted_ds.append({
            "input_text": format_input(example, contexts[i]),
            "target_text": example["cot_answer"]
        })
    
    return Dataset.from_list(formatted_ds)


def tokenize_examples(examples, tokenizer, max_length=512):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    labels = tokenizer(
        targets,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    
    # Replace pad tokens with -100 so they're ignored in loss
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine tune BART on RAFT dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="sample_ds4",
        help="Path to RAFT dataset",
    )
    parser.add_argument("--output_dir", type=str, default="bart-ft", help="Where to store checkpoints")
    parser.add_argument("--model", type=str, default="facebook/bart-base", help="Base model")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # Load and prepare dataset
    ds = prepare_datasets(args.dataset_path)
    
    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenize_examples(examples, tokenizer)
    
    tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    
    # Set up training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()