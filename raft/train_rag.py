import argparse
import os
from datasets import load_from_disk, Dataset
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagSequenceForGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def prepare_datasets(path: str):
    ds = load_from_disk(path)
    contexts = []
    for row in ds:
        if row.get("oracle_context"):
            contexts.append(row["oracle_context"])
        else:
            contexts.append(" ".join(row["context"]["sentences"]))
    rag_ds = ds.map(lambda ex: {"answers": [ex["cot_answer"]]}, remove_columns=[])
    passages = Dataset.from_dict({"title": [""] * len(contexts), "text": contexts})
    return rag_ds, passages


def tokenize_examples(examples, tokenizer):
    questions = examples["question"]
    answers = examples["answers"]
    inputs = tokenizer(
        questions,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            [a[0] for a in answers],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
    labels = [[(lid if lid != tokenizer.pad_token_id else -100) for lid in l] for l in labels]
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine tune RAG on RAFT dataset")
    parser.add_argument("--dataset_path", type=str, default="sample_ds4", help="Path to RAFT dataset")
    parser.add_argument("--output_dir", type=str, default="rag-ft", help="Where to store checkpoints")
    parser.add_argument("--model", type=str, default="facebook/rag-sequence-base", help="Base RAG model")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    ds, passages = prepare_datasets(args.dataset_path)
    passages_path = os.path.join(args.output_dir, "passages")
    os.makedirs(args.output_dir, exist_ok=True)
    passages.save_to_disk(passages_path)

    tokenizer = RagTokenizer.from_pretrained(args.model)
    retriever = RagRetriever.from_pretrained(
        args.model,
        index_name="custom",
        passages_path=passages_path,
    )
    model = RagSequenceForGeneration.from_pretrained(args.model, retriever=retriever)

    tokenized_ds = ds.map(lambda x: tokenize_examples(x, tokenizer), batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=2,
        logging_steps=10,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
