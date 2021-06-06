from pathlib import Path

from .generation_dataset import GenerationDataset


def load_datasets(dataset_dir: Path, tokenizer=None):
    train_dataset = GenerationDataset.from_jsonl(dataset_dir / "train.jsonl", tokenizer=tokenizer)
    val_dataset = GenerationDataset.from_jsonl(dataset_dir / "public.jsonl", tokenizer=tokenizer)
    return train_dataset, val_dataset
