from pathlib import Path

from .generation_dataset import GenerationDataset


def load_dataset(json_path: Path, **kwargs):
    return GenerationDataset.from_jsonl(json_path, **kwargs)


def load_datasets(dataset_dir: Path, **kwargs):
    train_dataset = GenerationDataset.from_jsonl(dataset_dir / "train.jsonl", **kwargs)
    val_dataset = GenerationDataset.from_jsonl(dataset_dir / "public.jsonl", **kwargs)
    return train_dataset, val_dataset
