from argparse import ArgumentParser
from pathlib import Path

from generation_dataset import GenerationDataset
from utils.logger import logger


def main(args):
    train_dataset = GenerationDataset(args.dataset_dir / "train.jsonl")

    logger.info("=== Training Split ===")
    train_dataset.overview()

    logger.info("=== Validation Split ===")
    val_dataset = GenerationDataset(args.dataset_dir / "public.jsonl")
    val_dataset.overview()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
