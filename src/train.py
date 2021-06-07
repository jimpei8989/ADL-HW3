import os

# Make Tensorflow less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import set_seed

from datasets.utils import load_datasets
from models.utils import load_model_and_tokenizer
from metrics.rouge_score import RougeScore

from utils import disable_tensorflow_gpu
from utils.io import json_dump

# Disable GPU for Tensorflow
disable_tensorflow_gpu()


def main(args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    json_dump(vars(args), args.checkpoint_dir / "args.json")

    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer("models/mt5-small")
    train_dataset, val_dataset = load_datasets(Path("dataset/"), tokenizer=tokenizer)

    # val_dataset.size = 128

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.checkpoint_dir,
        overwrite_output_dir=True,
        seed=args.seed,
        evaluation_strategy="steps",
        fp16=args.fp16,
        num_train_epochs=args.num_epochs,
        adafactor=args.use_adafactor,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        dataloader_num_workers=args.num_workers,
        logging_steps=512 // args.train_batch_size,
        logging_dir=args.tb_log_dir,
        eval_steps=2048 // args.train_batch_size,
        save_steps=2048 // args.train_batch_size,
    )

    compute_metrics = None if not args.compute_rouge else RougeScore(tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--name", default="TEST")
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--tb_log_dir", type=Path)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    # Trainer
    parser.add_argument("--use_adafactor", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--no_compute_rouge", dest="compute_rouge", action="store_false")

    parser.add_argument("--seed", type=int, default=0x06902029)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path("checkpoints") / args.name

    if args.tb_log_dir is None:
        args.tb_log_dir = Path("runs") / (args.name + "_" + datetime.now().strftime("%m%d-%H%M"))

    return args


if __name__ == "__main__":
    main(parse_arguments())
