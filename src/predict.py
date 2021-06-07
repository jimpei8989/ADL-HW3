import os
import json
import concurrent.futures

# Make Tensorflow less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed

from datasets.utils import load_dataset
from models.utils import load_model_and_tokenizer

from utils import disable_tensorflow_gpu
from utils.tqdmm import tqdmm

# Disable GPU for Tensorflow
disable_tensorflow_gpu()


def main(args):
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    dataset = load_dataset(
        args.dataset_json_path,
        tokenizer=tokenizer,
        tokenizer_return_tensors="pt",
        include_id=True,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.predict_batch_size, num_workers=args.num_workers
    )

    model.to(args.device)

    predictions = []
    for batch in tqdmm(
        dataloader,
        desc="Predicting",
        total=(len(dataset) - 1) // args.predict_batch_size + 1,
        leave=True,
    ):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].squeeze(1).to(args.device),
                min_length=1,
                max_length=dataset.content_max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **args.gen_kwargs,
            ).cpu()
            # batch_outputs.append(outputs)
            # ids.extend(list(batch["id"]))
        output_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for output_seq, ID in zip(output_sequences, batch["id"]):
            predictions.append({"title": output_seq, "id": ID})

    # def decode(batch_output):
    #     return tokenizer.batch_decode(batch_output, skip_special_tokens=True)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     decoded = sum(executor.map(decode, batch_outputs), [])
    #     for p, i in zip(decoded, ids):
    #         predictions.append({"title": p, "id": i})

    with open(args.prediction_output_path, "w") as f:
        for pred in predictions:
            print(json.dumps(pred, ensure_ascii=False), file=f)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=Path)
    parser.add_argument("--dataset_json_path", type=Path, default=Path("dataset") / "public.jsonl")
    parser.add_argument(
        "--prediction_output_path", type=Path, default=Path("predictions") / "test.jsonl"
    )

    parser.add_argument("--predict_batch_size", type=int, default=1)
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    parser.add_argument("--no_compute_rouge", dest="compute_rouge", action="store_false")

    parser.add_argument("--seed", type=int, default=0x06902029)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_sample", dest="do_sample", action="store_true")
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--temperature", type=float)

    args = parser.parse_args()

    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.gen_kwargs = {}
    args_dict = vars(args)
    for attr in ["do_sample", "num_beams", "top_k", "top_p", "temperature"]:
        if args_dict.get(attr) is not None:
            args.gen_kwargs.update({attr: args_dict.get(attr)})

    if args.model_name_or_path is None:
        args.model_name_or_path = "google/mt5-small"

    return args


if __name__ == "__main__":
    main(parse_arguments())
