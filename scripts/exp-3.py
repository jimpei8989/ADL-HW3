import subprocess as sp
import os

CHECKPOINT_PATH = "checkpoints/BETA/checkpoint-20480"

EXPERIMENTS = [
    {"name": "0_greedy", "args": ""},
    {"name": "1-1_beam_3", "args": "--num_beams 3"},
    {"name": "1-2_beam_5", "args": "--num_beams 5"},
    {"name": "2-1_sample", "args": "--use_sample"},
    {"name": "2-2_sample_beam", "args": "--use_sample --num_beams 3"},
    {"name": "3-1_topk_10", "args": "--use_sample --top_k 10"},
    {"name": "3-2_topk_25", "args": "--use_sample --top_k 25"},
    {"name": "4-1_topp_0.25", "args": "--use_sample --top_p 0.25"},
    {"name": "4-2_topp_0.50", "args": "--use_sample --top_p 0.5"},
    {"name": "5-1_temperature_0.5", "args": "--use_sample --temperature 0.5"},
    {"name": "5-2_temperature_2.0", "args": "--use_sample --temperature 2.0"},
]


def run_experiment(name, args):
    print(f"====== {name} ======")

    prediction_path = f"predictions/exp-3/{name}.jsonl"

    if os.path.isfile(prediction_path):
        print("Prediction file already exists, skipping......")
    else:
        sp.run(
            " ".join(
                [
                    "python3 src/predict.py",
                    f"--model_name_or_path {CHECKPOINT_PATH}",
                    "--predict_batch_size 32 --num_workers 4",
                    f"--prediction_output_path {prediction_path}",
                    args,
                ]
            ),
            shell=True,
        )

    sp.run(
        " ".join(
            [
                "python3 scripts/eval.py",
                "-r dataset/public.jsonl",
                f"-s {prediction_path}",
            ]
        ),
        shell=True,
        stderr=sp.DEVNULL,
    )


def main():
    for exp in EXPERIMENTS:
        run_experiment(**exp)


main()
