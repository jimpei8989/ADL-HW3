# Homework 3 - Natural Language Generation
> Applied Deep Learning (CSIE 5431)

## Shortcuts
- [Instruction slides (Google slides)](https://docs.google.com/presentation/d/1-a0Z8-sV6hudbraxD1FBXIgFNzyotMDDZOIEHIrXvTo/edit)
- [Dataset (Google Drive)](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view)
- There's no Kaggle competition page this time

## Environment
- Python `3.9.4`
- Requirements: please refer to [requirements.txt](requirements.txt)
    - Note that I installed transformers directly from GitHub, following the guide in slide.
    - And the tw_rouge is installed from <https://github.com/ntu-adl-ta/ADL21-HW3>
- Virtual environment using `pyenv`
- CPU: AMD Ryzen 7 3700X
- GPU: NVIDIA GeForce RTX 2070 Super

## Reproduce My Best Model
```bash
python3 src/train.py \
    --name FINAL \
    --learning_rate 1e-4 --weight_decay 1e-5 --num_epochs 20 \
    --train_batch_size 2 --eval_batch_size 4
```

## Run evaluation
- To download the latest model
    ```bash
    bash download.sh
    ```

- To run evaluation
    ```bash
    bash run.sh DATASET_INPUT_JSONL PREDICTION_OUTPUT_JSONL
    ```
