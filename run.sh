DATASET_PATH=${1}
PREDICTION_PATH=${2}

MODEL_NAME_OR_PATH="models/model/"

python3.9 src/predict.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_json_path ${DATASET_PATH} \
    --prediction_output_path ${PREDICTION_PATH} \
    --predict_batch_size 32 --num_workers 4 \
    --num_beams 3
