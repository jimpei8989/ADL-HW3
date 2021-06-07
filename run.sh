DATASET_PATH=${1}
PRDICTION_PATH=${2}

python3.9 src/predict.py \
    --dataset_json_path ${DATASET_PATH} \
    --prediction_output_path ${PREDICTION_PATH} \
    --predict_batch_size 4 \
    --do_predict
