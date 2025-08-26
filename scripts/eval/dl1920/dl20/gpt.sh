MODEL_NAME=gpt-4o-mini
SAVE_DIR=results/dl1920
DATA_PATH=data/dl20/inst/sparse/test.parquet
DATASET=dl20

python src/eval/dl1920/gpt.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --dataset $DATASET