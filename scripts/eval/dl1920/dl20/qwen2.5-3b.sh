MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
DATA_PATH=data/dl20/inst/sparse/test.parquet
SAVE_DIR=results/dl1920
MODEL_NAME=qwen2.5-3b-dl20


CUDA_VISIBLE_DEVICES=5 python src/eval/dl1920/model_generate.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME