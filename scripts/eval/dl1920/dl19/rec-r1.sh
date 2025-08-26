MODEL_PATH=/shared/rsaas/jl254/code/DeepRetrieval/checkpoints/msmarco_beir_search/msmarco_beir_search_3b/actor/global_step_400
DATA_PATH=data/dl19/inst/sparse/test.parquet
SAVE_DIR=results/dl1920
MODEL_NAME=rec-r1-dl19


CUDA_VISIBLE_DEVICES=5 python src/eval/dl1920/model_generate.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME