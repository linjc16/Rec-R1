DOMAIN_NAME=esci
SPLIT=$1
MODEL_PATH=checkpoints/Rec-R1-esci/esci-qwen2.5-3b-inst-grpo-2gpus/actor/global_step_1400
DATA_PATH=data/esci/inst/sparse/subset/$SPLIT.parquet
SAVE_DIR=results/esci/$SPLIT
MODEL_NAME=rec-r1-esci


CUDA_VISIBLE_DEVICES=5 python src/eval/esci/model_generate.py \
    --domain_name $DOMAIN_NAME \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME