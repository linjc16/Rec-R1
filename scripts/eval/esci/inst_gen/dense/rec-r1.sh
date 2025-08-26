for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do

    MODEL_PATH=checkpoints/Rec-R1-esci-Dense/esci-dense-qwen2.5-3b-inst-grpo-2gpus/actor/global_step_400
    DATA_PATH=data/esci/inst/dense/subset/test.parquet
    SAVE_DIR=results_dense/esci
    MODEL_NAME=rec-r1-esci


    CUDA_VISIBLE_DEVICES=7 python src/eval/esci/model_generate.py \
        --domain_name $DOMAIN_NAME \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR \
        --model_name $MODEL_NAME

done