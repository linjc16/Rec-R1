for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do

    MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
    DATA_PATH=data/reranker/inst/qwen/test.parquet
    SAVE_DIR=results/reranker
    MODEL_NAME=qwen2.5-3b-inst-reranker


    CUDA_VISIBLE_DEVICES=4 python src/eval/reranker/model_generate.py \
        --domain_name $DOMAIN_NAME \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR \
        --model_name $MODEL_NAME

done