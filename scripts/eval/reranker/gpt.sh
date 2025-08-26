for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do

    MODEL_NAME=gpt-4o
    SAVE_DIR=results/reranker
    DATA_PATH=data/reranker/inst/qwen/test.parquet
    
    python src/eval/reranker/gpt.py \
        --domain_name $DOMAIN_NAME \
        --model_name $MODEL_NAME \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR

done