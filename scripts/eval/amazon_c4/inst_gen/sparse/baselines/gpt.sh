for DOMAIN_NAME in 'Video_Games' 'Baby' 'Office' 'Sports'; do

    MODEL_NAME=gpt-4o-mini
    SAVE_DIR=results/amazon_c4
    DATA_PATH=data/amazon_c4/inst/sparse/subset_other/test.parquet
    
    python src/eval/amazon_c4/gpt.py \
        --domain_name $DOMAIN_NAME \
        --model_name $MODEL_NAME \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR

done