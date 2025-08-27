for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do

    MODEL_NAME=gpt-4o-mini
    SAVE_DIR=results/esci
    DATA_PATH=data/esci/inst/sparse/subset/test.parquet
    
    python src/eval/esci/gpt.py \
        --domain_name $DOMAIN_NAME \
        --model_name $MODEL_NAME \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR

done