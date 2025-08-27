
for DOMAIN_NAME in 'Video_Games' 'Baby' 'Office' 'Sports'; do

    MODEL_PATH=/srv/local/data/linjc/hub/dmis-lab/RetPO/llama2-7b-retpo-topiocqa/llama2-7b-retpo-topiocqa/
    DATA_PATH=data/amazon_c4/inst/sparse/subset_other/test.parquet
    SAVE_DIR=results/amazon_c4
    MODEL_NAME=retpo-amazon-c4
    

    CUDA_VISIBLE_DEVICES=7 python src/eval/amazon_c4/model_generate.py \
        --domain_name $DOMAIN_NAME \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR \
        --model_name $MODEL_NAME

done