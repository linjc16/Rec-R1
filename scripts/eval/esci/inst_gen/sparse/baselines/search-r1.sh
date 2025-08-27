for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do

    MODEL_PATH=PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo
    DATA_PATH=data/esci/inst/sparse/subset/test.parquet
    SAVE_DIR=results/esci
    MODEL_NAME=search-r1-esci
    
    
    CUDA_VISIBLE_DEVICES=3 python src/eval/esci/model_generate.py \
        --domain_name $DOMAIN_NAME \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --save_dir $SAVE_DIR \
        --model_name $MODEL_NAME

done