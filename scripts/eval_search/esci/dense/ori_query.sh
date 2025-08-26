
for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do
    echo $DOMAIN_NAME
    # MODEL_NAME=blair-base
    MODEL_NAME=blair-large
    TEST_DATA_DIR=data/esci/test_subset
    
    CUDA_VISIBLE_DEVICES=0 python src/eval_search/Dense/esci.py \
        --domain $DOMAIN_NAME \
        --model_name $MODEL_NAME \
        --test_data_dir $TEST_DATA_DIR

done