for DOMAIN_NAME in 'Video_Games' 'Baby' 'Office' 'Sports'; do
    echo $DOMAIN_NAME
    # MODEL_NAME=blair-base
    MODEL_NAME=blair-large
    # QUERY_GEN_MODEL_NAME=Qwen-inst
    QUERY_GEN_MODEL_NAME=rec-r1
    # QUERY_GEN_MODEL_NAME=gpt-4o
    TEST_FILE_PATH=results_dense/amazon_c4/$QUERY_GEN_MODEL_NAME-amazon-c4_$DOMAIN_NAME.json
    

    CUDA_VISIBLE_DEVICES=0 python src/eval_search/Dense/amazon_c4.py \
        --model_name $MODEL_NAME \
        --test_file_path $TEST_FILE_PATH

done