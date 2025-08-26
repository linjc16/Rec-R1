
for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do
    echo $DOMAIN_NAME
    MODEL_NAME=blair-base
    # MODEL_NAME=blair-large
    # QUERY_GEN_MODEL_NAME=Qwen-inst
    # QUERY_GEN_MODEL_NAME=gpt-4o
    QUERY_GEN_MODEL_NAME=rec-r1
    TEST_FILE_PATH=results_dense/esci/$QUERY_GEN_MODEL_NAME-esci_$DOMAIN_NAME.json


    CUDA_VISIBLE_DEVICES=0 python src/eval_search/Dense/esci.py \
        --model_name $MODEL_NAME \
        --test_file_path $TEST_FILE_PATH

done