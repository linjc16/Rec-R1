

for DOMAIN_NAME in 'Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors'; do
    echo $DOMAIN_NAME
    # QUERY_GEN_MODEL_NAME=Qwen-inst
    # QUERY_GEN_MODEL_NAME=gpt-4o
    # QUERY_GEN_MODEL_NAME=Qwen-0.5b-inst
    # QUERY_GEN_MODEL_NAME=Qwen-1.5b-inst
    # QUERY_GEN_MODEL_NAME=Qwen-7b-inst
    # QUERY_GEN_MODEL_NAME=llama3-3b-inst
    # QUERY_GEN_MODEL_NAME=claude-haiku
    # QUERY_GEN_MODEL_NAME=Qwen-3b-rej-sft
    # QUERY_GEN_MODEL_NAME=Qwen-3b-dpo
    # QUERY_GEN_MODEL_NAME=rec-r1-esci-cross-AC4
    # QUERY_GEN_MODEL_NAME=claude-3.5
    # QUERY_GEN_MODEL_NAME=rec-r1
    # QUERY_GEN_MODEL_NAME=search-r1
    # QUERY_GEN_MODEL_NAME=gpt-4o-mini
    # QUERY_GEN_MODEL_NAME=retpo
    # QUERY_GEN_MODEL_NAME=Qwen-sft
    # QUERY_GEN_MODEL_NAME=Qwen-sft-no-reason
    # QUERY_GEN_MODEL_NAME=Qwen-sft-no-json-no-reason
    # QUERY_GEN_MODEL_NAME=Qwen-sft-with-reason-no-json
    QUERY_GEN_MODEL_NAME=Qwen2.5-3b-rej-sft
    TEST_FILE_PATH=results/esci/$QUERY_GEN_MODEL_NAME-esci_$DOMAIN_NAME.json
    # TEST_FILE_PATH=results/esci/rec-r1-esci-multi-reward_$DOMAIN_NAME.json
    METRIC_RES_SAVE_DIR=results/esci/metric_res/query_metric_results-$QUERY_GEN_MODEL_NAME-$DOMAIN_NAME.json



    python src/eval_search/BM25/esci.py \
        --res_path $TEST_FILE_PATH \
        --save_path $METRIC_RES_SAVE_DIR \

done