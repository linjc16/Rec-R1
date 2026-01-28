for DOMAIN_NAME in 'Video_Games' 'Baby' 'Office' 'Sports'; do
    echo $DOMAIN_NAME
    # QUERY_GEN_MODEL_NAME=Qwen-inst
    # QUERY_GEN_MODEL_NAME=rec-r1
    # QUERY_GEN_MODEL_NAME=search-r1
    # QUERY_GEN_MODEL_NAME=gpt-4o-mini
    # QUERY_GEN_MODEL_NAME=retpos
    # QUERY_GEN_MODEL_NAME=rej-cross
    QUERY_GEN_MODEL_NAME=convsearch-r1
    # QUERY_GEN_MODEL_NAME=gpt-4o
    TEST_FILE_PATH=results/amazon_c4/$QUERY_GEN_MODEL_NAME-amazon-c4_$DOMAIN_NAME.json
    
    python src/eval_search/BM25/amazon_c4.py \
        --res_path $TEST_FILE_PATH

done