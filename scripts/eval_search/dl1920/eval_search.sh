# TEST_FILE_PATH=results/dl1920/rec-r1-dl19.json
# TEST_FILE_PATH=results/dl1920/rec-r1-dl20.json
# TEST_FILE_PATH=results/dl1920/qwen2.5-3b-dl19.json
# TEST_FILE_PATH=results/dl1920/qwen2.5-3b-dl20.json
# TEST_FILE_PATH=results/dl1920/gpt-4o-mini-dl20.json
# TEST_FILE_PATH=results/dl1920/gpt-4o-mini-dl19.json

METRIC_RES_SAVE_DIR=results/dl1920/metric_res/query_metric_results-$QUERY_GEN_MODEL_NAME.json



python src/eval_search/BM25/dl1920.py \
    --res_path $TEST_FILE_PATH \
    --save_path $METRIC_RES_SAVE_DIR \
