SPLIT=$1

TEST_FILE_PATH=results/esci/$SPLIT/rec-r1-esci_esci.json
METRIC_RES_SAVE_DIR=results/esci/metric_res/$SPLIT/query_metric_results-rec-r1.json



python src/eval_search/BM25/esci.py \
    --res_path $TEST_FILE_PATH \
    --save_path $METRIC_RES_SAVE_DIR \
