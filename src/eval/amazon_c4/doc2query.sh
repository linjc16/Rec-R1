python src/eval/amazon_c4/doc2query.py \
  --local_dir data/amazon_c4/subset \
  --save_dir results/amazon_c4/doc2query \
  --num_rewrites 5 \
  --model_name doc2query/msmarco-t5-base-v1 \
  --cache_dir /srv/local/data/linjc/hub \
  --domain_name_list Video_Games Baby Office Sports
