# MODEL_NAME=roberta-base
# MODEL_NAME=simcse-base
# MODEL_NAME=simcse-large
# MODEL_NAME=roberta-large
# MODEL_NAME=blair-base
MODEL_NAME=blair-large

for DOMAIN_NAME in 'Video_Games' 'Baby' 'Office' 'Sports'; do
    echo $DOMAIN_NAME
    
    python src/eval_search/Dense/amazon_c4.py \
        --test_data_dir=data/amazon_c4/subset \
        --model_name $MODEL_NAME \
        --domain $DOMAIN_NAME 

done