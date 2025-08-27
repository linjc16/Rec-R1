from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import pandas as pd
import pdb

CACHE_DIR = "/srv/local/data/linjc/hub"

# load model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=CACHE_DIR)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer("")

pdb.set_trace()