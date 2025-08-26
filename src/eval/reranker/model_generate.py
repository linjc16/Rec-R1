import argparse
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json
import os
import pdb
import re
import numpy as np

CACHE_DIR = "/srv/local/data/linjc/hub"

def evaluate_model(model_path, data_path, model_name, save_dir, domain_name, batch_size=8):
    df = pd.read_parquet(data_path)

    # only keep those ['data_source] contains domain_name, keep index unchanged
    df = df[df['data_source'].str.contains(domain_name, case=False, na=False)]
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['mapping_target'].tolist()
    qids = df.index.tolist()
    
    # if targets[i] is a array, then convert to list
    for i in range(len(targets)):
        # if is a np.ndarray, then convert to list
        if isinstance(targets[i], np.ndarray):
            targets[i] = targets[i].tolist()
    
    llm = LLM(model=model_path, dtype="bfloat16", tokenizer=model_path)
    
    sampling_params = SamplingParams(max_tokens=1024, temperature=0.6)
    generated_texts = {}

    count = 0
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_prompts = inputs[batch_start:batch_end]
    
        outputs = llm.generate(batch_prompts, sampling_params)


        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            # extract from \nAssistant:
            if "\nassistant\n" in generated_text:
                generated_text = generated_text.split("\nassistant\n")[1]
            
            # findall <answer> </answer>
            answer_pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(answer_pattern, generated_text, re.DOTALL)  # Use re.DOTALL to match multiline content

            if len(matches) > 0:
                generated_text = matches[0]
                try:
                    # json.loads(generated_text)
                    generated_text = json.loads(generated_text)['query']
                except:
                    generated_text = matches[0]

            idx = batch_start + i
            generated_texts[qids[idx]] = {
                "generated_text": generated_text,
                "target": str(targets[idx])
            }
        if count % 100 == 0:
            with open(os.path.join(save_dir, f"{model_name}_{domain_name}.json"), "w") as f:
                json.dump(generated_texts, f, indent=4)
        count += 1
    
    with open(os.path.join(save_dir, f"{model_name}_{domain_name}.json"), "w") as f:
        json.dump(generated_texts, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, choices=['Video_Games', 'Baby_Products', 'Office_Products', 'Sports_and_Outdoors'], default='Video_Games')
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/reranker/inst/qwen/test.parquet")
    parser.add_argument("--model_name", type=str, default="Qwen-inst-reranker")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    evaluate_model(args.model_path, args.data_path, args.model_name, args.save_dir, args.domain_name, args.batch_size)

if __name__ == "__main__":
    main()
