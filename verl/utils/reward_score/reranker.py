import re
import random
import numpy as np
import ast
import operator
import pdb
import json
import sys
import os
sys.path.append('./')

from src.Lucene.utils import ndcg_at_k

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].strip()
    else:
        print("[Error] Failed to locate model response header")
        return None, processed_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str
        

def validate_response_structure(processed_str: str, do_print: bool) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")
    validation_passed = True

    # processed_str = '<think> </think>' + processed_str
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if do_print:
            print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            if do_print:
                print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        if do_print:
            print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        if do_print:
            print("  Tag sequence validation passed")
    
    return validation_passed


def check_json_format(json_str, do_print=False):
    """Check if the given string is a valid JSON and follows the expected structure."""
    try:
        if not json_str:
            if do_print:
                print("[Error] Empty JSON string")
            return False
        
        data = json.loads(json_str)
        
        # Required keys
        required_keys = {"reranked_items"}
        if not all(key in data for key in required_keys):
            if do_print:
                print("[Error] Missing required keys in JSON")
            return False

        # Check reranked_items is a list
        if not isinstance(data["reranked_items"], list):
            if do_print:
                print("[Error] 'reranked_items' should be a list")
            return False
        
        # Check for duplicates in reranked_items
        if len(data["reranked_items"]) != len(set(data["reranked_items"])):
            if do_print:
                print("[Error] 'reranked_items' contains duplicates")
            return False
        

        return True
    except json.JSONDecodeError:
        if do_print:
            print("[Error] JSON decoding failed")
        return False
    
def calculate_answer_score(json_str, label, scores, top_k, do_print=False):
    """Calculate answer score based on final_prediction idx."""
    try:
        data = json.loads(json_str)
        pred_results = data['reranked_items']
        target = label
        answer_score = ndcg_at_k(pred_results, target, top_k, rel_scores=scores)

    except:
        print("[Error] Error in evaluation")
        answer_score = -2
    
    return answer_score

def compute_score(solution_str, ground_truth, data_source, format_reward=0.1, answer_reward=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """

    label = [str(x) for x in ground_truth['target']]
    scores = [1] * len(label)
    
    answer_text, processed_str = extract_solution(solution_str)
    
    do_print = random.randint(1, 32) == 1

    # Validate response structure
    response_format_correct = validate_response_structure(processed_str, do_print)
    json_format_correct = check_json_format(answer_text, do_print)
    format_correct = response_format_correct and json_format_correct
    
    format_score = format_reward if format_correct else -2
    # if do_print:
    #     print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    #     print(f"Format score: {format_score}")
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Target: {label} |")
    
    answer_score = 0
    if 'test' in data_source or 'val' in data_source:
        top_k = 10
    else:
        top_k = 16
    
    if format_correct and answer_text:
        answer_score = calculate_answer_score(answer_text, label, scores, top_k, do_print)

    if answer_score > 0:
        total_score = format_score + answer_score
    else:
        if format_score > 0:
            total_score = 0
        else:
            total_score = format_score
    
    if do_print:
        print("\n" + "-"*80)
        print(f" Final Score ".center(80, '-'))
        print(f"  Format: {format_score}")
        print(f"  Answer: {answer_score}")
        print(f"  Total: {total_score}")
        print("="*80 + "\n")

    return total_score


if __name__ == '__main__':
    solution_str = """<|im_start|>assistant:  <answer>{"query": "Microstructural development of human"}</answer>
"""
    ground_truth = {'target': '4983'}
    scores = compute_score(solution_str, ground_truth)
    print(scores)