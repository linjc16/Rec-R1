import argparse
import json
import re
import ast
import os
import sys
sys.path.append('./')

from src.Lucene.utils import ndcg_at_k


def extract_json_block(s: str):
    """
    Try to extract a JSON object block from a string. Returns the substring or None.
    Uses a simple brace-balance scan to pick the first plausible {...} block.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def parse_reranked_items_from_text(generated_text: str):
    """
    Robustly parse reranked_items list from a model output string.
    Steps:
      1) If <answer>...</answer> exists, use the last match as the working text.
      2) Try to JSON-decode a block and read 'reranked_items'.
      3) If JSON parse fails, try regex on 'reranked_items: [ ... ]' and extract quoted strings.
    Returns list of item_ids or None if cannot parse.
    """
    # 1) Prefer the last <answer>...</answer> block if present
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, generated_text, flags=re.DOTALL | re.IGNORECASE)
    working = matches[-1] if matches else generated_text

    # Clean common wrappers (code fences etc.)
    working = working.strip().strip("`").strip()

    # 2) Try to find a JSON object and parse it
    json_block = extract_json_block(working) or extract_json_block(generated_text)
    if json_block:
        try:
            obj = json.loads(json_block)
            # Typical shapes:
            # { "reranked_items": [...] }
            # { "generated_text": "{ \"reranked_items\": [...] }" }
            if "reranked_items" in obj and isinstance(obj["reranked_items"], list):
                return [str(x) for x in obj["reranked_items"]]
            if "generated_text" in obj and isinstance(obj["generated_text"], str):
                inner_block = extract_json_block(obj["generated_text"])
                if inner_block:
                    inner = json.loads(inner_block)
                    if "reranked_items" in inner and isinstance(inner["reranked_items"], list):
                        return [str(x) for x in inner["reranked_items"]]
        except Exception:
            pass  # fall through to regex parsing

    # # 3) Regex: find the reranked_items array and extract quoted strings in order
    # # Capture the inside of the array
    # arr_match = re.search(r'"reranked_items"\s*:\s*\[(.*?)\]', working, flags=re.DOTALL | re.IGNORECASE)
    # if not arr_match:
    #     arr_match = re.search(r'"reranked_items"\s*:\s*\[(.*?)\]', generated_text, flags=re.DOTALL | re.IGNORECASE)

    # if arr_match:
    #     inner = arr_match.group(1)
    #     # Extract quoted tokens preserving order: "Item_xxx"
    #     items = re.findall(r'"([^"]+)"', inner)
    #     if items:
    #         return [str(x) for x in items]

    # # As a last ditch, extract any JSON-like top-level array
    # any_arr = re.search(r'\[\s*(?:".*?")(?:\s*,\s*".*?")*\s*\]', working, flags=re.DOTALL)
    # if not any_arr:
    #     any_arr = re.search(r'\[\s*(?:".*?")(?:\s*,\s*".*?")*\s*\]', generated_text, flags=re.DOTALL)
    # if any_arr:
    #     try:
    #         arr = json.loads(any_arr.group(0))
    #         if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
    #             return arr
    #     except Exception:
    #         pass

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="results/reranker/gpt-4o-mini-esci_Office_Products.json",
        help="Path to the reranker result JSON file."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="k value for NDCG@k."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = 0
    skipped_parse = 0
    skipped_empty_target = 0
    ndcg_list = []

    for _key, rec in data.items():
        generated_text = rec.get("generated_text", "")
        target_str = rec.get("target", "[]")

        # Parse reranked items
        reranked = parse_reranked_items_from_text(generated_text)
        if not reranked:
            skipped_parse += 1
            ndcg_list.append(0.0)
            continue
        
        if len(reranked) != 16:
            skipped_parse += 1
            ndcg_list.append(0.0)
            continue
        
        # Parse target (string repr list)
        try:
            target = ast.literal_eval(target_str)
        except Exception:
            # If target isn't parsable, skip
            skipped_empty_target += 1
            continue

        if not isinstance(target, list) or len(target) == 0:
            skipped_empty_target += 1
            continue

        # Compute ndcg@k
        ndcg = ndcg_at_k(reranked, target, args.k, [1] * len(target))
        ndcg_list.append(ndcg)
        parsed += 1

    # Report
    print(f"Parsed samples: {parsed}")
    print(f"Skipped (parse failed): {skipped_parse}")
    print(f"Skipped (empty/invalid target): {skipped_empty_target}")

    if ndcg_list:
        avg_ndcg = sum(ndcg_list) / len(ndcg_list)
        print(f"Average NDCG@{args.k}: {avg_ndcg:.4f}")
    else:
        print(f"No valid samples to compute NDCG@{args.k}.")


if __name__ == "__main__":
    main()
