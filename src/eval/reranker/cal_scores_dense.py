import argparse
import json
import os
import glob

import sys
sys.path.append('./')

from src.Lucene.utils import ndcg_at_k


def eval_file(path: str, k: int = 10):
    """Evaluate one jsonl file and return (avg_ndcg, count, skipped)."""
    ndcgs = []
    total = 0
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                ranked = obj.get("ranked", [])
                target = obj.get("target", [])
                # Basic validation
                if not isinstance(ranked, list) or not isinstance(target, list) or len(target) == 0:
                    import pdb; pdb.set_trace()
                    skipped += 1
                    continue
                # Compute ndcg@k
                ndcg = ndcg_at_k(ranked, target, k, [1] * len(target))
                ndcgs.append(ndcg)
            except Exception:
                import pdb; pdb.set_trace()
                skipped += 1
                continue

    avg = (sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
    used = len(ndcgs)
    return avg, used, skipped, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/reranker/qwen3-8b",
        help="Path to a jsonl file or a directory containing multiple jsonl files."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="qwen3_rerank_test_*.jsonl",
        help="Glob pattern when --input is a directory (e.g., 'qwen3_rerank_test_*.jsonl')."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="k for NDCG@k (default: 10)."
    )
    args = parser.parse_args()

    paths = []
    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, args.pattern)))
    else:
        paths = [args.input]

    if not paths:
        print(f"[WARN] No files found for input: {args.input}")
        return

    overall_ndcgs = []
    overall_used = 0
    overall_skipped = 0
    overall_total = 0

    for p in paths:
        avg, used, skipped, total = eval_file(p, k=args.k)
        overall_ndcgs.append((avg, used))
        overall_used += used
        overall_skipped += skipped
        overall_total += total
        print(f"[{os.path.basename(p)}] samples used: {used}/{total - skipped} (valid/parsed), skipped: {skipped}, Average NDCG@{args.k}: {avg:.4f}")

    if overall_used > 0:
        # Weighted average by sample count from each file
        weighted_avg = sum(avg * used for avg, used in overall_ndcgs) / overall_used
    else:
        weighted_avg = 0.0

    print("-" * 60)
    print(f"Overall: files={len(paths)}, total_lines={overall_total}, used={overall_used}, skipped={overall_skipped}")
    print(f"Average NDCG@{args.k} (weighted): {weighted_avg:.4f}")


if __name__ == "__main__":
    main()
