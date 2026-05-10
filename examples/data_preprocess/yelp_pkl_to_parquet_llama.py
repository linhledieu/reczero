"""
Llama-3.2-3B-Instruct version of yelp_pkl_to_parquet.py.

Identical logic, but build_prompt uses Llama-3 chat tokens
(<|begin_of_text|> / <|start_header_id|> / <|end_header_id|> / <|eot_id|>)
instead of Qwen2 tokens (<|im_start|> / <|im_end|>).

Convert Yelp raw pkl files into the same schema as /workspace/reczero/data/yelp/test.parquet.

For val/test splits, pass --history_pkl_path pointing to train.pkl so that user/item
history is looked up from the training set (chronologically prior data).

Output columns:
  user_history, item_history, ground_truth, user_avg, item_avg,
  data_source, prompt, ability, reward_model, extra_info

Usage examples:
  # train (history within train itself)
  python yelp_pkl_to_parquet_llama.py --pkl_path train.pkl --output_path train.parquet --split train

  # val/test (history from train)
  python yelp_pkl_to_parquet_llama.py --pkl_path val.pkl --history_pkl_path train.pkl --output_path val.parquet --split val
  python yelp_pkl_to_parquet_llama.py --pkl_path test.pkl --history_pkl_path train.pkl --output_path test.parquet --split test
"""

import argparse
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to analyze a user's purchase history, "
    "summarize their preferences, analyze the given target item, and then analyze how well "
    "the given target item aligns with the user's preferences and predict a rating for that target item. \n"
    "Please follow these steps precisely:\n\n"
    "    1.  **Extract User Interest From Purchase History:**\n"
    "        Based on the provided user_purchase_history, consolidate the user's overall preferences. "
    "Adhere strictly to the following format:\n"
    "        ```\n"
    "        <analyze user>\n"
    "        User Preference: \n"
    "        ...\n"
    "        </analyze user>\n\n"
    "        ```\n"
    "    2.  **Summarize The Given Target Item's Key Aspects:** Analyze the features and metadata of the "
    "target item provided in `target_item`. Predict potential points the user might like (`[pos]`) or "
    "dislike (`[neg]`) about this specific item, considering general product attributes. Adhere strictly "
    "to the following format:\n"
    "        ```\n"
    "        <analyze item>\n"
    "        Product Name\n"
    "        [pos] Summary of positive points\n"
    "        [neg] Summary of negative points\n"
    "        </analyze item>\n\n"
    "        ```\n"
    "    3.  **Evaluate Compatibility:** Internally reason (`<match>`) about *why* the target item would "
    "(or would not) be a good recommendation for this user. This reasoning should explicitly connect the "
    "user's overall preferences (from Step 1) with the target item's potential positive/negative aspects "
    "(from Step 2). Provide detailed justifications.\n"
    "        ```\n"
    "        <match>\n"
    "        Reasoning here\n"
    "        </match>\n\n"
    "        ```\n"
    "    4.  **Predict Rating:**\n"
    "        Provide the final predicted rating for the target item within the `<rate>` tags. "
    "The answer should *only* contain the predicted numerical rating (e.g., on a 1-5 scale).\n"
    "        Adhere strictly to the following format:\n"
    "        ```\n"
    "        <rate>\n"
    "        Predicted rating for the target item\n"
    "        </rate>\n\n"
    "        ```\n"
    "    Your inputs will be `user_history`,  `target_item`, `user_avg_rating` and "
    "`target_item_avg_rating`. Ensure your output follows the specified formats and uses the "
    "`<analyze user>`, `<analyze item>`, `<match>` and `<rate>` tags correctly."
)


def format_item_raw_features(row, review_limit=400, summary_limit=200):
    parts = []
    if pd.notna(row.get("title")):
        parts.append(f"Title: {row['title']}")
    cats = row.get("categories")
    if cats is not None and not (isinstance(cats, float) and pd.isna(cats)):
        parts.append(f"Categories: {cats}")
    if pd.notna(row.get("reviews")):
        rev = str(row["reviews"])
        if len(rev) > review_limit:
            rev = rev[:review_limit] + "..."
        parts.append(f"Review: {rev}")
    return " | ".join(parts)


def build_item_history_text(item_history: pd.DataFrame, review_limit=400) -> str:
    if len(item_history) == 0:
        return ""
    first = item_history.iloc[0]
    parts = []
    if pd.notna(first.get("title")):
        parts.append(f"Title: {first['title']}")
    cats = first.get("categories")
    if cats is not None and not (isinstance(cats, float) and pd.isna(cats)):
        parts.append(f"Categories: {cats}")

    ratings, reviews = [], []
    for _, rec in item_history.iterrows():
        ratings.append(f"{float(rec['ratings']):.1f}")
        if pd.notna(rec.get("reviews")):
            rev = str(rec["reviews"])
            if len(rev) > review_limit:
                rev = rev[:review_limit] + "..."
            reviews.append(rev)

    if ratings:
        parts.append(f"Ratings: [{', '.join(ratings)}]")
    if reviews:
        parts.append(f"Reviews: [{'; '.join(reviews)}]")
    return " | ".join(parts)


def build_prompt(user_history_text: str, item_history_text: str,
                 user_avg: float, item_avg: float) -> np.ndarray:
    """
    Build a Llama-3 formatted prompt array.

    Llama-3 chat tokens:
      <|begin_of_text|>
      <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
      <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\n\n
    """
    user_block = f"user_history: {user_history_text}"
    item_block = f"target_item: {item_history_text}"
    user_content = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"    {user_block}\n\n"
        f"    user_avg_rating: {user_avg}\n\n"
        f"    {item_block}\n\n"
        f"    target_item_avg_rating: {item_avg}\n\n"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return np.array([{"role": "user", "content": user_content}], dtype=object)


def main(pkl_path: str, output_path: str, min_his_len: int, max_user_his: int,
         max_item_his: int, split: str, history_pkl_path: Optional[str]):
    print(f"Loading target split: {pkl_path} ...")
    data_df = pd.read_pickle(pkl_path)
    print(f"Loaded {len(data_df)} rows.")

    if history_pkl_path:
        print(f"Loading history source: {history_pkl_path} ...")
        history_df = pd.read_pickle(history_pkl_path)
        print(f"History source: {len(history_df)} rows.")
    else:
        history_df = data_df

    ts_col = "unixReviewTime" if "unixReviewTime" in history_df.columns else "date"

    records = []
    skipped = 0
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
        user_id = row["user_id"]
        item_id = row["item_id"]
        ts = row[ts_col]

        u_hist = history_df[(history_df["user_id"] == user_id) & (history_df[ts_col] < ts)]
        u_hist = u_hist.sort_values(ts_col, ascending=True).tail(max_user_his)

        i_hist = history_df[(history_df["item_id"] == item_id) & (history_df[ts_col] < ts)]
        i_hist = i_hist.sort_values(ts_col, ascending=True).tail(max_item_his)

        if len(u_hist) == 0 or len(i_hist) == 0:
            skipped += 1
            continue
        if len(u_hist) + len(i_hist) < min_his_len:
            skipped += 1
            continue

        user_avg = float(u_hist["ratings"].mean())
        item_avg = float(i_hist["ratings"].mean())

        user_history_text = ""
        for i, (_, h) in enumerate(u_hist.iterrows()):
            user_history_text += f"{i + 1}. {h['title']}, {float(h['ratings']):.1f};\n"
            user_history_text += format_item_raw_features(h) + "\n\n"
        user_history_text = user_history_text.strip()

        item_history_text = build_item_history_text(i_hist)
        ground_truth = float(row["ratings"])

        prompt = build_prompt(user_history_text, item_history_text, user_avg, item_avg)

        records.append({
            "user_history": user_history_text,
            "item_history": item_history_text,
            "ground_truth": ground_truth,
            "user_avg": user_avg,
            "item_avg": item_avg,
            "data_source": "rec_rate_format",
            "prompt": prompt,
            "ability": "rec_rate",
            "reward_model": {"ground_truth": {"ground_truth": str(int(ground_truth))}, "style": "rule"},
            "extra_info": {"index": len(records), "split": split},
        })

    print(f"Built {len(records)} samples ({skipped} skipped — no history).")
    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", default="/workspace/Data/Yelp_data/raw_data/test.pkl")
    parser.add_argument("--history_pkl_path", default=None,
                        help="Optional separate history source (e.g. train.pkl for val/test splits)")
    parser.add_argument("--output_path", default="/workspace/reczero/data/yelp/test_new.parquet")
    parser.add_argument("--min_his_len", type=int, default=5)
    parser.add_argument("--max_user_his", type=int, default=10)
    parser.add_argument("--max_item_his", type=int, default=10)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    main(args.pkl_path, args.output_path, args.min_his_len, args.max_user_his,
         args.max_item_his, args.split, args.history_pkl_path)
