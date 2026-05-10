"""
Convert distilling_high_quality_reasons.pkl → train/val parquet for RecOne cold-start SFT.

Each row in the pkl has:
  - user_id, item_id
  - ratings          : ground truth rating (float, 1-5)
  - title            : target item title
  - description      : target item description
  - categories, brand, price : item metadata
  - reviews          : one review of the target item
  - summary          : review summary
  - personalized_analysis : teacher LLM reasoning trajectory (the response to learn)
  - (no pre-built user_history / item_history columns — we build them from train_summarizer)

The SFT trainer (fsdp_sft_trainer) reads a parquet with two plain-string columns:
  - prompt   : the user-turn text (no chat template yet — trainer applies it)
  - response : the assistant-turn text to learn (personalized_analysis + rating)

We use the same prompt format as amazon_rate_format.py (qwen-instruct template body),
but strip the <|im_start|>/<|im_end|> tokens since the trainer applies chat_template itself.
"""

import os
import pickle
import random
import json

import pandas as pd
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
DISTILL_PKL  = '/data/uqlinh/Reason4Rec/Data/Book_data/distilling_high_quality_reasons.pkl'
HISTORY_PKL  = '/data/uqlinh/Reason4Rec/Data/Book_data/train_summarizer_generation_results.pkl'
OUTPUT_DIR   = '/data/uqlinh/Reason4Rec/Data/Book_data/recone_sft'

VAL_SIZE     = 500    # held-out rows for validation
RANDOM_SEED  = 42
MAX_USER_HIS = 10     # last N interactions to include in user history
MAX_ITEM_HIS = 10     # last N interactions for item history
# ──────────────────────────────────────────────────────────────────────────────


def format_item_fixed(row):
    parts = []
    if pd.notna(row.get('title')):
        parts.append(f"Title: {row['title']}")
    cats = row.get('categories')
    if cats is not None and not (isinstance(cats, float) and pd.isna(cats)):
        parts.append(f"Categories: {cats}")
    if pd.notna(row.get('brand')):
        parts.append(f"Brand: {row['brand']}")
    if pd.notna(row.get('price')):
        parts.append(f"Price: {row['price']}")
    if pd.notna(row.get('description')):
        desc = str(row['description'])[:400]
        parts.append(f"Description: {desc}")
    return " | ".join(parts)


def build_user_history_text(user_rows):
    """Build numbered user history string from chronologically sorted rows."""
    text = ''
    for i, (_, his) in enumerate(user_rows.iterrows()):
        title  = his.get('title', '')
        rating = float(his.get('ratings', 0))
        text += f"{i+1}. {title}, {rating:.1f};\n"
        asp = his.get('aspect_preference_summary', '')
        if pd.notna(asp) and asp:
            text += f"{asp}\n\n"
    return text.strip()


def build_item_history_text(item_rows):
    """Aggregate item history: fixed features once, then ratings/reviews array."""
    if len(item_rows) == 0:
        return ""
    result = format_item_fixed(item_rows.iloc[0])
    ratings  = [f"{float(r):.1f}" for r in item_rows['ratings']]
    reviews  = []
    summaries = []
    for _, rec in item_rows.iterrows():
        if pd.notna(rec.get('reviews')):
            reviews.append(str(rec['reviews'])[:400])
        if pd.notna(rec.get('summary')):
            summaries.append(str(rec['summary'])[:200])
    if ratings:
        result += f" | Ratings: [{', '.join(ratings)}]"
    if reviews:
        result += f" | Reviews: [{'; '.join(reviews)}]"
    if summaries:
        result += f" | Summaries: [{'; '.join(summaries)}]"
    return result


def make_prompt(user_history_text, item_history_text, user_avg, item_avg):
    return (
        "You are a helpful assistant. Your task is to analyze a user's purchase history, "
        "summarize their preferences, analyze the given target item, and then analyze how well "
        "the given target item aligns with the user's preferences and predict a rating for that target item.\n\n"
        f"user_history: {user_history_text}\n\n"
        f"user_avg_rating: {user_avg:.2f}\n\n"
        f"target_item: {item_history_text}\n\n"
        f"target_item_avg_rating: {item_avg:.2f}\n\n"
        "Please provide your analysis and predict a rating (1-5 scale)."
    )


def make_response(personalized_analysis, rating):
    """
    Plain prose teacher reasoning + ground truth rating tag.
    SFT teaches general reasoning ability; RL then shapes the structured format via reward.
    """
    rating_int = int(round(float(rating)))
    return f"{personalized_analysis.strip()}\n\n<rate>\n{rating_int}\n</rate>"


def main():
    print("Loading distilling pkl (D_trace)...")
    distill_df = pd.read_pickle(DISTILL_PKL)
    print(f"  D_trace size: {len(distill_df)}")

    print("Loading history pkl (train_summarizer)...")
    history_df = pd.read_pickle(HISTORY_PKL)
    print(f"  History size: {len(history_df)}")

    records = []
    skipped = 0

    for idx, row in tqdm(distill_df.iterrows(), total=len(distill_df), desc="Building SFT rows"):
        uid  = row['user_id']
        iid  = row['item_id']
        ts   = row['unixReviewTime']

        # ── user history: all interactions before this timestamp ──
        u_his = history_df[
            (history_df['user_id'] == uid) &
            (history_df['unixReviewTime'] < ts)
        ].sort_values('unixReviewTime').tail(MAX_USER_HIS)

        # ── item history: all interactions before this timestamp ──
        i_his = history_df[
            (history_df['item_id'] == iid) &
            (history_df['unixReviewTime'] < ts)
        ].sort_values('unixReviewTime').tail(MAX_ITEM_HIS)

        if len(u_his) == 0 or len(i_his) == 0:
            skipped += 1
            continue

        user_avg = float(u_his['ratings'].mean())
        item_avg = float(i_his['ratings'].mean())

        user_history_text = build_user_history_text(u_his)
        item_history_text = build_item_history_text(i_his)

        prompt   = make_prompt(user_history_text, item_history_text, user_avg, item_avg)
        response = make_response(row['personalized_analysis'], row['ratings'])

        records.append({'prompt': prompt, 'response': response})

    print(f"\nBuilt {len(records)} rows, skipped {skipped} (no history)")

    # ── split ──────────────────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    val_records   = records[:VAL_SIZE]
    train_records = records[VAL_SIZE:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    # ── save ───────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.DataFrame(train_records)
    val_df   = pd.DataFrame(val_records)

    train_path = os.path.join(OUTPUT_DIR, 'train.parquet')
    val_path   = os.path.join(OUTPUT_DIR, 'val.parquet')

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path,   index=False)

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")

    # ── sanity check ───────────────────────────────────────────────────────────
    print("\n=== Sanity check (first train row) ===")
    sample = train_records[0]
    print("PROMPT (first 500 chars):")
    print(sample['prompt'][:500])
    print("\nRESPONSE (first 500 chars):")
    print(sample['response'][:500])


if __name__ == '__main__':
    main()
