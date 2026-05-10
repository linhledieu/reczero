"""
Convert Yelp distilling_high_quality_reasons.pkl → train/val parquet for RecOne cold-start SFT.
Yelp-specific: uses address/city/categories (string) instead of brand/price/description.
"""

import os
import pickle
import random
import pandas as pd
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
DISTILL_PKL  = '/data/uqlinh/Reason4Rec/Data/Yelp_data/distilling_high_quality_reasons.pkl'
HISTORY_PKL  = '/data/uqlinh/Reason4Rec/Data/Yelp_data/train_summarizer_generation_results.pkl'
OUTPUT_DIR   = '/data/uqlinh/Reason4Rec/Data/Yelp_data/recone_sft'

VAL_SIZE     = 500
RANDOM_SEED  = 42
MAX_USER_HIS = 10
MAX_ITEM_HIS = 10
# ──────────────────────────────────────────────────────────────────────────────


def build_user_history_text(user_rows):
    text = ''
    for i, (_, his) in enumerate(user_rows.iterrows(), 1):
        title  = his.get('title', '')
        rating = float(his.get('ratings', 0))
        text += f"{i}. {title}, {rating:.1f};\n"
        asp = his.get('aspect_preference_summary', '')
        if pd.notna(asp) and asp:
            text += f"{asp}\n\n"
    return text.strip()


def build_item_history_text(row, item_rows):
    cats = row.get('categories', '')
    city = row.get('city', '')
    state = row.get('state', '')
    stars = row.get('stars_y', '')
    review_count = row.get('review_count', '')

    result = f"Title: {row['title']} | Categories: {cats} | Location: {city}, {state}"
    if pd.notna(stars):
        result += f" | Overall Stars: {stars}"
    if pd.notna(review_count):
        result += f" | Review Count: {int(review_count)}"

    if len(item_rows) > 0:
        ratings  = [f"{float(r):.1f}" for r in item_rows['ratings']]
        reviews  = [str(r)[:300] for r in item_rows['reviews'] if pd.notna(r)]
        if ratings:
            result += f" | Ratings: [{', '.join(ratings)}]"
        if reviews:
            result += f" | Reviews: [{'; '.join(reviews)}]"

    return result


def make_prompt(user_history_text, item_history_text, user_avg, item_avg):
    return (
        "You are a helpful assistant. Your task is to analyze a user's review history, "
        "summarize their preferences, analyze the given target business, and then analyze how well "
        "the given target business aligns with the user's preferences and predict a rating for that target business.\n\n"
        f"user_history: {user_history_text}\n\n"
        f"user_avg_rating: {user_avg:.2f}\n\n"
        f"target_item: {item_history_text}\n\n"
        f"target_item_avg_rating: {item_avg:.2f}\n\n"
        "Please provide your analysis and predict a rating (1-5 scale)."
    )


def make_response(personalized_analysis, rating):
    rating_int = int(round(float(rating)))
    return f"{personalized_analysis.strip()}\n\n<rate>\n{rating_int}\n</rate>"


def main():
    print("Loading Yelp distilling pkl (D_trace)...")
    distill_df = pd.read_pickle(DISTILL_PKL)
    print(f"  D_trace size: {len(distill_df)}")

    print("Loading Yelp history pkl...")
    history_df = pd.read_pickle(HISTORY_PKL)
    print(f"  History size: {len(history_df)}")

    records = []
    skipped = 0

    for idx, row in tqdm(distill_df.iterrows(), total=len(distill_df), desc="Building SFT rows"):
        uid = row['user_id']
        iid = row['item_id']
        ts  = row['unixReviewTime']

        u_his = history_df[
            (history_df['user_id'] == uid) &
            (history_df['unixReviewTime'] < ts)
        ].sort_values('unixReviewTime').tail(MAX_USER_HIS)

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
        item_history_text = build_item_history_text(row, i_his)

        prompt   = make_prompt(user_history_text, item_history_text, user_avg, item_avg)
        response = make_response(row['personalized_analysis'], row['ratings'])

        records.append({'prompt': prompt, 'response': response})

    print(f"\nBuilt {len(records)} rows, skipped {skipped} (no history)")

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    val_records   = records[:VAL_SIZE]
    train_records = records[VAL_SIZE:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(train_records).to_parquet(os.path.join(OUTPUT_DIR, 'train.parquet'), index=False)
    pd.DataFrame(val_records).to_parquet(os.path.join(OUTPUT_DIR, 'val.parquet'),   index=False)
    print(f"\nSaved to {OUTPUT_DIR}")

    # sanity check token lengths
    print("\nChecking token lengths...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/data/uqlinh/merged_models/10-3-reczero4800-tallrec480/to-test/Qwen2.5-3B-Instruct',
        trust_remote_code=True
    )
    import numpy as np
    lengths = []
    for r in train_records[:500]:
        p = tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}], add_generation_prompt=True, tokenize=True)
        resp = tokenizer(r['response'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
        lengths.append(len(p) + len(resp))
    lengths = np.array(lengths)
    print(f"  p50={int(np.percentile(lengths,50))}  p90={int(np.percentile(lengths,90))}  p95={int(np.percentile(lengths,95))}  max={lengths.max()}")


if __name__ == '__main__':
    main()
