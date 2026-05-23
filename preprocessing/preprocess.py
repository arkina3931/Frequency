#!/usr/bin/env python3
"""
Amazon Multimodal Dataset Preprocessing Pipeline
整合自: 0rating2inter / 1splitting / 2reindex-feat / 3feat-encoder / dualgnn-gen-u-u-matrix

Usage:
    python preprocess.py -d sports --steps all
    python preprocess.py -d baby  --steps 0 1 2 3 dualgnn
    python preprocess.py -d sports --steps 0 --raw-file ratings_Sports_and_Outdoors.csv
"""

import os
import sys
import gzip
import json
import array
import argparse
import numpy as np
import pandas as pd
import yaml
from collections import Counter, defaultdict
from tqdm import tqdm


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Amazon dataset preprocessing pipeline")
    parser.add_argument("-d", "--dataset",    type=str, required=True,
                        help="Dataset name, e.g. sports / baby / games")
    parser.add_argument("--data-dir",         type=str, default="./",
                        help="Directory containing raw dataset files (default: ./)")
    parser.add_argument("--out-dir",          type=str, default=None,
                        help="Output directory (default: same as --data-dir)")
    parser.add_argument("--raw-file",         type=str, default=None,
                        help="Ratings CSV filename (default: ratings_<DATASET>.csv)")
    parser.add_argument("--meta-file",        type=str, default=None,
                        help="Meta JSON.gz filename (default: meta_<DATASET>.json.gz)")
    parser.add_argument("--img-feat-file",    type=str, default=None,
                        help="Image feature binary file (default: image_features_<DATASET>.b)")
    parser.add_argument("--k-core",           type=int, default=5,
                        help="k-core filtering threshold (default: 5)")
    parser.add_argument("--steps", nargs="+", default=["all"],
                        help="Steps to run: 0 1 2 3 dualgnn  or  all (default: all)")
    # DualGNN specific
    parser.add_argument("--src-dir",          type=str, default="../src",
                        help="src/ directory for DualGNN config loading (default: ../src)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Step 0: 5-core filtering + re-indexing + timestamp split
# ──────────────────────────────────────────────

def get_illegal_ids(df, field, min_num):
    ids = df[field].values
    inter_num = Counter(ids)
    return {id_ for id_, cnt in inter_num.items() if cnt < min_num}


def filter_k_core(df, uid_field, iid_field, k):
    print(f"  Running {k}-core filtering ...")
    while True:
        ban_u = get_illegal_ids(df, uid_field, k)
        ban_i = get_illegal_ids(df, iid_field, k)
        if not ban_u and not ban_i:
            break
        mask = df[uid_field].isin(ban_u) | df[iid_field].isin(ban_i)
        df.drop(df.index[mask], inplace=True)
        print(f"    Removed {mask.sum()} interactions | remaining: {df.shape[0]}")
    df.reset_index(drop=True, inplace=True)
    return df


def step0_rating2inter(cfg):
    """Load ratings CSV → 5-core → re-index → timestamp split → write .inter + mappings."""
    print("\n[Step 0] 5-core filtering, re-indexing, timestamp split")

    raw_file = cfg.raw_file or f"ratings_{cfg.dataset_title}.csv"
    raw_path = os.path.join(cfg.data_dir, raw_file)
    print(f"  Loading: {raw_path}")

    df = pd.read_csv(raw_path, names=["userID", "itemID", "rating", "timestamp"], header=None)
    print(f"  Raw shape: {df.shape}")

    uid_field, iid_field, ts_field = "userID", "itemID", "timestamp"

    # Dedup
    df.dropna(subset=[uid_field, iid_field, ts_field], inplace=True)
    df.drop_duplicates(subset=[uid_field, iid_field, ts_field], inplace=True)
    print(f"  After dedup: {df.shape}")

    # 5-core
    filter_k_core(df, uid_field, iid_field, cfg.k_core)
    print(f"  After {cfg.k_core}-core: {df.shape}")

    # Re-index
    uni_users = pd.unique(df[uid_field])
    uni_items = pd.unique(df[iid_field])
    u_id_map = {k: i for i, k in enumerate(uni_users)}
    i_id_map = {k: i for i, k in enumerate(uni_items)}

    df[uid_field] = df[uid_field].map(u_id_map).astype(int)
    df[iid_field] = df[iid_field].map(i_id_map).astype(int)

    # Save mappings
    u_df = pd.DataFrame(list(u_id_map.items()), columns=["user_id", "userID"])
    i_df = pd.DataFrame(list(i_id_map.items()), columns=["asin", "itemID"])
    u_df.to_csv(os.path.join(cfg.out_dir, "u_id_mapping.csv"), sep="\t", index=False)
    i_df.to_csv(os.path.join(cfg.out_dir, "i_id_mapping.csv"), sep="\t", index=False)
    print("  Mapping files saved: u_id_mapping.csv, i_id_mapping.csv")

    # Timestamp-based 80/10/10 split (produces initial .inter; Step 1 will redo per-user)
    ratios = [0.8, 0.1, 0.1]
    split_ratios = np.cumsum(ratios)[:-1]
    split_ts = list(np.quantile(df[ts_field], split_ratios))

    df_train = df[df[ts_field] <  split_ts[0]].copy()
    df_val   = df[(df[ts_field] >= split_ts[0]) & (df[ts_field] < split_ts[1])].copy()
    df_test  = df[df[ts_field] >= split_ts[1]].copy()

    df_train["x_label"] = 0
    df_val["x_label"]   = 1
    df_test["x_label"]  = 2

    inter_df = pd.concat([df_train, df_val, df_test])
    inter_df = inter_df[[uid_field, iid_field, "rating", ts_field, "x_label"]]

    inter_file = f"{cfg.dataset}-indexed.inter"
    inter_path = os.path.join(cfg.out_dir, inter_file)
    inter_df.to_csv(inter_path, sep="\t", index=False)
    print(f"  Saved: {inter_file}  shape={inter_df.shape}")
    print(f"  Users: {inter_df[uid_field].nunique()}, Items: {inter_df[iid_field].nunique()}")

    cfg._inter_file = inter_file  # pass to next step


# ──────────────────────────────────────────────
# Step 1: Per-user train/val/test split
# ──────────────────────────────────────────────

def step1_splitting(cfg):
    """Reload .inter → per-user split → write -v4.inter."""
    print("\n[Step 1] Per-user train/val/test splitting")

    inter_file = getattr(cfg, "_inter_file", f"{cfg.dataset}-indexed.inter")
    inter_path = os.path.join(cfg.out_dir, inter_file)
    print(f"  Loading: {inter_path}")

    df = pd.read_csv(inter_path, sep="\t")
    uid_field, iid_field = "userID", "itemID"

    # Shuffle then group by user (preserves random order within user)
    df = df.sample(frac=1).reset_index(drop=True)
    df.sort_values(by=[uid_field], inplace=True)

    uid_freq = df.groupby(uid_field)[iid_field]
    u_i_dict = {u: list(items) for u, items in uid_freq}

    new_label = []
    for u in sorted(u_i_dict.keys()):
        items = u_i_dict[u]
        n = len(items)
        if n < 10:
            tmp = [0] * (n - 2) + [1, 2]
        else:
            val_test_len = int(n * 0.2)
            train_len = n - val_test_len
            val_len   = val_test_len // 2
            test_len  = val_test_len - val_len
            tmp = [0] * train_len + [1] * val_len + [2] * test_len
        new_label.extend(tmp)

    df["x_label"] = new_label

    out_file = inter_file.replace(".inter", "-v4.inter")
    out_path = os.path.join(cfg.out_dir, out_file)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved: {out_file}  shape={df.shape}")
    print(f"  Train: {(df['x_label']==0).sum()}, Val: {(df['x_label']==1).sum()}, Test: {(df['x_label']==2).sum()}")

    cfg._inter_v4_file = out_file


# ──────────────────────────────────────────────
# Step 2: Reindex meta features by generated item IDs
# ──────────────────────────────────────────────

def parse_gz_meta(path):
    """Parse Amazon-style meta JSON.gz (one eval'd dict per line)."""
    records = []
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                records.append(eval(line))
            except Exception:
                continue
    return pd.DataFrame(records)


def step2_reindex_feat(cfg):
    """Map meta file ASINs to generated itemIDs → write meta-<dataset>.csv."""
    print("\n[Step 2] Reindexing item features")

    mapping_path = os.path.join(cfg.out_dir, "i_id_mapping.csv")
    id_df = pd.read_csv(mapping_path, sep="\t")
    print(f"  Loaded mapping: {mapping_path}  ({len(id_df)} items)")

    meta_file = cfg.meta_file or f"meta_{cfg.dataset_title}.json.gz"
    meta_path = os.path.join(cfg.data_dir, meta_file)
    print(f"  Loading meta file: {meta_path}")

    meta_df = parse_gz_meta(meta_path)
    print(f"  Raw meta records: {meta_df.shape}")

    map_dict = dict(zip(id_df["asin"], id_df["itemID"]))
    meta_df["itemID"] = meta_df["asin"].map(map_dict)
    meta_df.dropna(subset=["itemID"], inplace=True)
    meta_df["itemID"] = meta_df["itemID"].astype("int64")
    meta_df.sort_values(by=["itemID"], inplace=True)

    # Put itemID first
    cols = meta_df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    meta_df = meta_df[cols]

    out_file = f"meta-{cfg.dataset}.csv"
    out_path = os.path.join(cfg.out_dir, out_file)
    meta_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_file}  shape={meta_df.shape}")
    print(f"  itemID range: {meta_df['itemID'].min()} ~ {meta_df['itemID'].max()}")

    cfg._meta_file = out_file


# ──────────────────────────────────────────────
# Step 3: Text + Image feature encoding
# ──────────────────────────────────────────────

def read_image_features(path):
    """Read Amazon binary image feature file (4096-dim float per ASIN)."""
    with open(path, "rb") as f:
        while True:
            asin = f.read(10).decode("UTF-8")
            if asin == "":
                break
            a = array.array("f")
            a.fromfile(f, 4096)
            yield asin, a.tolist()


def step3_feat_encoder(cfg):
    """Encode text (SentenceTransformer) and image (binary .b file) features."""
    print("\n[Step 3] Feature encoding (text + image)")

    meta_file = getattr(cfg, "_meta_file", f"meta-{cfg.dataset}.csv")
    meta_path = os.path.join(cfg.out_dir, meta_file)
    print(f"  Loading: {meta_path}")

    df = pd.read_csv(meta_path)
    df.sort_values(by=["itemID"], inplace=True)

    # ── Text ──────────────────────────────────
    print("  Encoding text features ...")
    for col in ["description", "title", "brand", "categories"]:
        if col in df.columns:
            df[col] = df[col].fillna(" ")

    sentences = []
    for _, row in df.iterrows():
        sen = row.get("title", " ") + " " + row.get("brand", " ") + " "
        cats = row.get("categories", " ")
        try:
            cates = eval(cats)
            if isinstance(cates, list) and cates:
                for c in cates[0]:
                    sen += c + " "
        except Exception:
            pass
        sen += row.get("description", " ")
        sen = sen.replace("\n", " ")
        sentences.append(sen)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        txt_embeddings = model.encode(sentences, show_progress_bar=True)
        assert txt_embeddings.shape[0] == df.shape[0]
        txt_out = os.path.join(cfg.out_dir, "text_feat.npy")
        np.save(txt_out, txt_embeddings)
        print(f"  Text features saved: text_feat.npy  shape={txt_embeddings.shape}")
    except ImportError:
        print("  [SKIP] sentence_transformers not installed. Run: pip install sentence-transformers")

    # ── Image ─────────────────────────────────
    img_file = cfg.img_feat_file or f"image_features_{cfg.dataset_title}.b"
    img_path = os.path.join(cfg.data_dir, img_file)

    if not os.path.isfile(img_path):
        print(f"  [SKIP] Image feature file not found: {img_path}")
        return

    print(f"  Loading image features: {img_path}")
    item2id = dict(zip(df["asin"], df["itemID"])) if "asin" in df.columns else {}

    feats = {}
    avg_pool = []
    for asin, feat in read_image_features(img_path):
        if asin in item2id:
            feats[int(item2id[asin])] = feat
            avg_pool.append(feat)

    avg = np.array(avg_pool).mean(0).tolist() if avg_pool else [0.0] * 4096
    n_items = df["itemID"].max() + 1
    ret = []
    missed = []
    for i in range(n_items):
        if i in feats:
            ret.append(feats[i])
        else:
            missed.append(i)
            ret.append(avg)

    img_out = os.path.join(cfg.out_dir, "image_feat.npy")
    np.save(img_out, np.array(ret))
    missed_out = os.path.join(cfg.out_dir, "missed_img_itemIDs.csv")
    np.savetxt(missed_out, missed, delimiter=",", fmt="%d")
    print(f"  Image features saved: image_feat.npy  shape={np.array(ret).shape}")
    print(f"  Items without image: {len(missed)} → {missed_out}")


# ──────────────────────────────────────────────
# DualGNN: user-user co-interaction matrix
# ──────────────────────────────────────────────

def gen_user_matrix(all_edge, no_users):
    import torch
    edge_dict = defaultdict(set)
    for user, item in all_edge:
        edge_dict[user].add(item)

    key_list = sorted(edge_dict.keys())
    user_graph_matrix = torch.zeros(no_users, no_users)

    for head in tqdm(range(len(key_list)), desc="  Building u-u matrix"):
        for rear in range(head + 1, len(key_list)):
            h, r = key_list[head], key_list[rear]
            inter_len = len(edge_dict[h] & edge_dict[r])
            if inter_len > 0:
                user_graph_matrix[h][r] = inter_len
                user_graph_matrix[r][h] = inter_len

    return user_graph_matrix


def step_dualgnn(cfg):
    """Generate user-user graph dict for DualGNN."""
    import torch

    print(f"\n[Step DualGNN] Generating u-u graph for {cfg.dataset}")

    # Load config (mirroring original script)
    src_dir = os.path.abspath(cfg.src_dir)
    con_dir = os.path.join(src_dir, "configs")
    conf_files = [
        os.path.join(con_dir, "overall.yaml"),
        os.path.join(con_dir, "dataset", f"{cfg.dataset}.yaml"),
    ]
    config = {}
    for f in conf_files:
        if os.path.isfile(f):
            with open(f, "r", encoding="utf-8") as fh:
                config.update(yaml.safe_load(fh) or {})

    if not config:
        print("  [WARN] No config files found. Falling back to direct paths.")
        inter_v4 = getattr(cfg, "_inter_v4_file",
                           f"{cfg.dataset}-indexed-v4.inter")
        inter_path = os.path.join(cfg.out_dir, inter_v4)
        uid_field, iid_field = "userID", "itemID"
        out_dir = cfg.out_dir
        graph_file = "user_graph_dict.npy"
    else:
        dataset_path = os.path.abspath(config["data_path"] + cfg.dataset)
        uid_field  = config["USER_ID_FIELD"]
        iid_field  = config["ITEM_ID_FIELD"]
        inter_path = os.path.join(dataset_path, config["inter_file_name"])
        out_dir    = dataset_path
        graph_file = config["user_graph_dict_file"]

    print(f"  Loading interactions: {inter_path}")
    train_df = pd.read_csv(inter_path, sep="\t")
    num_user = train_df[uid_field].nunique()
    train_df = train_df[train_df["x_label"] == 0].copy()
    train_data = train_df[[uid_field, iid_field]].to_numpy()

    user_graph_matrix = gen_user_matrix(train_data, num_user)

    # Build sparse top-k dict
    user_graph_dict = {}
    for i in range(num_user):
        nonzero_cnt = int(torch.count_nonzero(user_graph_matrix[i]))
        k = min(nonzero_cnt, 200)
        if k > 0:
            topk = torch.topk(user_graph_matrix[i], k)
            user_graph_dict[i] = [
                topk.indices.numpy().tolist(),
                topk.values.numpy().tolist(),
            ]
        else:
            user_graph_dict[i] = [[], []]

    out_path = os.path.join(out_dir, graph_file)
    np.save(out_path, user_graph_dict, allow_pickle=True)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# Dataset name → title-case helper
# ──────────────────────────────────────────────

# Maps short dataset name to the title used in raw file names
# (matches Amazon review dataset filenames, e.g. ratings_Sports_and_Outdoors.csv)
DATASET_TITLE_MAP = {
    # ── Amazon 2018 per-category datasets ─────────────────────────────────
    "fashion":          "Amazon_Fashion",
    "beauty":           "All_Beauty",
    "appliances":       "Appliances",
    "arts":             "Arts_Crafts_and_Sewing",
    "automotive":       "Automotive",
    "books":            "Books",
    "cds":              "CDs_and_Vinyl",
    "vinyl":            "CDs_and_Vinyl",
    "phones":           "Cell_Phones_and_Accessories",
    "cell":             "Cell_Phones_and_Accessories",
    "clothing":         "Clothing_Shoes_and_Jewelry",
    "music":            "Digital_Music",
    "electronics":      "Electronics",
    "giftcards":        "Gift_Cards",
    "grocery":          "Grocery_and_Gourmet_Food",
    "food":             "Grocery_and_Gourmet_Food",
    "home":             "Home_and_Kitchen",
    "kitchen":          "Home_and_Kitchen",
    "industrial":       "Industrial_and_Scientific",
    "scientific":       "Industrial_and_Scientific",
    "kindle":           "Kindle_Store",
    "luxurybeauty":     "Luxury_Beauty",
    "luxury":           "Luxury_Beauty",
    "magazine":         "Magazine_Subscriptions",
    "movies":           "Movies_and_TV",
    "tv":               "Movies_and_TV",
    "instruments":      "Musical_Instruments",
    "office":           "Office_Products",
    "patio":            "Patio_Lawn_and_Garden",
    "garden":           "Patio_Lawn_and_Garden",
    "pets":             "Pet_Supplies",
    "baby":             "Baby",               # older dataset subset
    "pantry":           "Prime_Pantry",
    "software":         "Software",
    "sports":           "Sports_and_Outdoors",
    "tools":            "Tools_and_Home_Improvement",
    "toys":             "Toys_and_Games",
    "games":            "Video_Games",
    "videogames":       "Video_Games",
}


def resolve_dataset_title(name):
    return DATASET_TITLE_MAP.get(name.lower(), name)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve output dir
    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    # Bundle config
    class Cfg:
        pass
    cfg = Cfg()
    cfg.dataset       = args.dataset
    cfg.dataset_title = resolve_dataset_title(args.dataset)
    cfg.data_dir      = args.data_dir
    cfg.out_dir       = out_dir
    cfg.raw_file      = args.raw_file
    cfg.meta_file     = args.meta_file
    cfg.img_feat_file = args.img_feat_file
    cfg.k_core        = args.k_core
    cfg.src_dir       = args.src_dir

    steps = args.steps
    if "all" in steps:
        steps = ["0", "1", "2", "3", "dualgnn"]

    step_map = {
        "0":       step0_rating2inter,
        "1":       step1_splitting,
        "2":       step2_reindex_feat,
        "3":       step3_feat_encoder,
        "dualgnn": step_dualgnn,
    }

    print(f"Dataset : {cfg.dataset}  ({cfg.dataset_title})")
    print(f"Data dir: {cfg.data_dir}")
    print(f"Out  dir: {cfg.out_dir}")
    print(f"Steps   : {steps}")

    for s in steps:
        if s not in step_map:
            print(f"[WARN] Unknown step '{s}', skipping.")
            continue
        step_map[s](cfg)

    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()