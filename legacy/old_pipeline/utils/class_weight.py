import torch
import pandas as pd
def compute_pos_weight(df, presence_cols):
    # 强制转为数值，非法值变 NaN
    presence_df = df[presence_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # NaN 当作 0（即“未出现”）
    presence_df = presence_df.fillna(0.0)

    counts = presence_df.sum(axis=0)
    neg = len(presence_df) - counts

    pos_weight = neg / (counts + 1e-6)

    return torch.tensor(
        pos_weight.values,
        dtype=torch.float32
    )