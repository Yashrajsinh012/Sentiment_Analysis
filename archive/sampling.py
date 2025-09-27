import pandas as pd
import os
from scripts.utils import normalize

def sampling_from_scores(inferenced_file, total=600, alpha=0.6, beta=0.0, output_path=None, path=None):
    df = pd.read_parquet(inferenced_file).copy()
    required_cols = {"rid", "text", "confidence_score", "entropy"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in inference file: {missing}")

    entropy_norm = normalize(df["entropy"])
    inv_confidence_norm = normalize(1 - df["confidence_score"])
    df["uncertainty_score"] = beta * entropy_norm + (1 - beta) * inv_confidence_norm

    top_k = int(alpha * total)
    bottom_k = int((1 - alpha) * total)

    top_df = df.nlargest(top_k, "uncertainty_score").reset_index(drop=True)
    bottom_df = df.nsmallest(bottom_k, "uncertainty_score").reset_index(drop=True)

    sampled_df = pd.concat([top_df, bottom_df])
    if output_path:
        sampled_df.to_parquet(output_path, index=False)

    sampled_rids = set(sampled_df["rid"])
    remaining_df = df[~df["rid"].isin(sampled_rids)][["rid", "text"]].reset_index(drop=True)
    if path:
        if os.path.exists(path):
            print(f"Warning: {os.path.basename(path)} will be overwritten.")
        remaining_df.to_parquet(path, index=False)

    print(remaining_df.head(3))
    return sampled_df, remaining_df

if __name__ == "__main__":
    inferenced_filepath = "/teamspace/studios/this_studio/Final_pipeline/inferences/inference_06.parquet"
    out_path = "/teamspace/studios/this_studio/Final_pipeline/sampled_by_score/sample_07.parquet"
    remain_path = "/teamspace/studios/this_studio/Final_pipeline/remaining/remaining_06.parquet"
    sampled_df, remaining_df = sampling_from_scores(inferenced_file=inferenced_filepath, total=600,
                                                    alpha=0.6, beta=0.0, output_path=out_path, path=remain_path)
