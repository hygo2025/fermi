import argparse
from pathlib import Path

import pandas as pd
import wandb


def main():
    parser = argparse.ArgumentParser(description="Generate W&B results tables and plots")
    parser.add_argument("--project", default="hygo2025-ufes/fermi", help="W&B project")
    parser.add_argument("--group", default="run_final", help="W&B run group")
    parser.add_argument("--out-dir", default="outputs/results", help="Output directory")
    parser.add_argument(
        "--primary-metric",
        default="test_MRR@10",
        help="Primary metric for ranking",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(args.project, filters={"group": args.group})

    rows = []
    for run in runs:
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}

        row = {
            "run_name": run.name,
            "run_id": run.id,
            "group": run.group,
            "state": run.state,
            "model": config.get("model", None),
            "dataset": config.get("dataset", None),
            "is_eval": str(run.name).startswith("Eval_"),
        }

        # Keep only test_* metrics
        for k, v in summary.items():
            if isinstance(k, str) and k.startswith("test_"):
                row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "runs_raw.csv", index=False)

    print(f"[INFO] Runs fetched: {len(df)}")
    if not df.empty:
        test_cols = [c for c in df.columns if c.startswith("test_")]
        print(f"[INFO] test_* columns in raw runs: {sorted(test_cols)}")

    if df.empty:
        print("[INFO] No runs found for the specified group.")
        return

    # Merge eval runs with training runs by model+dataset.
    metric_cols = [c for c in df.columns if c.startswith("test_")]
    merged_rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"], dropna=False):
        eval_g = g[g["is_eval"]]
        train_g = g[~g["is_eval"]]

        # Prefer eval metrics if available; fallback to training runs.
        source_g = eval_g if not eval_g.empty else train_g
        if source_g.empty:
            continue

        # Use latest run by name sort as a simple heuristic.
        source_g = source_g.sort_values("run_name")
        best_row = source_g.iloc[-1].to_dict()

        # If eval exists, copy eval metrics into base row.
        if not eval_g.empty:
            eval_latest = eval_g.sort_values("run_name").iloc[-1]
            for c in metric_cols:
                if c in eval_latest and pd.notna(eval_latest[c]):
                    best_row[c] = eval_latest[c]
            best_row["eval_run_name"] = eval_latest["run_name"]
            best_row["eval_run_id"] = eval_latest["run_id"]
        else:
            best_row["eval_run_name"] = None
            best_row["eval_run_id"] = None

        # Track latest training run too
        if not train_g.empty:
            train_latest = train_g.sort_values("run_name").iloc[-1]
            best_row["train_run_name"] = train_latest["run_name"]
            best_row["train_run_id"] = train_latest["run_id"]
        else:
            best_row["train_run_name"] = None
            best_row["train_run_id"] = None

        merged_rows.append(best_row)

    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(out_dir / "results.csv", index=False)

    print(f"[INFO] Merged rows: {len(merged_df)}")
    if not merged_df.empty:
        merged_test_cols = [c for c in merged_df.columns if c.startswith("test_")]
        print(f"[INFO] test_* columns after merge: {sorted(merged_test_cols)}")

    # Ranking table
    primary_metric = args.primary_metric
    if primary_metric not in merged_df.columns:
        lower_map = {c.lower(): c for c in merged_df.columns}
        if primary_metric.lower() in lower_map:
            primary_metric = lower_map[primary_metric.lower()]

    print(f"[INFO] Primary metric requested: {args.primary_metric} -> resolved: {primary_metric}")
    if primary_metric in merged_df.columns:
        rank_df = (
            merged_df[["model", "dataset", "run_name", primary_metric]]
            .sort_values(primary_metric, ascending=False)
            .reset_index(drop=True)
        )
        rank_df.to_csv(out_dir / "ranking.csv", index=False)
    else:
        print(f"[WARN] Primary metric not found: {args.primary_metric}")

    # Heatmap: model x metric
    if metric_cols:
        import matplotlib.pyplot as plt
        import seaborn as sns

        heat_df = merged_df.groupby("model")[metric_cols].mean().sort_index()
        plt.figure(figsize=(1 + 1.2 * len(metric_cols), 0.6 * max(3, len(heat_df))))
        sns.heatmap(heat_df, annot=True, fmt=".4f", cmap="viridis")
        plt.title(f"Metrics Heatmap ({args.group})")
        plt.tight_layout()
        plt.savefig(out_dir / "heatmap.png", dpi=200)
        plt.close()

        # Boxplot: distribution by metric
        melt_df = merged_df.melt(
            id_vars=["model", "dataset", "run_name"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        ).dropna()
        if not melt_df.empty:
            plt.figure(figsize=(1 + 1.2 * len(metric_cols), 6))
            sns.boxplot(data=melt_df, x="metric", y="value")
            plt.xticks(rotation=30, ha="right")
            plt.title(f"Metric Distributions ({args.group})")
            plt.tight_layout()
            plt.savefig(out_dir / "boxplot_metrics.png", dpi=200)
            plt.close()

        # Ranking by topk
        topk_rows = []
        for col in metric_cols:
            if "@" in col:
                name, topk = col.replace("test_", "", 1).split("@", 1)
                topk_rows.append((col, name, int(topk)))
        if topk_rows:
            rank_rows = []
            for col, metric_name, topk in topk_rows:
                ranked = (
                    merged_df.groupby("model")[col]
                    .mean()
                    .sort_values(ascending=False)
                    .reset_index()
                )
                for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                    rank_rows.append(
                        {
                            "metric": metric_name,
                            "topk": topk,
                            "rank": rank,
                            "model": row["model"],
                            "value": row[col],
                        }
                    )
            pd.DataFrame(rank_rows).to_csv(out_dir / "ranking_by_topk.csv", index=False)


if __name__ == "__main__":
    main()
