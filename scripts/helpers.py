import pandas as pd
import numpy as np

METRICS_MAIN = ["mrr@10", "ndcg@10", "recall@10"]
K_ORDER = ["03-04", "04-05", "05-06", "06-07", "all"]

MODEL_ORDER = None
EVAL_ORDER = ["full", "uni100"]

def _fmt_float(x, digits=4):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"

def _latex_escape(s: str) -> str:
    return (s.replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&"))

def _to_latex_table(df_tab, caption, label, resize=True, column_format="lrrr"):
    latex = df_tab.to_latex(index=False, escape=False, column_format=column_format)

    if resize:
        return (
            "\\begin{table*}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"{latex}"
            "}\n\\end{table*}\n"
        )
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        "\\end{table}\n"
    )

def bold_best_per_column_numeric(df_tab, metric_cols, digits=4):
    out = df_tab.copy()
    for m in metric_cols:
        maxv = out[m].max()
        out[m] = out[m].apply(lambda v: f"\\textbf{{{v:.{digits}f}}}" if v == maxv else f"{v:.{digits}f}")
    return out

def make_main_mean_table(df, eval_mode="full", metrics=METRICS_MAIN):
    d = df[df["eval_mode"] == eval_mode].copy()
    agg = d.groupby("model")[metrics].mean().reset_index()
    agg = agg.sort_values("mrr@10", ascending=False)
    agg_fmt = bold_best_per_column_numeric(agg, metrics)

    agg_fmt["model"] = agg_fmt["model"].astype(str).str.replace("_", r"\_", regex=False)

    agg_fmt = agg_fmt.rename(columns={
        "model": "Modelo",
        "mrr@10": "MRR@10",
        "ndcg@10": "NDCG@10",
        "recall@10": "Recall@10"
    })

    return _to_latex_table(
        agg_fmt,
        caption=f"Resultados agregados (média) por modelo em modo {eval_mode}.",
        label=f"tab:results_mean_{eval_mode}",
        resize=True,
        column_format="lrrr"
    )


def add_rank(df_tab, metric_cols, higher_is_better=True):
    ranks = []
    for m in metric_cols:
        ranks.append(df_tab[m].rank(ascending=not higher_is_better, method="min"))
    df_tab["rank_sum"] = pd.concat(ranks, axis=1).sum(axis=1)
    return df_tab

def bold_best_per_column(df_tab: pd.DataFrame, metric_cols):
    out = df_tab.copy()
    for m in metric_cols:
        maxv = out[m].max()
        out[m] = out[m].apply(lambda v: f"\\textbf{{{_fmt_float(v)}}}" if v == maxv else _fmt_float(v))
    return out


def make_main_agg_tables(df: pd.DataFrame, metrics=METRICS_MAIN, agg="mean"):
    tables = {}
    for em in EVAL_ORDER:
        d = df[df["eval_mode"] == em].copy()
        if d.empty:
            continue

        if "month" in d.columns:
            d["month"] = pd.Categorical(d["month"], categories=K_ORDER, ordered=True)

        agg_df = (
            d.groupby("model")[metrics]
             .agg(agg)
             .reset_index()
        )

        if MODEL_ORDER:
            agg_df["model"] = pd.Categorical(agg_df["model"], categories=MODEL_ORDER, ordered=True)
            agg_df = agg_df.sort_values("model")
        else:
            agg_df = agg_df.sort_values(metrics[0], ascending=False)

        agg_df_num = agg_df.copy()
        for c in metrics:
            agg_df_num[c] = agg_df_num[c].astype(float)

        agg_df_fmt = bold_best_per_column(agg_df_num, metrics)
        agg_df_fmt["model"] = agg_df_fmt["model"].astype(str).map(_latex_escape)

        caption = f"Resultados agregados ({agg}) por modelo em modo {em}."
        label = f"tab:results_{agg}_{em}"
        tables[em] = _to_latex_table(agg_df_fmt, caption, label)

    return tables

def make_monthly_tables(df: pd.DataFrame, month_col="month", metrics=METRICS_MAIN, topn=10):
    tables = []
    months = sorted(df[month_col].unique().tolist())
    if K_ORDER:
        months = [m for m in K_ORDER if m in months] + [m for m in months if m not in K_ORDER]

    for em in EVAL_ORDER:
        d_em = df[df["eval_mode"] == em].copy()
        if d_em.empty:
            continue

        for mth in months:
            d = d_em[d_em[month_col] == mth].copy()
            if d.empty:
                continue

            tab = d[["model"] + metrics].copy()
            tab = tab.sort_values(metrics[0], ascending=False).head(topn)

            tab_num = tab.copy()
            for c in metrics:
                tab_num[c] = tab_num[c].astype(float)

            tab_fmt = bold_best_per_column(tab_num, metrics)
            tab_fmt["model"] = tab_fmt["model"].astype(str).map(_latex_escape)

            caption = f"Resultados por modelo no período {mth} (modo {em})."
            label = f"tab:results_{mth}_{em}"
            tables.append(_to_latex_table(tab_fmt, caption, label))

    return tables

def make_stability_tables(df: pd.DataFrame, metrics=["mrr@10"], month_col="month"):
    tables = {}
    for em in EVAL_ORDER:
        d = df[df["eval_mode"] == em].copy()
        if d.empty:
            continue

        stat = (
            d.groupby("model")[metrics]
             .agg(["mean", "std"])
        )

        stat.columns = [f"{m}_{s}" for m, s in stat.columns]
        stat = stat.reset_index()

        out = stat[["model"]].copy()
        for m in metrics:
            out[m] = stat.apply(lambda r: f"{r[f'{m}_mean']:.4f} $\\pm$ {r[f'{m}_std']:.4f}", axis=1)

        out["_sort"] = stat[f"{metrics[0]}_mean"]
        out = out.sort_values("_sort", ascending=False).drop(columns="_sort")

        out["model"] = out["model"].astype(str).map(_latex_escape)

        caption = f"Estabilidade entre meses (média $\\pm$ desvio-padrão) em modo {em}."
        label = f"tab:stability_{em}"
        tables[em] = _to_latex_table(out, caption, label)

    return tables


def make_winners_table(df: pd.DataFrame, metrics=METRICS_MAIN, month_col="month"):
    tables = {}
    months = sorted(df[month_col].unique().tolist())
    if K_ORDER:
        months = [m for m in K_ORDER if m in months] + [m for m in months if m not in K_ORDER]

    for em in EVAL_ORDER:
        d_em = df[df["eval_mode"] == em].copy()
        if d_em.empty:
            continue

        rows = []
        for mth in months:
            d = d_em[d_em[month_col] == mth]
            if d.empty:
                continue
            row = {"month": mth}
            for met in metrics:
                best = d.loc[d[met].idxmax()]
                row[met] = f"{_latex_escape(str(best['model']))} ({best[met]:.4f})"
            rows.append(row)

        out = pd.DataFrame(rows)
        caption = f"Melhor modelo por métrica em cada período (modo {em})."
        label = f"tab:winners_{em}"
        tables[em] = _to_latex_table(out, caption, label)

    return tables

def temporal_stability(df, metric="mrr@10", eval_mode="full"):
    d = df[df["eval_mode"] == eval_mode]

    g = (
        d.groupby("model")[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    g["cv"] = g["std"] / g["mean"]

    g = g.sort_values("cv")

    return g
