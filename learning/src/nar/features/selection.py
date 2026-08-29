"""特徴量選択（3段構成）。

第1段 Null Importance、第2段 相関・VIF、第3段 RFE。

決定的に重要なのは、**この3段すべてを各 fold の学習期間内でのみ実行する**こと。
全期間で特徴量を選んでから CV すると選択バイアスが入る。計算量は跳ね上がるが、
正しさ優先の方針に従う。API 上も fold の学習データしか受け取らない形にしてある。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    stage1_survivors: list[str]
    stage2_survivors: list[str]
    selected: list[str]
    null_importance: pd.DataFrame
    dropped_collinear: list[str]
    rfe_path: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "n_stage1": len(self.stage1_survivors),
            "n_stage2": len(self.stage2_survivors),
            "n_selected": len(self.selected),
            "selected": ", ".join(self.selected),
        }])


def shuffle_labels_within_race(df: pd.DataFrame, rng: np.random.Generator,
                               race_col: str = "race_id",
                               pos_col: str = "finish_pos") -> pd.Series:
    """レース内で着順をシャッフルした帰無ラベル。

    行全体をシャッフルすると頭数分布まで壊れるので、必ずレース内に閉じる。
    """
    out = df[pos_col].copy()
    for _, idx in df.groupby(race_col, sort=False).indices.items():
        vals = out.to_numpy()[idx].copy()
        rng.shuffle(vals)
        out.iloc[idx] = vals
    return out


def null_importance(
    train: pd.DataFrame, feature_cols: list[str], n_runs: int = 30,
    percentile: float = 95.0, seed: int = 0, num_boost_round: int = 80,
) -> pd.DataFrame:
    """第1段。

    ラベルをレース内シャッフルして n_runs 回学習し、実 importance が帰無分布の
    95 パーセンタイルを超えない特徴量を落とす。
    """
    from ..models.lgbm import LgbmRanker

    d = train.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    actual = LgbmRanker(num_boost_round=num_boost_round).fit(d, feature_cols).importance()

    rng = np.random.default_rng(seed)
    null_runs = []
    for i in range(n_runs):
        shuffled = d.copy()
        shuffled["finish_pos"] = shuffle_labels_within_race(d, rng).to_numpy()
        model = LgbmRanker({"seed": seed + i}, num_boost_round=num_boost_round)
        null_runs.append(model.fit(shuffled, feature_cols).importance())

    null_df = pd.DataFrame(null_runs)
    threshold = null_df.quantile(percentile / 100.0)
    out = pd.DataFrame({
        "feature": feature_cols,
        "actual_gain": [float(actual.get(c, 0.0)) for c in feature_cols],
        "null_p95": [float(threshold.get(c, 0.0)) for c in feature_cols],
        "null_mean": [float(null_df[c].mean()) if c in null_df else 0.0 for c in feature_cols],
    })
    out["survives"] = out["actual_gain"] > out["null_p95"]
    return out.sort_values("actual_gain", ascending=False).reset_index(drop=True)


def variance_inflation(train: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """VIF。GBDT には不要なので、ロジット系にのみ適用する。"""
    x = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median()).to_numpy(float)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    corr = np.corrcoef(x, rowvar=False)
    try:
        inv = np.linalg.pinv(corr)
        vif = np.diag(inv)
    except np.linalg.LinAlgError:
        vif = np.full(len(feature_cols), np.nan)
    return pd.Series(vif, index=feature_cols).sort_values(ascending=False)


def drop_collinear(train: pd.DataFrame, feature_cols: list[str],
                   corr_threshold: float = 0.95, vif_threshold: float = 10.0
                   ) -> tuple[list[str], list[str]]:
    """第2段。相関が極端に高いペアの片方と、VIF が高い列を落とす。"""
    x = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman").abs()
    dropped: list[str] = []
    keep = list(feature_cols)
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1:]:
            if a in dropped or b in dropped:
                continue
            if corr.loc[a, b] >= corr_threshold:
                # 欠損が多いほうを落とす。情報量が同じなら埋まっている列を残す
                loser = b if x[b].isna().mean() >= x[a].isna().mean() else a
                dropped.append(loser)
    keep = [c for c in keep if c not in dropped]

    if len(keep) > 2:
        vif = variance_inflation(train, keep)
        high = vif[vif > vif_threshold].index.tolist()
        # VIF は一度に全部落とすと落としすぎる。最悪の1本だけ落として再計算する
        while high and len(keep) > 2:
            worst = vif.idxmax()
            keep.remove(worst)
            dropped.append(worst)
            vif = variance_inflation(train, keep)
            high = vif[vif > vif_threshold].index.tolist()
    return keep, dropped


def recursive_elimination(
    train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str],
    min_features: int = 5, seed: int = 0,
) -> tuple[list[str], pd.DataFrame]:
    """第3段。CV スコア（レース内 NLL）を見ながら弱い順に落とす。

    train/valid は**外側 fold の学習期間を内部分割したもの**であり、
    外側の valid は絶対に渡さない。
    """
    from ..eval.metrics import race_nll
    from ..models.lgbm import LgbmRanker

    cols = list(feature_cols)
    rows = []
    best_cols, best_score = cols, float("inf")

    while len(cols) >= min_features:
        model = LgbmRanker({"seed": seed}, num_boost_round=120).fit(train, cols)
        p = model.predict_proba(valid)
        score = race_nll(p, valid["is_win"].to_numpy(), valid["race_id"].to_numpy())
        rows.append({"n_features": len(cols), "nll": score, "features": ", ".join(cols)})
        if score < best_score - 1e-6:
            best_score, best_cols = score, list(cols)
        imp = model.importance()
        weakest = imp.index[-1] if len(imp) else cols[-1]
        cols = [c for c in cols if c != weakest]

    return best_cols, pd.DataFrame(rows)


def select(
    train: pd.DataFrame, feature_cols: list[str], seed: int = 0,
    n_null_runs: int = 20, apply_vif: bool = False, run_rfe: bool = True,
    inner_split: float = 0.75,
) -> SelectionResult:
    """3段を fold の学習データだけで実行する。

    apply_vif はロジット系のときだけ True にする（GBDT には不要）。
    """
    d = train.sort_values(["race_id", "horse_no"]).reset_index(drop=True)

    ni = null_importance(d, feature_cols, n_runs=n_null_runs, seed=seed)
    stage1 = ni.loc[ni["survives"], "feature"].tolist() or list(feature_cols)

    if apply_vif:
        stage2, dropped = drop_collinear(d, stage1)
    else:
        stage2, dropped = stage1, []

    if run_rfe and len(stage2) > 5:
        races = d["race_id"].drop_duplicates()
        cut = races.iloc[int(len(races) * inner_split)]
        inner_tr = d[d["race_id"] < cut]
        inner_va = d[d["race_id"] >= cut]
        if inner_tr["race_id"].nunique() > 30 and inner_va["race_id"].nunique() > 30:
            selected, path = recursive_elimination(inner_tr, inner_va, stage2, seed=seed)
        else:
            selected, path = stage2, pd.DataFrame()
    else:
        selected, path = stage2, pd.DataFrame()

    return SelectionResult(stage1, stage2, selected, ni, dropped, path)
