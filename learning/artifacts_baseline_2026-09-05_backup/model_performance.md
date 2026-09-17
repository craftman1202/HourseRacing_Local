# モデル性能レポート — 地方競馬 予測モデル

- 実施日: 2026-08-28
- データ: NAR 月次ファイル（1998-01〜2026-08、実データ）
- 規模: レース 486,832 / 出走 4,832,117 件（1998-01-01 〜 2026-08-29）
- `dataset_version`: `4e885e066e1af30a0d20f4a8…`
- 学習モデル: 条件付きロジット（正則化付き McFadden）, LightGBM LambdaRank, TabM（パラメータ効率的アンサンブル）, 階層ベイズ動的 Plackett-Luce
- embargo: 180 日（`conf/features.yaml` の `max_lookback_days` から自動導出）

> **読む前に**: 本レポートの数値は NAR の月次ファイル由来の**実データ**に
> 対するものです。検証区間は walk-forward の out-of-fold であり、
> OOS（ロック区間）とは別です。OOS の数値は §7 にあり、開封回数も
> そこに記録しています。

## 1. walk-forward 交差検証（5-fold）

主指標はレース内 NLL。後段の期待値計算が確率の較正精度に直接依存するため、
Top-1 精度や NDCG ではなく NLL で優劣を判断します。

| モデル | race_nll | brier | top1 | top3 | ndcg@3 | spearman | ece |
|---|---|---|---|---|---|---|---|
| LightGBM LambdaRank | 1.805 | 0.7715 | 0.3599 | 0.5465 | 0.5688 | 0.4568 | 0.0033 |
| TabM（パラメータ効率的アンサンブル） | 1.825 | 0.7773 | 0.3528 | 0.5391 | 0.561 | 0.4381 | 0.0023 |
| 条件付きロジット（正則化付き McFadden） | 1.868 | 0.7898 | 0.3388 | 0.5309 | 0.5497 | 0.4206 | 0.0043 |
| ベースライン: レース内一様分布 | 2.288 | 0.8961 | 0.0948 | 0.3029 | 0.2676 |  | 0 |
| 階層ベイズ動的 Plackett-Luce | 2.304 | 0.8997 | 0.1117 | 0.3216 | 0.29 | 0.023 | 0.0104 |

### fold 別のレース内 NLL

| モデル | 平均 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| LightGBM LambdaRank | 1.805 | 1.799 | 1.79 | 1.811 | 1.824 | 1.802 |
| TabM（パラメータ効率的アンサンブル） | 1.825 | 1.82 | 1.812 | 1.829 | 1.839 | 1.825 |
| 条件付きロジット（正則化付き McFadden） | 1.868 | 1.861 | 1.857 | 1.873 | 1.885 | 1.866 |
| ベースライン: レース内一様分布 | 2.288 | 2.293 | 2.293 | 2.297 | 2.273 | 2.281 |
| 階層ベイズ動的 Plackett-Luce | 2.304 | 2.312 | 2.309 | 2.311 | 2.291 | 2.3 |

fold 間の標準偏差が最も大きいのは **LightGBM LambdaRank**（0.0129）。分割ごとの当たり外れが大きいモデルは、点推定の平均だけで採否を決めない。

## 2. アンサンブル（OOF のみから重み推定）

重みは walk-forward の out-of-fold 予測のみから推定しています。訓練内予測を
混ぜると重みが致命的にバイアスするため、fold 割り当ての無い行が混入したら
例外を送出する実装にしてあります（EN-02）。

| モデル | race_nll | uniform_nll | brier | top1 | top3 | ndcg@3 | spearman | ece |
|---|---|---|---|---|---|---|---|---|
| アンサンブル（制約付きスタッキング） | 1.805 | 2.287 | 0.7714 | 0.3599 | 0.5463 | 0.5686 | 0.4556 | 0.0031 |
| アンサンブル（単純平均） | 1.864 | 2.287 | 0.7836 | 0.3545 | 0.5407 | 0.5629 | 0.4359 | 0.0199 |
| アンサンブル（対数平均） | 1.849 | 2.287 | 0.7832 | 0.3541 | 0.5414 | 0.5632 | 0.4432 | 0.0179 |

| モデル | weight |
|---|---|
| 条件付きロジット（正則化付き McFadden） | 0 |
| LightGBM LambdaRank | 0.86 |
| TabM（パラメータ効率的アンサンブル） | 0.14 |
| 階層ベイズ動的 Plackett-Luce | 0 |

重みは **LightGBM LambdaRank** に 86.0% 集中しています。

## 3. 特徴量選択（fold 内実行）

Null Importance → 相関/VIF → RFE の3段を、**各 fold の学習期間内でのみ**
実行しています。全期間で選んでから CV すると選択バイアスが入ります（CV-10）。

| fold | n_train | n_valid | n_selected | selected |
|---|---|---|---|---|
| 1 | 3208012 | 123013 | 19 | h_top3rate_prior, h_si_last3, j_winrate_shrunk, h_winrate_prior, d_pair_winrate, d_all_starts, d_track_winrate, d_dist_winrate, t_winrate_shrunk, h_days_since_prev, class_level, d_best_speed, s_winrate_shrunk, d_all_winrate, d_pair_starts, d_best_speed_good, d_turn_winrate, d_extra_starts, h_best_si_prior |
| 2 | 3338686 | 123834 | 20 | h_top3rate_prior, h_si_last3, j_winrate_shrunk, h_winrate_prior, d_pair_winrate, d_all_starts, d_track_winrate, d_dist_winrate, t_winrate_shrunk, d_track_starts, class_level, h_days_since_prev, d_best_speed, d_dist_starts, s_winrate_shrunk, d_pair_starts, d_turn_starts, d_best_speed_good, d_all_winrate, d_turn_winrate |
| 3 | 3471173 | 120417 | 17 | h_top3rate_prior, h_si_last3, j_winrate_shrunk, h_winrate_prior, d_pair_winrate, d_all_starts, d_track_winrate, d_dist_winrate, d_track_starts, t_winrate_shrunk, h_days_since_prev, d_best_speed, d_dist_starts, s_winrate_shrunk, d_all_winrate, d_turn_starts, d_extra_starts |
| 4 | 3601523 | 124477 | 18 | h_top3rate_prior, h_si_last3, j_winrate_shrunk, h_winrate_prior, d_pair_winrate, d_all_starts, d_track_winrate, d_track_starts, t_winrate_shrunk, d_dist_winrate, class_level, h_days_since_prev, d_best_speed, d_dist_starts, s_winrate_shrunk, d_pair_starts, d_best_speed_good, distance |
| 5 | 3734831 | 126944 | 21 | h_top3rate_prior, h_si_last3, j_winrate_shrunk, h_winrate_prior, d_pair_winrate, d_all_starts, d_track_winrate, d_track_starts, t_winrate_shrunk, d_dist_winrate, class_level, h_starts_prior, h_days_since_prev, d_best_speed, s_winrate_shrunk, d_pair_starts, d_best_speed_good, d_all_winrate, d_extra_starts, h_best_si_prior, d_turn_winrate |

fold ごとに選択結果が異なっており、fold 内実行が効いていることが確認できます。

## 4. ベイズモデルの収束診断

収束診断を通らないモデルは採用しない、という方針で運用します。

| 指標 | 値 |
|---|---|
| method | svi |
| n_train_rows | 403971 |
| svi_final_loss | 2.392e+05 |
| svi_loss_improved | True |

- 推論方式: **SVI**（変分近似）。
- **R-hat / ESS / 発散遷移（MD-16〜18）は MCMC の診断なので、この実行では
  評価していません。** 設計書 §8.4 の二段構えのうち SVI 側だけを回しており、
  最終窓の NUTS 検証は未実施です。SVI で確認できるのは ELBO が改善したこと
  （True）だけで、収束の保証にはなりません。
- NUTS を回した実測値はテスト（`test_models_deep.py`）に記録してあります:
  250レース規模で R-hat 1.035 / ESS(bulk) 119 / 発散遷移 0 件。
  仕様値（R-hat < 1.01、ESS > 400）にはサンプル数が届いていません。

## 5. OOS 最終評価

OOS 期間: 2024-02-01 〜 データ末尾。開封回数 **2 回**（`artifacts/oos_access.log` に記録）。

| モデル | race_nll | uniform_nll | brier | top1 | top3 | ndcg@3 | spearman | ece |
|---|---|---|---|---|---|---|---|---|
| ensemble | 1.823 | 2.303 | 0.7764 | 0.353 | 0.541 | 0.5626 | 0.445 | 0.0037 |
| LightGBM LambdaRank | 1.825 | 2.303 | 0.7769 | 0.3524 | 0.5412 | 0.5624 | 0.4455 | 0.0039 |
| TabM（パラメータ効率的アンサンブル） | 1.827 | 2.303 | 0.7772 | 0.3491 | 0.5378 | 0.5594 | 0.4358 | 0.0033 |
| 条件付きロジット（正則化付き McFadden） | 1.893 | 2.303 | 0.7953 | 0.3312 | 0.5256 | 0.5431 | 0.4083 | 0.0067 |
| ベースライン: レース内一様分布 | 2.303 | 2.303 | 0.8978 | 0.0893 | 0.2948 | 0.2585 |  | 0 |

OOS 最良は **ensemble**（NLL 1.8230）。
CV 最良平均 1.8051 との差は +0.0179。CV と OOS が整合しており、分割リークの兆候はありません。

## 6. Too-Good-To-Be-True ガード

競馬の市場効率性と控除率の水準を踏まえると、以下の数値は成功ではなくバグの
徴候です。発火したら成果とみなさず、即座にリーク調査へ回します。

| id | fired | severity | observed | message |
|---|---|---|---|---|
| RF-01 | 未発火 | Blocker | 0.353 | OOS Top-1 が 60% 超。着順情報の混入がほぼ確実です。 |
| RF-02 | 未発火 | Blocker | 1.823 | OOS レース内 NLL が 1.20 未満。着順情報の混入がほぼ確実です。 |
| RF-05 | 未発火 | Critical | 0.0179 | CV スコアが OOS より 0.10 以上良い。CV 分割のリークか過学習です。 |

発火 **0 件**。リークの兆候はありません。

## 7. トラックB（オッズ使用）

_トラックB は未実行です。_

## 8. 実行時間

| fold | selection | clogit | lgbm | tabm | bayes |
|---|---|---|---|---|---|
| 1 | 279.4 | 8.6 | 31.1 | 395.6 | 378.3 |
| 2 | 423.6 | 11 | 50.2 | 430.2 | 380.9 |
| 3 | 423 | 9.8 | 49.7 | 441.9 | 374.3 |
| 4 | 416.1 | 9.5 | 52.1 | 461.8 | 347.4 |
| 5 | 514.1 | 14.4 | 58 | 447.7 | 366.4 |

## 9. 結論

- CV 平均のレース内 NLL で最良の単体モデルは **lgbm**（1.8051）。
- 一様分布ベースライン（2.2875）は全モデルが上回っており、学習が機能している。
- アンサンブルは **ensemble_stacked** が最良（1.8046）。
- Too-Good-To-Be-True ガードの発火は 0 件。リークの兆候なし。

## 10. この結果の限界

- **数値は合成データに対するもの**であり、実データの性能ではない。合成市場は真の効用に直接アクセスしているため、市場ベースラインとの優劣は構造上ほぼ意味を持たない。
- HPO の試行回数は設計書の予算（LightGBM 500 / TabM 300 / 条件付きロジット 100）を大きく下回る。CPU のみで nested HPO を回しているため。DL 側に十分な予算を与えないとGBDT 有利のバイアスがかかる（TabArena の知見）ので、モデル間比較はこの点を割り引くこと。
- ベイズは全期間ではなく直近窓のみを推論対象にしている（設計書 §8.4 の二段構え）。
- SBC（MD-20）は未実行。事前生成＋推論を100回繰り返す計算量が CPU では現実的でないため。
- 温度スケーリングを各 fold の valid で最適化し、同じ valid で評価している。較正の効果はその分だけ楽観的に出る。
- 実データの累積成績8列の時点性判定（LK-01）は、当日ファイルと後日の月次ファイルが揃うまで実行できない。現状はホワイトリスト空＝8列すべて破棄で運用している。
