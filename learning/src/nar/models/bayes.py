"""階層ベイズ動的 Plackett-Luce。

各馬の潜在能力 θ_{h,t} を状態空間モデル（ランダムウォーク）で時変にし、
レース結果を Plackett-Luce 尤度で観測する。騎手・調教師・種牡馬効果は階層事前分布で
部分プーリングし、出走数の少ない個体を全体平均へ自動収縮させる。

    θ_{h,t} = θ_{h,t-1} + ε,  ε ~ N(0, σ_h²)
    μ_i = θ_{h(i),t} + γ_{j(i)} + δ_{s(i)} + z_i'β

全期間フル NUTS は現実的でないので、直近窓を NUTS、全期間を SVI で近似する二段構え。
「計算量を気にしない」方針でも、収束診断（R-hat < 1.01、ESS > 400、発散遷移ゼロ）を
通らないモデルは採用しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# jax/numpyro は重いので、モジュール読み込み時ではなく使用時に import する
_JAX: Any = None


def _lazy_import():
    global _JAX
    if _JAX is None:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

        _JAX = dict(jax=jax, jnp=jnp, numpyro=numpyro, dist=dist, MCMC=MCMC, NUTS=NUTS,
                    SVI=SVI, Trace_ELBO=Trace_ELBO, autoguide=autoguide)
    return _JAX


NEG_INF = -1e30


@dataclass
class RaceArrays:
    """レース単位にパディングした観測。

    order[r, k] は r 番目のレースで k 着だった馬の、レース内インデックス。
    Plackett-Luce はこの順序に沿って「残り集合からの選択」を繰り返す。
    """

    horse: np.ndarray       # (n_races, max_n) 馬インデックス
    jockey: np.ndarray
    sire: np.ndarray
    seq: np.ndarray         # (n_races, max_n) その馬の何走目か（0始まり）
    z: np.ndarray           # (n_races, max_n, n_cov) 共変量
    mask: np.ndarray        # (n_races, max_n)
    order: np.ndarray       # (n_races, max_n) 着順→レース内 index
    n_ranked: np.ndarray    # (n_races,) 尤度に使う着順の数
    n_horses: int
    n_jockeys: int
    n_sires: int
    max_starts: int
    race_ids: np.ndarray
    row_index: np.ndarray
    covariates: tuple[str, ...]


def build_arrays(df: pd.DataFrame, covariates: list[str], max_positions: int = 3) -> RaceArrays:
    """silver/gold の行を Plackett-Luce 用の配列に詰め直す。

    max_positions で尤度に使う着順を打ち切る。地方の下位着順はノイズが大きく、
    全順位を使うと計算量だけ増えて事後分布がほとんど動かない。
    """
    d = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    d["_row"] = np.arange(len(d))
    h_codes, _ = pd.factorize(d["horse_sk"])
    j_codes, _ = pd.factorize(d["jockey_sk"])
    s_codes, _ = pd.factorize(d["sire_sk"])
    d = d.assign(_h=h_codes, _j=j_codes, _s=s_codes)
    d["_seq"] = d.sort_values(["_h", "start_ts", "race_id", "horse_no"]).groupby("_h").cumcount()

    groups = list(d.groupby("race_id", sort=False).indices.items())
    n_races = len(groups)
    max_n = max(len(i) for _, i in groups)
    n_cov = len(covariates)

    horse = np.zeros((n_races, max_n), int)
    jockey = np.zeros((n_races, max_n), int)
    sire = np.zeros((n_races, max_n), int)
    seq = np.zeros((n_races, max_n), int)
    z = np.zeros((n_races, max_n, n_cov), float)
    mask = np.zeros((n_races, max_n), bool)
    order = np.zeros((n_races, max_n), int)
    n_ranked = np.zeros(n_races, int)
    row_index = np.full((n_races, max_n), -1, int)
    race_ids = np.empty(n_races, object)

    zv = d[covariates].to_numpy(float) if n_cov else np.zeros((len(d), 0))
    pos = d["finish_pos"].to_numpy()
    for r, (rid, idx) in enumerate(groups):
        n = len(idx)
        horse[r, :n] = d["_h"].to_numpy()[idx]
        jockey[r, :n] = d["_j"].to_numpy()[idx]
        sire[r, :n] = d["_s"].to_numpy()[idx]
        seq[r, :n] = d["_seq"].to_numpy()[idx]
        if n_cov:
            z[r, :n] = zv[idx]
        mask[r, :n] = True
        row_index[r, :n] = idx
        race_ids[r] = rid
        rank = np.argsort(pos[idx], kind="stable")
        order[r, :n] = rank
        n_ranked[r] = min(max_positions, n - 1)

    return RaceArrays(
        horse, jockey, sire, seq, z, mask, order, n_ranked,
        int(h_codes.max()) + 1, int(j_codes.max()) + 1, int(s_codes.max()) + 1,
        int(d["_seq"].max()) + 1, race_ids, row_index, tuple(covariates),
    )


def restrict_likelihood_to(arr: RaceArrays, observed_race_ids) -> RaceArrays:
    """尤度に使うレースを限定した配列を返す。

    予測対象（valid）のレースまで尤度に入れると、モデルが答えを見たまま
    その答えを予測することになる。一方でインデックス空間は train と valid で
    共有していないと、valid の馬・騎手の事後を引けない。そこで配列は結合したまま
    作り、**尤度に効くレースだけを絞る**（n_ranked=0 の行は寄与ゼロ）。
    """
    keep = np.isin(arr.race_ids, np.asarray(list(observed_race_ids), dtype=object))
    n_ranked = np.where(keep, arr.n_ranked, 0)
    return RaceArrays(
        arr.horse, arr.jockey, arr.sire, arr.seq, arr.z, arr.mask, arr.order, n_ranked,
        arr.n_horses, arr.n_jockeys, arr.n_sires, arr.max_starts,
        arr.race_ids, arr.row_index, arr.covariates,
    )


def _model(arr: RaceArrays, observed: bool = True):
    """NumPyro モデル。

    パラメータ化について2点、収束のために必要な工夫がある。

    1. **非中心化**: θ = σ · cumsum(z) と書く。中心化のままだとランダムウォークの
       分散パラメータと θ が強く相関し、NUTS が発散する。

    2. **和ゼロ制約**: レース内 softmax は定数シフトに対して不変なので、
       θ・γ・δ の全体に同じ定数を足しても尤度が変わらない。この非識別性を
       放置すると事後分布に平坦な方向が残り、R-hat が 1.4 前後から下がらない。
       初期能力・騎手効果・種牡馬効果それぞれを中心化して自由度を潰す。
    """
    J = _lazy_import()
    jnp, dist, numpyro = J["jnp"], J["dist"], J["numpyro"]

    sigma_h = numpyro.sample("sigma_h", dist.HalfNormal(0.3))   # 走ごとの能力変動
    sigma_0 = numpyro.sample("sigma_0", dist.HalfNormal(1.0))   # 初期能力のばらつき
    sigma_j = numpyro.sample("sigma_j", dist.HalfNormal(0.5))
    sigma_s = numpyro.sample("sigma_s", dist.HalfNormal(0.5))

    with numpyro.plate("horses", arr.n_horses):
        z_0 = numpyro.sample("z_0", dist.Normal(0, 1))
        z_h = numpyro.sample("z_h", dist.Normal(0, 1).expand([arr.max_starts]).to_event(1))
    theta_0 = sigma_0 * (z_0 - z_0.mean())
    steps = jnp.cumsum(sigma_h * z_h, axis=1) - (sigma_h * z_h[:, :1])
    theta = theta_0[:, None] + steps                   # (n_horses, max_starts)

    with numpyro.plate("jockeys", arr.n_jockeys):
        z_j = numpyro.sample("z_j", dist.Normal(0, 1))
    gamma = numpyro.deterministic("gamma", sigma_j * (z_j - z_j.mean()))

    with numpyro.plate("sires", arr.n_sires):
        z_s = numpyro.sample("z_s", dist.Normal(0, 1))
    delta = numpyro.deterministic("delta", sigma_s * (z_s - z_s.mean()))

    n_cov = arr.z.shape[2]
    if n_cov:
        beta = numpyro.sample("beta", dist.Normal(0, 1).expand([n_cov]).to_event(1))
        cov = jnp.einsum("rnc,c->rn", jnp.asarray(arr.z), beta)
    else:
        cov = 0.0

    mu = (
        theta[jnp.asarray(arr.horse), jnp.asarray(arr.seq)]
        + gamma[jnp.asarray(arr.jockey)]
        + delta[jnp.asarray(arr.sire)]
        + cov
    )
    mu = jnp.where(jnp.asarray(arr.mask), mu, NEG_INF)

    if observed:
        numpyro.factor("pl", plackett_luce_logprob(mu, arr))
    return mu


def plackett_luce_logprob(mu, arr: RaceArrays):
    """Plackett-Luce 対数尤度。

    k 着目の項は「まだ選ばれていない集合の中で k 着馬が選ばれる確率」。
    選択済みを順に -inf でマスクしていくことでベクトル化する。
    """
    J = _lazy_import()
    jnp = J["jnp"]

    mu = jnp.asarray(mu)
    order = jnp.asarray(arr.order)
    n_races, max_n = mu.shape
    max_k = int(arr.n_ranked.max())

    total = jnp.zeros(())
    available = jnp.asarray(arr.mask)
    for k in range(max_k):
        pick = order[:, k]                                    # (n_races,)
        active = jnp.asarray(arr.n_ranked) > k                # このレースで k 着目を使うか
        scores = jnp.where(available, mu, NEG_INF)
        log_denom = jax_logsumexp(scores)
        log_num = jnp.take_along_axis(mu, pick[:, None], axis=1)[:, 0]
        total = total + jnp.sum(jnp.where(active, log_num - log_denom, 0.0))
        available = available.at[jnp.arange(n_races), pick].set(False)
    return total


def jax_logsumexp(x):
    J = _lazy_import()
    jnp = J["jnp"]
    m = jnp.max(x, axis=-1, keepdims=True)
    return (m + jnp.log(jnp.sum(jnp.exp(x - m), axis=-1, keepdims=True)))[..., 0]


@dataclass
class BayesConfig:
    num_warmup: int = 500
    num_samples: int = 500
    num_chains: int = 2
    svi_steps: int = 3000
    svi_lr: float = 5e-3
    max_positions: int = 3
    seed: int = 0
    target_accept: float = 0.9
    diagnostics: dict = field(default_factory=dict)


class HierarchicalPlackettLuce:
    """SVI で全期間を近似し、最終窓を NUTS で検証する二段構え。"""

    def __init__(self, cfg: BayesConfig | None = None) -> None:
        self.cfg = cfg or BayesConfig()
        self.method: str | None = None
        self.posterior: dict[str, np.ndarray] | None = None
        self._arr: RaceArrays | None = None
        self._guide = None

    # ------------------------------------------------------------------ 推論
    def fit_svi(self, arr: RaceArrays) -> "HierarchicalPlackettLuce":
        J = _lazy_import()
        jax, numpyro = J["jax"], J["numpyro"]
        guide = J["autoguide"].AutoNormal(_model)
        svi = J["SVI"](_model, guide, numpyro.optim.Adam(self.cfg.svi_lr), J["Trace_ELBO"]())
        res = svi.run(jax.random.PRNGKey(self.cfg.seed), self.cfg.svi_steps, arr,
                      progress_bar=False)
        self._guide, self._arr, self.method = guide, arr, "svi"
        samples = guide.sample_posterior(
            jax.random.PRNGKey(self.cfg.seed + 1), res.params, sample_shape=(200,))
        self.posterior = {k: np.asarray(v) for k, v in samples.items()}
        self.cfg.diagnostics["svi_final_loss"] = float(res.losses[-1])
        self.cfg.diagnostics["svi_loss_improved"] = bool(res.losses[-1] < res.losses[0])
        return self

    def fit_nuts(self, arr: RaceArrays) -> "HierarchicalPlackettLuce":
        J = _lazy_import()
        jax, numpyro = J["jax"], J["numpyro"]
        numpyro.set_host_device_count(self.cfg.num_chains)
        kernel = J["NUTS"](_model, target_accept_prob=self.cfg.target_accept)
        mcmc = J["MCMC"](kernel, num_warmup=self.cfg.num_warmup,
                         num_samples=self.cfg.num_samples, num_chains=self.cfg.num_chains,
                         progress_bar=False, chain_method="sequential")
        mcmc.run(jax.random.PRNGKey(self.cfg.seed), arr, extra_fields=("diverging",))
        self._arr, self.method = arr, "nuts"
        self.posterior = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        self.cfg.diagnostics.update(diagnose(mcmc))
        return self

    # ------------------------------------------------------------------ 予測
    def predict_proba(self, arr: RaceArrays, n_draws: int = 200) -> np.ndarray:
        """事後平均の予測確率 (n_races, max_n)。

        事後分布から μ を引いてレース内 softmax を取り、draw 平均する。
        点推定ではなく draw 平均にするのは、不確実性が Kelly 配分で実質的な
        価値を持つため（設計書 §8.4）。
        """
        if self.posterior is None:
            raise RuntimeError("fit_svi() か fit_nuts() を先に呼んでください")
        J = _lazy_import()
        jnp = J["jnp"]

        post = self.posterior
        n_available = len(next(iter(post.values())))
        draws = np.linspace(0, n_available - 1, min(n_draws, n_available)).astype(int)

        acc = np.zeros(arr.mask.shape)
        for d in draws:
            theta = _theta_of(post, d)
            # 学習時より長い出走系列は最終状態で外挿する（ランダムウォークの平均は現状維持）
            seq = np.clip(arr.seq, 0, theta.shape[1] - 1)
            horse = np.clip(arr.horse, 0, theta.shape[0] - 1)
            gamma, delta = _effect_of(post, d, "gamma"), _effect_of(post, d, "delta")
            mu = (theta[horse, seq]
                  + gamma[np.clip(arr.jockey, 0, len(gamma) - 1)]
                  + delta[np.clip(arr.sire, 0, len(delta) - 1)])
            if "beta" in post and arr.z.shape[2]:
                mu = mu + arr.z @ post["beta"][d]
            mu = np.where(arr.mask, mu, NEG_INF)
            e = np.exp(mu - mu.max(axis=1, keepdims=True))
            e = np.where(arr.mask, e, 0.0)
            acc += e / e.sum(axis=1, keepdims=True)
        p = acc / len(draws)
        p[~arr.mask] = 0.0
        return p

    def flat_predictions(self, arr: RaceArrays, n_rows: int, n_draws: int = 200) -> np.ndarray:
        p = self.predict_proba(arr, n_draws)
        out = np.full(n_rows, np.nan)
        valid = arr.row_index >= 0
        out[arr.row_index[valid]] = p[valid]
        return out


def _theta_of(post: dict[str, np.ndarray], d: int) -> np.ndarray:
    """事後サンプル1本から θ（n_horses, max_starts）を組み立てる。

    モデル側の非中心パラメータ化と同じ式をここでも使う。式が二重管理になると
    予測だけ静かにズレるので、変更時は _model と必ず揃えること。
    """
    z_h = post["z_h"][d]
    steps = np.cumsum(post["sigma_h"][d] * z_h, axis=1) - post["sigma_h"][d] * z_h[:, :1]
    z0 = post["z_0"][d]
    theta_0 = post["sigma_0"][d] * (z0 - z0.mean())
    return theta_0[:, None] + steps


def _effect_of(post: dict[str, np.ndarray], d: int, name: str) -> np.ndarray:
    """gamma / delta。deterministic として記録されていればそれを、無ければ再構成する。"""
    if name in post:
        return post[name][d]
    z_key, sig_key = {"gamma": ("z_j", "sigma_j"), "delta": ("z_s", "sigma_s")}[name]
    z = post[z_key][d]
    return post[sig_key][d] * (z - z.mean())


def diagnose(mcmc) -> dict:
    """収束診断。これを通らないモデルは採用しない（MD-16/17/18）。"""
    import numpyro.diagnostics as diag

    samples = mcmc.get_samples(group_by_chain=True)
    # 高次元の z_h まで全部見ると R-hat の最悪値が常に悪くなるので、
    # 構造パラメータ（σ, β）と効果パラメータを分けて報告する
    scalar_keys = [k for k in samples
                   if k in ("sigma_h", "sigma_0", "sigma_j", "sigma_s", "beta")]
    rhat, ess_bulk, ess_tail = [], [], []
    for k in scalar_keys:
        s = np.asarray(samples[k])
        rhat.append(np.asarray(diag.split_gelman_rubin(s)).ravel())
        ess_bulk.append(np.asarray(diag.effective_sample_size(s)).ravel())
    extra = mcmc.get_extra_fields()
    n_div = int(np.sum(np.asarray(extra["diverging"]))) if "diverging" in extra else 0
    return {
        "max_rhat": float(np.max(np.concatenate(rhat))) if rhat else float("nan"),
        "min_ess_bulk": float(np.min(np.concatenate(ess_bulk))) if ess_bulk else float("nan"),
        "n_divergent": n_div,
        "params_checked": scalar_keys,
    }


def shrinkage_check(model: HierarchicalPlackettLuce, arr: RaceArrays) -> dict:
    """収縮の方向性（MD-21）。

    符号の向きについて注意がある。テスト仕様は「出走数と事前からの乖離が負相関
    （Spearman < -0.3）」としているが、部分プーリングの定義からは逆になる。
    出走数が少ない個体ほど事前平均（0）へ強く引かれるので、

        出走数 vs |事後平均|      → **正**の相関（データが増えるほど 0 から離れられる）
        出走数 vs 事後標準偏差    → **負**の相関（データが増えるほど不確実性が減る）

    が期待される向き。仕様の符号は後者を指していたものと解釈し、両方を返す。
    どちらが破れても収縮が効いていない。
    """
    from scipy.stats import spearmanr

    post = model.posterior
    draws = np.stack([_effect_of(post, d, "gamma")
                      for d in range(len(post["sigma_j"]))])
    mean_abs = np.abs(draws.mean(axis=0))
    sd = draws.std(axis=0)
    counts = np.bincount(arr.jockey[arr.mask], minlength=len(mean_abs))
    keep = counts > 0
    return {
        "spearman_starts_vs_abs_mean": float(spearmanr(counts[keep], mean_abs[keep]).statistic),
        "spearman_starts_vs_posterior_sd": float(spearmanr(counts[keep], sd[keep]).statistic),
        "n_entities": int(keep.sum()),
    }


def simulation_based_calibration(
    arr: RaceArrays, n_sims: int = 60, n_draws: int = 100, seed: int = 0,
) -> dict:
    """SBC（MD-20）。

    事前分布からデータを生成 → 推論 → 事後分位のランク統計量が一様分布に
    従うかを KS 検定する。実装ミスがあると事後分布が系統的に偏り、
    ランク統計量が一様から外れる。ベイズ実装の正しさを検証する最も厳密な方法。
    """
    J = _lazy_import()
    jax, numpyro = J["jax"], J["numpyro"]
    from numpyro.infer import Predictive
    from scipy.stats import kstest

    rng = jax.random.PRNGKey(seed)
    ranks: dict[str, list[int]] = {"sigma_h": [], "sigma_0": [], "sigma_j": [], "sigma_s": []}

    for i in range(n_sims):
        rng, k1, k2 = jax.random.split(rng, 3)
        # 1. 事前分布からパラメータと観測を生成
        prior = Predictive(_model, num_samples=1)(k1, arr, observed=False)
        truth = {k: np.asarray(prior[k])[0] for k in ranks if k in prior}
        sim = _simulate_orders(arr, prior, k1)

        # 2. 生成データで推論
        model = HierarchicalPlackettLuce(BayesConfig(svi_steps=600, seed=int(i)))
        model.fit_svi(sim)

        # 3. 事後分位のランク統計量
        for name in ranks:
            if name not in model.posterior or name not in truth:
                continue
            post = np.asarray(model.posterior[name]).ravel()[:n_draws]
            ranks[name].append(int((post < truth[name]).sum()))

    out = {}
    for name, r in ranks.items():
        if len(r) < 10:
            continue
        u = (np.asarray(r) + 0.5) / (n_draws + 1)
        ks = kstest(u, "uniform")
        out[name] = {"ks_stat": float(ks.statistic), "p_value": float(ks.pvalue),
                     "n_sims": len(r), "uniform": bool(ks.pvalue > 0.05)}
    return out


def _simulate_orders(arr: RaceArrays, prior, key) -> RaceArrays:
    """事前分布の μ から Gumbel-max で着順を生成し、order を差し替えた配列を返す。"""
    J = _lazy_import()
    jax, jnp = J["jax"], J["jnp"]

    mu = np.asarray(prior["_mu"][0]) if "_mu" in prior else None
    if mu is None:
        post = {k: np.asarray(v) for k, v in prior.items()}
        theta = _theta_of(post, 0)
        mu = (theta[arr.horse, arr.seq]
              + _effect_of(post, 0, "gamma")[arr.jockey]
              + _effect_of(post, 0, "delta")[arr.sire])
        if "beta" in prior and arr.z.shape[2]:
            mu = mu + arr.z @ np.asarray(prior["beta"])[0]
    mu = np.where(arr.mask, mu, NEG_INF)

    g = np.asarray(jax.random.gumbel(key, mu.shape))
    utility = np.where(arr.mask, mu + g, -np.inf)
    order = np.argsort(-utility, axis=1)
    return RaceArrays(
        arr.horse, arr.jockey, arr.sire, arr.seq, arr.z, arr.mask, order, arr.n_ranked,
        arr.n_horses, arr.n_jockeys, arr.n_sires, arr.max_starts,
        arr.race_ids, arr.row_index, arr.covariates,
    )
