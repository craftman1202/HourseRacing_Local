

# ------------------------------------------------------- BQ スキーマと DDL の一致
def test_bigquery_schemas_match_the_ddl():
    """infra/bq/*.json は DDL と同じ列を持つこと。

    片方だけに列を足すと、ローカルでは通るのに BigQuery 側の書き込みが落ちる。
    """
    import json
    from pathlib import Path

    from narops.db.schema import table_columns

    for path in sorted((Path(__file__).parent.parent / "infra" / "bq").glob("*.json")):
        try:
            expected = table_columns(path.stem)
        except KeyError:
            continue
        actual = [c["name"] for c in json.loads(path.read_text(encoding="utf-8"))]
        # 列の**集合**が一致していればよい。BigQuery は既存テーブルの列順を
        # 変えられず、後から足した列は必ず末尾に付く。書き込みは列名を明示して
        # いるので順序に依存しない。
        assert set(actual) == set(expected), (
            f"{path.stem}: BQ スキーマと DDL の列が違います "
            f"（DDL のみ: {sorted(set(expected) - set(actual))} / "
            f"BQ のみ: {sorted(set(actual) - set(expected))}）")


def test_container_installs_every_runtime_import():
    """Dockerfile が実行時 import を全部入れていること。

    出馬表の解析に bs4 を使い始めたとき Dockerfile に足し忘れ、
    コンテナが起動時に ModuleNotFoundError で落ちた。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "operation" / "Dockerfile").read_text(encoding="utf-8")
    installed = set(re.findall(r'"([A-Za-z0-9_.\-\[\]]+)[><=]', dockerfile))
    installed |= set(re.findall(r'"([A-Za-z0-9_.\-]+)"', dockerfile))

    # import 名 → 配布名
    required = {"bs4": "beautifulsoup4", "lightgbm": "lightgbm",
                "onnxruntime": "onnxruntime", "duckdb": "duckdb",
                "fastapi": "fastapi", "httpx": "httpx", "yaml": "pyyaml",
                "scipy": "scipy", "sklearn": "scikit-learn"}
    # import 文には出ないが実行時に要るもの。db-dtypes は BigQuery の
    # DATE 列を pandas に落とすときだけ必要で、無いと本番で初めて落ちる
    always = ["db-dtypes", "uvicorn", "lxml"]
    src = root / "operation" / "src"
    used = set()
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for mod in required:
            if re.search(rf"^\s*(from|import)\s+{re.escape(mod)}\b", text, re.M):
                used.add(mod)

    missing = [required[m] for m in sorted(used)
               if not any(required[m] in i for i in installed)]
    missing += [a for a in always if not any(a in i for i in installed)]
    assert not missing, f"Dockerfile に入っていない依存: {missing}"


def test_error_handler_does_not_raise_itself():
    """例外ハンドラが落ちると、本来の原因が握り潰される。

    `JSONResponse(detail=...)` は存在しない引数で、渡すとハンドラ自身が
    TypeError になる（実際にコンテナ起動時の 500 の原因が見えなくなった）。
    """
    from fastapi.testclient import TestClient

    from narops.app import create_app

    app = create_app()

    @app.get("/__boom")
    def boom():
        raise RuntimeError("秘密 https://discord.com/api/webhooks/xxx/yyy を含む")

    client = TestClient(app, raise_server_exceptions=False)
    res = client.get("/__boom")
    assert res.status_code == 500
    body = res.json()
    assert "detail" in body
    assert "discord.com/api/webhooks" not in body["detail"], "webhook が漏れています"


def test_no_module_reaches_into_the_duckdb_connection():
    """`wh.con` は DuckDB 実装にしか無い。

    BigQuery 構成では属性ごと存在しないので、直接触っている箇所は本番で
    必ず AttributeError になる（実際に /health が 500 を返した）。
    アクセスは両方が持つ query()/execute() 経由に限る。
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "narops"
    offenders = []
    for f in src.rglob("*.py"):
        if f.name == "backend.py":          # DuckDB 実装そのもの
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]        # 注釈の中の言及は対象外
            if re.search(r"\bwh\.con\b|\bwarehouse\.con\b", code):
                offenders.append(f"{f.relative_to(src)}:{i}")
    assert not offenders, f"DuckDB の接続を直接触っています: {offenders}"


def test_both_warehouse_backends_share_the_query_interface():
    """2つのバックエンドが同じ呼び出し方を提供していること。"""
    import inspect

    from narops.db.backend import Warehouse
    from narops.gcp import BigQueryWarehouse

    for name in ("query", "total_bytes_scanned"):
        assert hasattr(Warehouse, name) and hasattr(BigQueryWarehouse, name), name
    duck = inspect.signature(Warehouse.query).parameters
    bq = inspect.signature(BigQueryWarehouse.query).parameters
    for arg in ("sql", "params", "max_bytes_billed", "allow_full_scan"):
        assert arg in duck and arg in bq, f"query に {arg} がありません"


def test_history_tables_are_not_partitioned_by_day():
    """1998年からの履歴を日次で切ると BigQuery の上限（10,000）を超える。

    実際に 10,210 パーティションになり、2025 年以降の投入が拒否された。
    履歴を持つ表は月次にする。
    """
    from narops.deploy import PARTITION_COLUMN, PARTITION_TYPE

    history_tables = ("entry_result_final", "feature_snapshot", "prediction",
                      "bet_candidate")
    for t in history_tables:
        assert PARTITION_TYPE.get(t) == "MONTH", f"{t} が日次のままです"
    # 全テーブルに粒度の指定があること
    assert set(PARTITION_TYPE) == set(PARTITION_COLUMN)


def test_daily_partitioning_would_exceed_the_limit_for_28_years():
    """上限の根拠を数字で残す。"""
    from datetime import date

    days = (date(2026, 8, 26) - date(1998, 1, 1)).days
    assert days > 10_000, f"{days} 日。日次では上限 10,000 を超える"
    months = (2026 - 1998) * 12 + 8
    assert months < 10_000


def test_lifecycle_config_is_passed_as_a_file_not_stdin(tmp_path, monkeypatch):
    """`--lifecycle-file -` は標準入力を受け付けない。

    `-` をファイル名として開こうとして失敗する（実際にライフサイクル設定が
    2バケットとも FAIL した）。一時ファイルに落として渡すこと。
    """
    from pathlib import Path

    from narops import deploy

    seen = {}

    def fake_run(cmd, timeout=None, stdin=None):
        seen["cmd"] = list(cmd)
        seen["stdin"] = stdin
        # コマンドが指すファイルが実在し、中身が JSON であること
        target = cmd[cmd.index("--lifecycle-file") + 1]
        seen["body"] = Path(target).read_text(encoding="utf-8")
        return 0, ""

    monkeypatch.setattr(deploy, "_run", fake_run)
    plan = deploy.Plan("p", "r")
    plan.steps.append(deploy.Step(
        "gcs_lifecycle", "b:lifecycle",
        ["gcloud", "storage", "buckets", "update", "gs://b",
         "--lifecycle-file", "-"],
        stdin='{"lifecycle": {"rule": []}}'))

    results = deploy.apply_plan(plan, confirm=True)
    assert results[0][1] is True
    assert "-" not in seen["cmd"][-1], "`-` のまま渡しています"
    assert seen["stdin"] is None
    assert "lifecycle" in seen["body"]


def test_writes_go_through_the_shared_insert_path():
    """書き込みは両バックエンドが持つ insert_frame に寄せること。

    `register` + `INSERT ... SELECT *` は DuckDB 固有で、BigQuery 実装には
    register 自体が無い。直接使っていると本番で AttributeError になる
    （実際に /weekly-report が 500 で落ちた）。
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "narops"
    offenders = []
    for f in src.rglob("*.py"):
        if f.name == "backend.py":          # DuckDB 実装そのもの
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\b\w*wh\w*\.register\s*\(", code):
                offenders.append(f"{f.relative_to(src)}:{i}")
    assert not offenders, f"register を直接使っています: {offenders}"


def test_both_backends_expose_insert_frame():
    from narops.db.backend import Warehouse
    from narops.gcp import BigQueryWarehouse

    for cls in (Warehouse, BigQueryWarehouse):
        assert hasattr(cls, "insert_frame"), cls.__name__


def test_bigquery_backend_refuses_to_create_schema():
    """スキーマ作成はアプリからやらせない。

    BigQuery のテーブルはパーティションとフィルタ必須の設定を伴う。
    アプリが CREATE TABLE を通せると、設定の無い表が本番にできてしまう。
    """
    from unittest.mock import MagicMock

    import pytest

    pytest.importorskip("google.cloud.bigquery",
                        reason="GCP クライアント未導入の環境ではスキップ")
    from narops.gcp import BigQueryWarehouse

    wh = BigQueryWarehouse(project="sample-335613", dataset="nar_ops")
    wh._client = MagicMock()
    wh.execute("CREATE TABLE x (a INT64)")
    wh._client.query.assert_not_called()


def test_bigquery_execute_qualifies_tables_with_the_dataset():
    """修飾なしのテーブル名は BigQuery で 400 になる。

    query 側には default_dataset があったのに execute には無く、
    /plan-day だけが本番で落ちていた。両方に同じ設定が要る。
    """
    from unittest.mock import MagicMock

    import pytest

    pytest.importorskip("google.cloud.bigquery",
                        reason="GCP クライアント未導入の環境ではスキップ")
    from narops.gcp import BigQueryWarehouse

    wh = BigQueryWarehouse(project="sample-335613", dataset="nar_ops",
                           max_bytes_billed=10_000)
    wh._client = MagicMock()
    wh.execute("DELETE FROM race_schedule WHERE race_id = ?", ["r1"])

    _, kwargs = wh._client.query.call_args
    assert kwargs["job_config"].default_dataset is not None, "既定データセット未設定"


def test_bigquery_query_enforces_partition_filters_like_duckdb():
    """allow_full_scan でパーティション条件を免除しない。

    BigQuery は require_partition_filter で実行そのものを拒否するので、
    免除しても通らない。ローカルだけ通る抜け道になる。
    """
    import pytest

    pytest.importorskip("google.cloud.bigquery",
                        reason="GCP クライアント未導入の環境ではスキップ")
    from narops.errors import PartitionFilterRequired
    from narops.gcp import BigQueryWarehouse

    wh = BigQueryWarehouse(project="sample-335613", dataset="nar_ops",
                           max_bytes_billed=10_000)
    with pytest.raises(PartitionFilterRequired):
        wh.query("SELECT COUNT(*) FROM entry_result_final", allow_full_scan=True)


# ------------------------------------------------------------------ 運用モード
def test_mode_env_var_name_matches_what_deploy_sets():
    """デプロイが渡す変数名と、コードが読む変数名が一致していること。

    以前は deploy が `NAROPS_MODE` を渡し、コードは `OPS_MODE` を読んでいた。
    モード指定がまったく効かず、常に shadow のまま動いていた。
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "narops"
    deploy_src = (src / "deploy.py").read_text(encoding="utf-8")
    mode_src = (src / "mode.py").read_text(encoding="utf-8")

    set_names = set(re.findall(r'"(\w*MODE)=', deploy_src))
    read_names = set(re.findall(r'environ\.get\("(\w*MODE)"', mode_src))
    assert set_names, "deploy がモードを渡していません"
    assert set_names <= read_names, (
        f"deploy が渡す {set_names} をコードが読んでいません（読むのは {read_names}）")


def test_conflicting_mode_variables_stop_startup(monkeypatch):
    import pytest

    from narops.mode import OperatingState

    monkeypatch.setenv("NAROPS_MODE", "paper")
    monkeypatch.setenv("OPS_MODE", "live")
    with pytest.raises(RuntimeError, match="食い違"):
        OperatingState.from_env()


def test_secret_resolution_reaches_secret_manager_on_gcp(monkeypatch):
    """Cloud Run に .env は無い。Secret Manager を渡さないと webhook が解決できない。

    渡し忘れると「通知できるつもりで何も出ない」状態になる。
    """
    import inspect

    from narops import app as app_mod

    src = inspect.getsource(app_mod._attach_discord)
    assert "SecretManagerResolver" in src, "Secret Manager を繋いでいません"
    assert "secret_manager=" in src, "SecretResolver に渡していません"


def test_only_the_inference_service_gets_the_model_bucket():
    """モデルを読むのは推論を持つサービスだけ。

    API と Web は BigQuery の結果を出すだけで、GCS 読み取り権限を持たない
    （最小権限）。バケットを渡すと起動時に取りに行って 403 で落ちる。
    """
    from narops.deploy import build_plan

    plan = build_plan(image="img", mode="paper")
    for step in plan.steps:
        if step.kind != "cloud_run":
            continue
        env = next(a for a in step.command if "NAROPS_PROJECT=" in a)
        has_bucket = "NAROPS_MODEL_BUCKET=" in env
        assert has_bucket == (step.name == "nar-ops"), (
            f"{step.name}: モデルバケットの有無が役割と合いません")


def test_only_the_ops_role_sends_notifications(monkeypatch):
    """API と Web は Secret Manager への権限を持たない（最小権限）。

    役割を見ずに webhook を要求すると、通知に関係ないサービスが起動できない。
    """
    from types import SimpleNamespace

    from narops import app as app_mod

    svc = SimpleNamespace()
    monkeypatch.setenv("NAROPS_ROLE", "api")
    app_mod._attach_discord(svc, SimpleNamespace(discord_rate_limit_rps=1),
                            SimpleNamespace())
    assert not hasattr(svc, "sender") and not hasattr(svc, "alert_sender")


def test_deploy_passes_the_role_to_every_service():
    from narops.deploy import build_plan

    plan = build_plan(image="img", mode="paper")
    roles = {}
    for step in plan.steps:
        if step.kind != "cloud_run":
            continue
        env = next(a for a in step.command if "NAROPS_ROLE=" in a)
        roles[step.name] = next(
            p.split("=", 1)[1] for p in env.split(",") if p.startswith("NAROPS_ROLE="))
    assert roles == {"nar-ops": "ops", "nar-api": "api", "nar-web": "web"}


def test_optional_secret_degrades_when_access_is_denied():
    """権限が無いのは「未設定」と同じ扱い。required のときだけ止める。"""
    import pytest

    from narops.config import SecretResolver

    class Denying:
        def access(self, name):
            raise PermissionError("denied")

    r = SecretResolver(None, secret_manager=Denying(), project="p")
    assert r.get("DISCORD_WEBHOOK_ALERT", required=False) == ""
    with pytest.raises(RuntimeError, match="未設定"):
        r.get("DISCORD_WEBHOOK_PREDICTION", required=True)


def test_ops_service_has_memory_for_three_models():
    """1Gi では推論中にインスタンスが落ちる。

    conditional logit / LightGBM / TabM(ONNX) を同時に載せ、そこへ
    BigQuery の履歴取得結果が乗る。実測 1041 MiB で上限超過になり、
    応答は 503（Service Unavailable）になった。
    """
    import json
    import pathlib

    spec = json.loads(
        (pathlib.Path(__file__).parents[1] / "infra/services.json").read_text())
    mem = spec["cloud_run"]["nar-ops"]["memory"]
    assert mem.endswith("Gi") and int(mem[:-2]) >= 2, mem
