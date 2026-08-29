# Cloud Run 用イメージ。
#
# ビルドコンテキストはリポジトリのルート:
#   docker build -f operation/Dockerfile -t <image> .
#
# 学習側パッケージ（nar）も入れる。特徴量計算・確率の正規化・期待値計算は
# 学習と同じ実装を共有するのが SK-02 の要件で、運用側に写しを置いた時点で
# train-serving skew が入る。
#
# PyTorch は入れない。TabM は ONNX に変換して onnxruntime で動かす。
# torch を入れると数百MB 増えてコールドスタートが延び、min-instances=0 の
# 構成では起動のたびにその代償を払うことになる。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM は libgomp を要求する。slim には入っていない
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && apt-get clean \
 && find /var/lib/apt/lists -type f -delete

# 依存だけ先に入れてレイヤキャッシュを効かせる
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      "pandas>=2.2" "numpy>=2.0" "duckdb>=1.1" "pyarrow>=17" "httpx>=0.27" \
      "pyyaml>=6.0" "fastapi>=0.115" "pydantic>=2.9" "uvicorn[standard]>=0.30" \
      "scipy>=1.14" "fsspec>=2024.6" "lightgbm>=4.5" "scikit-learn>=1.5" \
      "beautifulsoup4>=4.12" "lxml>=5.0" \
      "onnxruntime>=1.19" \
      "google-cloud-bigquery" "db-dtypes>=1.2" \
      "google-cloud-storage" "google-cloud-tasks" \
      "google-cloud-secret-manager" "gcsfs>=2024.6"

COPY learning/src /app/learning/src
COPY learning/conf /app/learning/conf
COPY operation/src /app/operation/src
COPY operation/conf /app/operation/conf

# max_lookback_days のような学習側の設定は nar.config がパッケージ相対で読む。
# 運用側と学習側の両方を import パスに載せる。
ENV PYTHONPATH=/app/operation/src:/app/learning/src

# Cloud Run が $PORT を渡してくる。8080 に固定しない
ENV PORT=8080
EXPOSE 8080

# ワーカーは増やさない。DuckDB のローカルファイルを掴む構成ではワーカー間で
# ロックが衝突する。並列度は Cloud Run のインスタンス数で出す。
CMD exec uvicorn narops.app:create_app --factory --host 0.0.0.0 --port ${PORT}
