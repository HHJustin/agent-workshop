# Agent Workshop — Production Docker Image
# Build:  docker build -t agent-workshop .
# Run:    docker run -p 9900:9900 --env-file .env agent-workshop

FROM python:3.12-slim-bookworm

# 非 root 用户
RUN groupadd -r app && useradd -r -g app -u 10001 app

WORKDIR /app

# 先装依赖（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple

# 复制源码
COPY --chown=app:app . .

# 数据目录
RUN mkdir -p /app/data /app/logs /app/chroma_db && chown -R app:app /app

USER app

EXPOSE 9900

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9900/health')" || exit 1

CMD ["python", "-m", "app.main"]
