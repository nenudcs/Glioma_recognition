ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COMPETITION_WORKSPACE=/2026aicompetition/workspace

WORKDIR /opt/competition
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# 构建期 fail-fast：容器里的服务需要**基础镜像自带**的 CUDA 版 torch
# （requirements.txt 故意不声明 torch，见其中注释）。
# 不加这道断言的话，用不含 torch 的基础镜像（如默认的 python:3.11-slim）
# 构建出来的镜像要等**测评容器启动**时才会暴露 —— 那时 /health 都不通，
# 平台探针 1800s 超时后直接判失败，且比赛没有重来的机会。
#   构建时必须指定平台基础镜像：
#     docker build --build-arg BASE_IMAGE=<平台基础镜像 pytorch:*-cuda*-runtime> .
RUN python -c "import torch, numpy, scipy, yaml, fastapi, uvicorn, pydantic; print('deps ok: torch', torch.__version__)"
COPY . .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]

