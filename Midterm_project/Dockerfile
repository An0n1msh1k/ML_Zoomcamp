FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Встановлюємо OpenMP (обов'язкова системна залежність для XGBoost)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY features.py predict.py model.bin ./

EXPOSE 9696

CMD ["uv", "run", "--no-dev", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]