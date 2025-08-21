FROM python:3.11-slim

# 기본 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    OMP_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1

# 필수 OS 패키지 (과도 설치 금지)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 의존성
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install gunicorn

# 앱 소스
COPY . .

# Hugging Face Spaces는 0.0.0.0:${PORT} 리슨을 기대
EXPOSE 7860

# Gunicorn (WSGI) 실행: 워커/스레드는 메모리에 맞춰 튜닝
CMD bash -lc "gunicorn app:app -w 2 -k gthread --threads 8 --bind 0.0.0.0:${PORT} --timeout 180"
