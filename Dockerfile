# TensorFlow GPU 기반 이미지 사용
FROM tensorflow/tensorflow:2.13.0-gpu

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 모델 저장 디렉토리 생성
RUN mkdir -p /tmp/models/train_tf /tmp/models/yolo11x_tf

# 포트 노출 (FastAPI 기본 포트)
EXPOSE 8000

# 컨테이너 시작 시 FastAPI 서버 실행
CMD ["python", "-m", "quiz.fastapi_app"]

