"""
CAPTCHA 퀴즈 생성기 설정
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# PyMySQL 데이터베이스 설정
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'charset': 'utf8mb4',
    'autocommit': True,
    'connect_timeout': 60
}

# 기본 설정 (TensorFlow 모델)
# Docker 환경과 로컬 환경 모두 지원하는 경로 설정
import os
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 모델 경로 (컨테이너 실행 시 오브젝트 스토리지에서 다운로드)
# /tmp/ 대신 영구 디렉토리 사용 (Windows에서는 현재 작업 디렉토리 하위)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "train_tf")  # TensorFlow YOLO11s 파인튜닝 모델 경로(문제 생성용)
BASIC_MODEL_PATH = os.path.join(_BASE_DIR, "models", "yolo11x_tf")  # 기본 TensorFlow YOLO11x 모델 경로 (문제 검증용)

# 이미지 폴더 경로
ORIGINAL_IMAGE_FOLDER = "images/"  # 새 버킷의 원본 이미지 경로
QUIZ_IMAGE_FOLDER = "quiz_images/"  # 기존 버킷의 문제 이미지 경로
IMAGE_FOLDER = "images/"  # 오브젝트 스토리지 문제 이미지 경로 (기존 호환성)

# 모델 다운로드 설정
MODEL_DOWNLOAD_TIMEOUT = 300  # 모델 다운로드 타임아웃 (초)
MODEL_DOWNLOAD_RETRY = 3      # 모델 다운로드 재시도 횟수
CONFIDENCE_THRESHOLD = 0.6  # 신뢰도 임계값 (문제 생성용)
IOU_THRESHOLD = 0.5  # IoU 임계값 (문제 생성용)
CREATED_AT = datetime.now().isoformat()  # 생성 시간 (문제 생성용)
EXPIRES_AT = (datetime.now() + timedelta(days=1)).isoformat()  # 만료 시간 하루 (문제 생성용)
PROMPT = "스크래치 후 정답을 선택하세요."  # 기본 퀴즈 문제(동적 프롬프트가 없을 때 사용)
MAX_RETRY_ATTEMPTS = 10  # 문제 생성 수 (문제 생성용)

# 노이즈 처리 설정
NOISE_FREQUENCY = 0.2  # 고주파 노이즈 강도(고정)
NOISE_INTENSITY = 0.2  # 기본 노이즈 강도(20%)
NOISE_ALPHA = 0.4  # 기본 알파 블랜딩 강도(40%)

# 난이도별 저장 구성 (노이즈 파라미터는 랜덤 생성)
DIFFICULTY_CONFIGS = {
    'high':   {'count': MAX_RETRY_ATTEMPTS, 'intensity_pct': '70~100', 'alpha_pct': '30~35'},  # 노이즈 70~100%, 알파 30~35% 랜덤
    'middle': {'count': MAX_RETRY_ATTEMPTS, 'intensity_pct': '50~70', 'alpha_pct': '20~30'},   # 노이즈 50~70%, 알파 20~30% 랜덤
    'low':    {'count': MAX_RETRY_ATTEMPTS, 'intensity_pct': '30~50', 'alpha_pct': '10~20'},    # 노이즈 30~50%, 알파 10~20% 랜덤
}

# 난이도 문자열을 숫자로 매핑
DIFFICULTY_TO_NUMBER = {
    'low': 0,
    'middle': 1,
    'high': 2
}

# 유효한 클래스 목록
VALID_CLASSES = [
    'backpack', 'bear', 'bed', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair',
    'clock', 'cow', 'cup', 'dog', 'elephant', 'mouse', 'refrigerator', 'sheep'
]

# 영어 클래스명을 한글 맵핑
CLASS_NAME_TO_KOREAN = {
    'backpack': '가방',
    'bear': '곰',
    'bed': '침대',
    'bird': '새',
    'boat': '배',
    'bottle': '병',
    'car': '자동차',
    'cat': '고양이',
    'chair': '의자',
    'clock': '시계',
    'cow': '소',
    'cup': '컵',
    'dog': '개',
    'elephant': '코끼리',
    'mouse': '쥐',
    'refrigerator': '냉장고',
    'sheep': '양'
}

# 유사 객체 오답 조합 (정답 - 오답 리스트)
SIMILAR_OBJECT_PAIRS = {
    '고양이': ['개', '여우'],
    '개': ['고양이', '늑대','여우','코요테'],
    '곰': ['소', '개', '늑대'],
    '쥐': ['개', '곰'],
    '자동차': ['배'],
    '배': ['자동차'],
    '시계': ['자동차'],
    '가방': ['병', '컵'],
    '병': ['가방', '컵'],
    '컵': ['가방', '병'],
    '침대': ['의자', '냉장고'],
    '의자': ['침대', '냉장고'],
    '냉장고': ['세탁기', '의자'],
    '새': [],
    '소': ['코뿔소', '양', '코끼리'],
    '양': ['염소', '소'],
    '코끼리': ['매머드','코뿔소']
}

# FastAPI 및 스케줄링 설정
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))

# 스케줄링 설정
SCHEDULE_ENABLED = os.getenv('SCHEDULE_ENABLED', 'true').lower() == 'true'
SCHEDULE_INTERVAL_HOURS = int(os.getenv('SCHEDULE_INTERVAL_HOURS', 8))  # 8시간마다 실행
SCHEDULE_TIMEZONE = os.getenv('SCHEDULE_TIMEZONE', 'Asia/Seoul')

# 스케줄링용 퀴즈 생성 수량 (크론잡 모드의 1/3)
SCHEDULED_QUIZ_COUNTS = {
    'high': max(1, MAX_RETRY_ATTEMPTS // 3),
    'middle': max(1, MAX_RETRY_ATTEMPTS // 3),
    'low': max(1, MAX_RETRY_ATTEMPTS // 3)
}
