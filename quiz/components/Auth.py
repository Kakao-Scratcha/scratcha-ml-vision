import boto3
import os
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# Kakao Cloud Object Storage 연결 설정 최적화
config = Config(
    region_name="kr-central-2",
    read_timeout=300,  # 읽기 타임아웃 5분
    connect_timeout=120,  # 연결 타임아웃 2분
    retries={
        'max_attempts': 5,
        'mode': 'adaptive'
    }
)

client = boto3.client(
    endpoint_url="https://objectstorage.kr-central-2.kakaocloud.com",
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    service_name="s3",
    config=config
)

# 버킷 이름
quiz_bucket_name = os.getenv("BUCKET_NAME")
dev_bucket_name = os.getenv("DEV_BUCKET_NAME")

# 클라이언트 연결 테스트 및 로그
try:
    print("boto3 클라이언트 연결 테스트 중...")
    print(f"  - 엔드포인트: https://objectstorage.kr-central-2.kakaocloud.com")
    
    # 환경변수 확인
    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
    print(f"  - ACCESS_KEY 존재: {'Yes' if access_key else 'No'}")
    print(f"  - SECRET_KEY 존재: {'Yes' if secret_key else 'No'}")
    if access_key:
        print(f"  - ACCESS_KEY 시작: {access_key[:10]}...")
    
    print(f"  - 퀴즈 버킷: {quiz_bucket_name}")
    print(f"  - 개발 버킷: {dev_bucket_name}")
    print(f"  - 연결 타임아웃: {config.connect_timeout}초")
    print(f"  - 읽기 타임아웃: {config.read_timeout}초")
    
    # 실제 연결 테스트 (간단한 API 호출)
    print("  - 실제 연결 테스트 수행 중...")
    # 빈 버킷 리스트 호출로 연결 확인
    response = client.list_buckets()
    print(f"  - 연결 성공! 계정에 {len(response.get('Buckets', []))}개 버킷 확인")
    
    print("✓ boto3 클라이언트 초기화 완료")
except Exception as e:
    print(f"✗ boto3 클라이언트 초기화 실패: {e}")
    print(f"  - 오류 타입: {type(e).__name__}")
    import traceback
    print(f"  - 상세 오류: {traceback.format_exc()}")

# 기존 호환성을 위한 변수
bucket_name = quiz_bucket_name

#버킷 파일 목록 조회
def get_list_objects(bucket_name):
    try:
        response = client.list_objects(Bucket=bucket_name)
        return [obj.get('Key') for obj in response.get('Contents', [])]
    except Exception as e:
        raise Exception(f"Failed to list objects: {e}")

# 개발용 버킷에서 파일 목록 조회
def get_dev_objects(prefix=None):
    try:
        # prefix가 None이면 dev_bucket_name을 사용
        bucket_name = dev_bucket_name
        
        # S3에서 특정 prefix로 시작하는 객체들 조회
        if prefix:
            response = client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
        else:
            response = client.list_objects_v2(Bucket=bucket_name)
            
        return [obj.get('Key') for obj in response.get('Contents', [])]
    except Exception as e:
        raise Exception(f"Failed to list dev objects: {e}")

#파일 업로드
def upload_file(local_path, bucket_name, file_name) :
    try :
        return client.upload_file(local_path, bucket_name, file_name)
    except Exception as e:
        raise Exception(f"Failed to upload file: {e}")

# 개발용 버킷에 파일 업로드
def upload_dev_file(local_path, bucket_name, file_name):
    try:
        if bucket_name is None:
            bucket_name = dev_bucket_name
        return client.upload_file(local_path, bucket_name, file_name)
    except Exception as e:
        raise Exception(f"Failed to upload dev file: {e}")
    
#파일 다운로드
def download_file(bucket_name, file_name, local_path) :
    try :
        return client.download_file(bucket_name, file_name, local_path)
    except Exception as e:
        raise Exception(f"Failed to download file: {e}")

# 개발용 버킷에서 파일 다운로드
def download_dev_file(bucket_name, file_name, local_path):
    try:
        if bucket_name is None:
            bucket_name = dev_bucket_name
        return client.download_file(bucket_name, file_name, local_path)
    except Exception as e:
        raise Exception(f"Failed to download dev file: {e}")

#파일 삭제
def delete_object(bucket_name, file_name) :
    try :
        return client.delete_object(Bucket=bucket_name, Key=file_name)
    except Exception as e :
        raise Exception(f"Failed to delete object: {e}")

# 개발용 버킷에서 파일 삭제
def delete_dev_object(bucket_name, file_name):
    try:
        if bucket_name is None:
            bucket_name = dev_bucket_name
        return client.delete_object(Bucket=bucket_name, Key=file_name)
    except Exception as e:
        raise Exception(f"Failed to delete dev object: {e}")

if __name__ == "__main__":
    print(get_list_objects(bucket_name))






