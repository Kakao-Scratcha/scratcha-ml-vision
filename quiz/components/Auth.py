import boto3
import os
from dotenv import load_dotenv

load_dotenv()

client = boto3.client(
    region_name="kr-central-2",
    endpoint_url="https://objectstorage.kr-central-2.kakaocloud.com",
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    service_name="s3"
)

# 버킷 이름
quiz_bucket_name = os.getenv("BUCKET_NAME")
dev_bucket_name = os.getenv("DEV_BUCKET_NAME")

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






