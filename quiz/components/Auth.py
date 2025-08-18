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

bucket_name = os.getenv("BUCKET_NAME")
#버킷 파일 목록 조회
def get_list_objects(bucket_name):
    try:
        response = client.list_objects(Bucket=bucket_name)
        return [obj.get('Key') for obj in response.get('Contents', [])]
    except Exception as e:
        raise Exception(f"Failed to list objects: {e}")

#파일 업로드
def upload_file(local_path, bucket_name, file_name) :
    try :
        return client.upload_file(local_path, bucket_name, file_name)
    except Exception as e:
        raise Exception(f"Failed to upload file: {e}")
    
#파일 다운로드
def download_file(bucket_name, file_name, local_path) :
    try :
        return client.download_file(bucket_name, file_name, local_path)
    except Exception as e:
        raise Exception(f"Failed to download file: {e}")

#파일 삭제
def delete_object(bucket_name, file_name) :
    try :
        return client.delete_object(Bucket=bucket_name, Key=file_name)
    except Exception as e :
        raise Exception(f"Failed to delete object: {e}")

if __name__ == "__main__":
    print(get_list_objects(bucket_name))
    upload_file("test.pt", bucket_name, "test.pt")






