"""
오브젝트 스토리지 분리 관리 모듈
"""

import os
import time
from typing import List, Tuple, Optional
from .Auth import (
    quiz_bucket_name, dev_bucket_name,
    get_list_objects, get_dev_objects,
    upload_file, upload_dev_file,
    download_file, download_dev_file,
    delete_object, delete_dev_object
)
from config.settings import MODEL_DOWNLOAD_TIMEOUT, MODEL_DOWNLOAD_RETRY

class StorageManager:
    """오브젝트 스토리지 분리 관리 클래스"""
    
    def __init__(self):
        """초기화 - Auth.py의 설정을 사용"""
        print("  - StorageManager 시작...")
        print("  - Auth.py import 및 설정 로딩 중...")
        print(f"  - 퀴즈 이미지 버킷: {quiz_bucket_name}")
        print(f"  - 개발용 버킷: {dev_bucket_name}")
        print("  - boto3 클라이언트 연결 상태 확인 중...")
        
        # boto3 클라이언트 연결 테스트
        try:
            from .Auth import client
            print("  - boto3 클라이언트 가져오기 성공")
        except Exception as e:
            print(f"  - boto3 클라이언트 가져오기 실패: {e}")
            raise
        
        print("  - StorageManager 초기화 완료")
    
    def download_model_from_storage(self, model_key: str, local_path: str) -> bool:
        """
        ML 모델을 개발용 버킷에서 다운로드
        
        Args:
            model_key: 모델 파일 키 (예: "models/train_tf/saved_model.pb")
            local_path: 로컬 저장 경로
            
        Returns:
            bool: 다운로드 성공 여부
        """
        try:
            print(f"모델 다운로드 시작: {model_key} -> {local_path}")
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 다운로드 재시도 로직
            for attempt in range(MODEL_DOWNLOAD_RETRY):
                try:
                    start_time = time.time()
                    
                    download_dev_file(dev_bucket_name, model_key, local_path)
                    
                    download_time = time.time() - start_time
                    file_size = os.path.getsize(local_path)
                    
                    print(f"모델 다운로드 완료: {model_key}")
                    print(f"  - 파일 크기: {file_size / (1024*1024):.2f} MB")
                    print(f"  - 다운로드 시간: {download_time:.2f}초")
                    
                    return True
                    
                except Exception as e:
                    print(f"모델 다운로드 시도 {attempt + 1} 실패: {e}")
                    if attempt < MODEL_DOWNLOAD_RETRY - 1:
                        time.sleep(2 ** attempt)  # 지수 백오프
                    else:
                        raise e
                        
        except Exception as e:
            print(f"모델 다운로드 실패: {e}")
            return False
    
    def download_all_models(self, base_local_path: str = None) -> bool:
        """
        모든 필요한 모델을 개발용 버킷에서 다운로드
        
        Args:
            base_local_path: 로컬 모델 저장 기본 경로
            
        Returns:
            bool: 모든 모델 다운로드 성공 여부
        """
        try:
            print("모든 모델 다운로드 시작...")
            
            # 기본 경로가 없으면 설정에서 가져오기
            if base_local_path is None:
                from config.settings import MODEL_PATH, BASIC_MODEL_PATH
                base_local_path = os.path.dirname(MODEL_PATH)
            
            # 다운로드할 모델 목록 (폴더 경로)
            models_to_download = [
                ("models/train_tf", MODEL_PATH),
                ("models/yolo11x_tf", BASIC_MODEL_PATH)
            ]
            
            success_count = 0
            for model_folder, local_path in models_to_download:
                # 모델 폴더의 모든 파일 다운로드
                try:
                    # 개발용 버킷에서 특정 폴더의 파일 목록 조회
                    # get_dev_objects()는 폴더 경로를 prefix로 사용
                    model_files = get_dev_objects(model_folder)
                    
                    if model_files:
                        for file_key in model_files:
                            # file_key는 "models/train_tf/saved_model.pb" 형태
                            # 상대 경로 추출 (models/train_tf/saved_model.pb -> saved_model.pb)
                            relative_path = file_key.replace(f"{model_folder}/", "")
                            
                            # variables 폴더인 경우 하위 폴더 구조 유지
                            if relative_path.startswith("variables/"):
                                local_file_path = os.path.join(local_path, relative_path)
                            else:
                                local_file_path = os.path.join(local_path, relative_path)
                            
                            if self.download_model_from_storage(file_key, local_file_path):
                                success_count += 1
                            else:
                                print(f"모델 파일 다운로드 실패: {file_key}")
                                return False
                    else:
                        print(f"경고: {model_folder} 폴더에 파일이 없습니다.")
                    
                    print(f"모델 폴더 다운로드 완료: {model_folder}")
                    
                except Exception as e:
                    print(f"모델 폴더 다운로드 실패: {model_folder} - {e}")
                    return False
            
            print(f"모든 모델 다운로드 완료: {success_count}개 파일")
            return True
            
        except Exception as e:
            print(f"모델 다운로드 중 오류 발생: {e}")
            return False
    
    def get_random_original_image(self, folder_prefix: str = "images/") -> Tuple[str, bytes]:
        """
        개발용 버킷에서 문제 생성용 원본 이미지 가져오기
        
        Args:
            folder_prefix: 이미지 폴더 경로
            
        Returns:
            Tuple[str, bytes]: (이미지 키, 이미지 바이트 데이터)
        """
        try:
            # 개발용 버킷에서 이미지 파일 목록 조회
            image_files = get_dev_objects(folder_prefix)
            
            # 이미지 파일 필터링
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            filtered_images = []
            
            for obj_key in image_files:
                if any(obj_key.lower().endswith(ext) for ext in image_extensions):
                    filtered_images.append(obj_key)
            
            if not filtered_images:
                raise ValueError(f"'{folder_prefix}' 폴더에 이미지 파일이 없습니다.")
            
            # 랜덤 이미지 선택
            import random
            random_image_key = random.choice(filtered_images)
            print(f"선택된 원본 이미지: {random_image_key}")
            
            # 이미지 다운로드
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            try:
                download_dev_file(dev_bucket_name, random_image_key, temp_path)
                
                # 파일을 바이트로 읽기
                with open(temp_path, 'rb') as f:
                    image_bytes = f.read()
                
                return random_image_key, image_bytes
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"원본 이미지 가져오기 실패: {e}")
            raise ValueError(f"원본 이미지를 가져올 수 없습니다: {e}")
    
    def save_quiz_image(self, image_data: bytes, quiz_id: str, difficulty: str) -> str:
        """
        노이즈 처리된 문제 이미지를 기존 버킷에 저장 (WebP 형식으로 압축)
        
        Args:
            image_data: 이미지 바이트 데이터
            quiz_id: 퀴즈 ID
            difficulty: 난이도
            
        Returns:
            str: 저장된 이미지의 스토리지 키
        """
        try:
            # 기존 버킷에 저장 (WebP 형식)
            storage_key = f"quiz_images/{difficulty}/{quiz_id}.webp"
            
            # 이미지 바이트를 WebP 형식으로 압축
            import cv2
            import numpy as np
            
            # 바이트 데이터를 numpy 배열로 변환
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("이미지 디코딩에 실패했습니다")
            
            # WebP 형식으로 압축 (품질 85%)
            webp_params = [cv2.IMWRITE_WEBP_QUALITY, 85]
            
            # 임시 파일로 저장 후 업로드
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webp') as temp_file:
                temp_path = temp_file.name
                
                # WebP 형식으로 압축하여 저장
                success = cv2.imwrite(temp_path, img, webp_params)
                if not success:
                    raise ValueError("WebP 압축 저장에 실패했습니다")
            
            try:
                upload_file(temp_path, quiz_bucket_name, storage_key)
                
                print(f"문제 이미지 저장 완료: {storage_key}")
                return storage_key
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"문제 이미지 저장 실패: {e}")
            raise ValueError(f"문제 이미지를 저장할 수 없습니다: {e}")
    
    def list_quiz_images(self, difficulty: str = None) -> List[str]:
        """
        기존 버킷에서 문제 이미지 목록 조회
        
        Args:
            difficulty: 특정 난이도 (선택사항)
            
        Returns:
            List[str]: 이미지 키 목록
        """
        try:
            prefix = f"quiz_images/{difficulty}/" if difficulty else "quiz_images/"
            
            # 기존 방식으로 조회 (호환성 유지)
            all_objects = get_list_objects(quiz_bucket_name)
            filtered_objects = [obj for obj in all_objects if obj.startswith(prefix)]
            
            return filtered_objects
                
        except Exception as e:
            print(f"문제 이미지 목록 조회 실패: {e}")
            return []
