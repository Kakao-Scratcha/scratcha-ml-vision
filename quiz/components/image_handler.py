"""
이미지 처리 및 저장 모듈
"""

import os
import cv2
import numpy as np
import random
import threading
import uuid
from typing import Tuple, Optional
from .Auth import bucket_name, get_list_objects, upload_file, download_file
from .image_preprocessor import ImagePreprocessor
from config.settings import (
    IMAGE_FOLDER, 
    NOISE_FREQUENCY, 
    NOISE_INTENSITY, 
    NOISE_ALPHA
)


class ImageHandler:
    """이미지 처리 및 저장 클래스"""
    
    def __init__(self):
        """초기화"""
        self.image_preprocessor = ImagePreprocessor()
        self.used_images = set()  # 사용된 이미지 추적
        self._lock = threading.Lock()  # 스레드 안전성을 위한 락
        print("ImageHandler 초기화 완료")
    
    def get_random_image_from_storage(self, folder_prefix: str = IMAGE_FOLDER) -> Tuple[str, bytes]:
        """
        오브젝트 스토리지에서 랜덤 이미지 가져오기 (중복 방지)
        
        Args:
            folder_prefix: 폴더 경로 (예: "images/")
            
        Returns:
            Tuple[str, bytes]: (이미지 키, 이미지 바이트 데이터)
        """
        try:
            # 오브젝트 스토리지에서 특정 폴더의 이미지 파일 목록 가져오기
            object_list = get_list_objects(bucket_name)
            
            # 지정된 폴더의 이미지 파일만 필터링
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = [obj for obj in object_list 
                          if obj.startswith(folder_prefix) and 
                          any(obj.lower().endswith(ext) for ext in image_extensions)]
            
            if not image_files:
                print(f"'{folder_prefix}' 폴더에 이미지 파일이 없습니다. 전체 버킷에서 검색합니다.")
                # 폴백: 전체 버킷에서 검색
                image_files = [obj for obj in object_list 
                              if any(obj.lower().endswith(ext) for ext in image_extensions)]
                
                if not image_files:
                    raise ValueError("오브젝트 스토리지에 이미지 파일이 없습니다.")
            
            # 스레드 안전하게 사용되지 않은 이미지 필터링
            with self._lock:
                available_images = [img for img in image_files if img not in self.used_images]
                
                if not available_images:
                    print("경고: 모든 이미지를 사용했습니다. 사용된 이미지 목록을 초기화합니다.")
                    self.used_images.clear()
                    available_images = image_files
                
                # 랜덤 이미지 선택
                random_image_key = random.choice(available_images)
                self.used_images.add(random_image_key)  # 사용된 이미지로 표시
            print(f"선택된 이미지: {random_image_key}")
            
            # 임시 파일로 다운로드
            temp_file_path = f"temp_{uuid.uuid4().hex}.jpg"
            
            try:
                download_file(bucket_name, random_image_key, temp_file_path)
                
                # 파일을 바이트로 읽기
                with open(temp_file_path, 'rb') as f:
                    image_bytes = f.read()
                
                return random_image_key, image_bytes
                
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            print(f"오브젝트 스토리지에서 이미지 가져오기 실패: {e}")
            raise ValueError(f"오브젝트 스토리지에서 이미지를 가져올 수 없습니다: {e}")
    
    def get_random_noise_params(self, difficulty: str) -> Tuple[float, float]:
        """
        난이도별 랜덤 노이즈 파라미터 생성
        
        Args:
            difficulty: 난이도 ('low', 'middle', 'high')
            
        Returns:
            Tuple[float, float]: (노이즈 강도, 알파 블렌딩 강도)
        """
        if difficulty == 'low':
            # low: 노이즈 30~50%, 알파 10~20%
            intensity = random.uniform(0.3, 0.5)
            alpha = random.uniform(0.1, 0.2)
        elif difficulty == 'middle':
            # middle: 노이즈 50~70%, 알파 20~30%
            intensity = random.uniform(0.5, 0.7)
            alpha = random.uniform(0.2, 0.3)
        elif difficulty == 'high':
            # high: 노이즈 70~100%, 알파 30~35%
            intensity = random.uniform(0.7, 1.0)
            alpha = random.uniform(0.3, 0.35)
        else:
            # 기본값 사용
            intensity = NOISE_INTENSITY
            alpha = NOISE_ALPHA
            
        return intensity, alpha

    def process_image_with_noise(self, image_bytes: bytes, *, 
                                intensity: Optional[float] = None, 
                                alpha: Optional[float] = None, 
                                difficulty: Optional[str] = None) -> np.ndarray:
        """
        이미지에 노이즈 처리 적용 (3가지 노이즈 조합)
        
        Args:
            image_bytes: 원본 이미지 바이트 데이터
            intensity: 노이즈 강도 (0.0~1.0). None이면 난이도별 랜덤값 또는 기본값 사용
            alpha: 알파 블렌딩 강도 (0.0~1.0). None이면 난이도별 랜덤값 또는 기본값 사용
            difficulty: 난이도 ('low', 'middle', 'high'). intensity와 alpha가 None일 때 사용
            
        Returns:
            np.ndarray: 노이즈 처리된 이미지 배열
        """
        # 바이트 데이터를 numpy 배열로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다.")
        
        # 난이도별 랜덤 파라미터 생성
        if intensity is None and difficulty:
            intensity, alpha = self.get_random_noise_params(difficulty)
        else:
            intensity = NOISE_INTENSITY if intensity is None else float(intensity)
            alpha = NOISE_ALPHA if alpha is None else float(alpha)
        
        processed_image = self.image_preprocessor.preprocess(
            image=image,
            frequency=NOISE_FREQUENCY,  # 고주파 노이즈 강도
            intensity=intensity,
            alpha=alpha,
            output_path=None  # 로컬 저장 안함
        )
        
        return processed_image
    
    def save_image_array_to_object_storage(self, image_array: np.ndarray, quiz_id: str, 
                                         difficulty: Optional[str] = None) -> str:
        """
        처리된 이미지 배열을 오브젝트 스토리지에 webp 압축으로 저장
        
        Args:
            image_array: 노이즈 처리된 이미지 배열
            quiz_id: 퀴즈 ID
            difficulty: 난이도 폴더명 (예: 'high' | 'middle' | 'low')
            
        Returns:
            str: 오브젝트 스토리지 키
        """
        try:
            # 파일명 생성 (webp 형식)
            filename = f"quiz_{quiz_id}.webp"
            subdir = (difficulty.strip('/') + '/') if difficulty else ''
            storage_key = f"quiz_images/{subdir}{filename}"
            
            # numpy 배열을 webp 바이트로 인코딩 (압축 품질 80%)
            success, encoded_image = cv2.imencode('.webp', image_array, [cv2.IMWRITE_WEBP_QUALITY, 80])
            if not success:
                raise ValueError("webp 이미지 인코딩에 실패했습니다.")
            
            # 임시 파일 생성하여 업로드
            temp_file_path = f"temp_{quiz_id}.webp"
            try:
                cv2.imwrite(temp_file_path, image_array, [cv2.IMWRITE_WEBP_QUALITY, 80])
                upload_file(temp_file_path, bucket_name, storage_key)
                return storage_key
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            
        except Exception as e:
            # 폴백: 퀴즈 ID를 키로 사용 (webp 형식)
            return f"quiz_images/quiz_{quiz_id}.webp"
