"""
ML 모델 관리 모듈
"""

import os
import time
from typing import Optional
from .storage_manager import StorageManager
from config.settings import MODEL_PATH, BASIC_MODEL_PATH, MODEL_DOWNLOAD_TIMEOUT

class ModelManager:
    """ML 모델 다운로드 및 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.storage_manager = StorageManager()
        self.models_downloaded = False
        print("ModelManager 초기화 완료")
    
    def ensure_models_available(self, force_redownload: bool = False) -> bool:
        """
        필요한 모델들이 로컬에 있는지 확인하고, 없으면 다운로드
        
        Args:
            force_redownload: 강제 재다운로드 여부
            
        Returns:
            bool: 모델 준비 완료 여부
        """
        try:
            # 이미 다운로드된 경우 체크
            if not force_redownload and self.models_downloaded:
                if self._check_models_exist():
                    print("모델이 이미 준비되어 있습니다.")
                    return True
            
            print("모델 준비 상태 확인 중...")
            
            # 모델 파일 존재 여부 확인
            if self._check_models_exist():
                print("모델 파일이 이미 존재합니다.")
                self.models_downloaded = True
                return True
            
            print("모델 파일이 없습니다. 오브젝트 스토리지에서 다운로드를 시작합니다...")
            
            # 모델 다운로드
            start_time = time.time()
            success = self.storage_manager.download_all_models()
            
            if success:
                download_time = time.time() - start_time
                print(f"모든 모델 다운로드 완료! 소요 시간: {download_time:.2f}초")
                
                # 모델 파일 검증
                if self._validate_models():
                    self.models_downloaded = True
                    print("모델 검증 완료!")
                    return True
                else:
                    print("모델 검증 실패!")
                    return False
            else:
                print("모델 다운로드 실패!")
                return False
                
        except Exception as e:
            print(f"모델 준비 중 오류 발생: {e}")
            return False
    
    def _check_models_exist(self) -> bool:
        """
        필요한 모델 파일들이 존재하는지 확인
        
        Returns:
            bool: 모든 모델 파일 존재 여부
        """
        try:
            # 훈련된 모델 파일 체크
            train_model_files = [
                os.path.join(MODEL_PATH, "saved_model.pb"),
                os.path.join(MODEL_PATH, "variables", "variables.index"),
                os.path.join(MODEL_PATH, "variables", "variables.data-00000-of-00001")
            ]
            
            # 기본 모델 파일 체크
            basic_model_files = [
                os.path.join(BASIC_MODEL_PATH, "saved_model.pb"),
                os.path.join(BASIC_MODEL_PATH, "variables", "variables.index"),
                os.path.join(BASIC_MODEL_PATH, "variables", "variables.data-00000-of-00001")
            ]
            
            all_files = train_model_files + basic_model_files
            
            for file_path in all_files:
                if not os.path.exists(file_path):
                    print(f"모델 파일 누락: {file_path}")
                    return False
            
            print("모든 모델 파일이 존재합니다.")
            return True
            
        except Exception as e:
            print(f"모델 파일 존재 여부 확인 실패: {e}")
            return False
    
    def _validate_models(self) -> bool:
        """
        다운로드된 모델 파일들의 무결성 검증
        
        Returns:
            bool: 모델 검증 성공 여부
        """
        try:
            print("모델 파일 무결성 검증 중...")
            
            # 파일 크기 체크 (최소 크기) - 실제 파일 크기에 맞춤
            min_file_sizes = {
                "saved_model.pb": 1024,  # 1KB
                "variables.index": 100,   # 100B
                "variables.data-00000-of-00001": 16 * 1024  # 16KB (실제 파일 크기에 맞춤)
            }
            
            # 훈련된 모델 검증
            for filename, min_size in min_file_sizes.items():
                if filename.startswith("variables."):
                    # variables 폴더 내의 파일들
                    file_path = os.path.join(MODEL_PATH, "variables", filename)
                else:
                    # 루트 폴더의 파일들
                    file_path = os.path.join(MODEL_PATH, filename)
                
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size < min_size:
                        print(f"훈련된 모델 파일 크기 부족: {filename} ({file_size} < {min_size})")
                        return False
                else:
                    print(f"훈련된 모델 파일 누락: {filename}")
                    return False
            
            # 기본 모델 검증
            for filename, min_size in min_file_sizes.items():
                if filename.startswith("variables."):
                    # variables 폴더 내의 파일들
                    file_path = os.path.join(BASIC_MODEL_PATH, "variables", filename)
                else:
                    # 루트 폴더의 파일들
                    file_path = os.path.join(BASIC_MODEL_PATH, filename)
                
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size < min_size:
                        print(f"기본 모델 파일 크기 부족: {filename} ({file_size} < {min_size})")
                        return False
                else:
                    print(f"기본 모델 파일 누락: {filename}")
                    return False
            
            print("모든 모델 파일 검증 완료!")
            return True
            
        except Exception as e:
            print(f"모델 검증 중 오류 발생: {e}")
            return False
    
    def get_model_paths(self) -> dict:
        """
        현재 사용 가능한 모델 경로 반환
        
        Returns:
            dict: 모델 경로 정보
        """
        return {
            'train_model': MODEL_PATH,
            'basic_model': BASIC_MODEL_PATH,
            'models_ready': self.models_downloaded
        }
    
    def cleanup_models(self) -> bool:
        """
        로컬 모델 파일 정리 (디버깅/테스트용)
        
        Returns:
            bool: 정리 성공 여부
        """
        try:
            print("로컬 모델 파일 정리 중...")
            
            import shutil
            
            # 모델 디렉토리 삭제
            if os.path.exists(MODEL_PATH):
                shutil.rmtree(MODEL_PATH)
                print(f"훈련된 모델 디렉토리 삭제: {MODEL_PATH}")
            
            if os.path.exists(BASIC_MODEL_PATH):
                shutil.rmtree(BASIC_MODEL_PATH)
                print(f"기본 모델 디렉토리 삭제: {BASIC_MODEL_PATH}")
            
            self.models_downloaded = False
            print("모델 파일 정리 완료!")
            return True
            
        except Exception as e:
            print(f"모델 파일 정리 실패: {e}")
            return False
