"""
YOLO 객체 검출 모듈
"""

import cv2
import numpy as np
from typing import List, Dict
from ultralytics import YOLO
from config.settings import VALID_CLASSES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD


class YOLODetector:
    """YOLO 객체 검출 클래스"""
    
    def __init__(self, model_path: str, basic_model_path: str):
        """
        초기화
        
        Args:
            model_path: YOLO 모델 경로 (문제 생성용)
            basic_model_path: 기본 YOLO 모델 경로 (검증용)
        """
        print("YOLO 검출기 초기화 중...")
        
        # YOLO 모델 로딩
        try:
            self.model = YOLO(model_path)
            print(f"✓ YOLO 모델 로딩 성공: {model_path}")
        except Exception as e:
            print(f"✗ YOLO 모델 로딩 실패: {e}")
            raise
            
        try:
            self.basic_model = YOLO(basic_model_path)  # 기본 YOLO 모델 (검증용)
            print(f"✓ 기본 YOLO 모델 로딩 성공: {basic_model_path}")
        except Exception as e:
            print(f"✗ 기본 YOLO 모델 로딩 실패: {e}")
            raise
        
        # GPU 사용 가능 여부 확인
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ 디바이스: {self.device}")
        if torch.cuda.is_available():
            print(f"  사용 가능한 GPU: {torch.cuda.device_count()}개")
        
        print("YOLO 검출기 초기화 완료!")
    
    def detect_objects(self, image_bytes: bytes) -> List[Dict]:
        """
        YOLO 모델로 객체 검출
        
        Args:
            image_bytes: 이미지 바이트 데이터
            
        Returns:
            List[Dict]: 검출된 객체 목록
        """
        # 바이트 데이터를 numpy 배열로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다.")
        
        # YOLO 모델로 객체 검출
        try:
            results = self.model(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            detected_objects = []
            
            # 결과 파싱
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # 바운딩 박스 좌표 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # 클래스 이름 가져오기
                        if class_id < len(result.names):
                            class_name = result.names[class_id]
                            
                            # VALID_CLASSES에 포함된 클래스만 처리
                            if class_name in VALID_CLASSES:
                                detected_objects.append({
                                    'class_name': class_name,
                                    'class_id': class_id,
                                    'confidence': confidence,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'area': int((x2 - x1) * (y2 - y1))
                                })
            
            # 신뢰도 순으로 정렬
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_objects
            
        except Exception as e:
            print(f"YOLO 객체 검출 실패: {e}")
            return []
    
    def detect_objects_with_basic_model(self, image_bytes: bytes) -> List[Dict]:
        """
        기본 YOLO 모델로 객체 검출 (검증용)
        
        Args:
            image_bytes: 이미지 바이트 데이터
            
        Returns:
            List[Dict]: 검출된 객체 목록
        """
        # 바이트 데이터를 numpy 배열로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다.")
        
        # 기본 YOLO 모델로 객체 검출
        try:
            results = self.basic_model(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            detected_objects = []
            
            # 결과 파싱
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # 바운딩 박스 좌표 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # 클래스 이름 가져오기
                        if class_id < len(result.names):
                            class_name = result.names[class_id]
                            
                            # VALID_CLASSES에 포함된 클래스만 처리
                            if class_name in VALID_CLASSES:
                                detected_objects.append({
                                    'class_name': class_name,
                                    'class_id': class_id,
                                    'confidence': confidence,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'area': int((x2 - x1) * (y2 - y1))
                                })
            
            # 신뢰도 순으로 정렬
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            return detected_objects
            
        except Exception as e:
            print(f"기본 YOLO 모델 객체 검출 실패: {e}")
            return []
    
    def validate_with_basic_model(self, processed_image_bytes: bytes, current_answer: Dict) -> bool:
        """
        노이즈 처리된 이미지를 기본 모델로 검증하여 결과가 다른지 확인
        
        Args:
            processed_image_bytes: 노이즈 처리된 이미지 바이트 데이터
            current_answer: 현재 모델이 선택한 정답 객체
            
        Returns:
            bool: 기본 모델의 결과가 현재 모델과 다른 경우 True
        """
        try:            
            # 기본 모델로 노이즈 처리된 이미지 검출
            basic_detected_objects = self.detect_objects_with_basic_model(processed_image_bytes)
            
            if not basic_detected_objects:
                return True  # 검출 실패는 다른 결과로 간주
            
            # 기본 모델의 최고 신뢰도 객체 선택
            basic_best_object = max(basic_detected_objects, key=lambda x: x['confidence'])
            
            # 신뢰도 임계값 확인
            if basic_best_object['confidence'] < CONFIDENCE_THRESHOLD:
                print(f" 기본 모델 최고 신뢰도가 임계값 미달: {basic_best_object['confidence']:.3f}")
                print(f"   - 기본 모델 인식률: {basic_best_object['confidence']*100:.1f}% (임계값 미달)")
                return True  # 신뢰도 미달은 다른 결과로 간주
            
            # 클래스명 비교
            current_class = current_answer['class_name']
            basic_class = basic_best_object['class_name']
            
            print(f" 기본 모델 검증 결과:")
            print(f" - 현재 모델 정답: {current_class} (신뢰도: {current_answer['confidence']:.3f})")
            print(f" - 기본 모델 결과: {basic_class} (신뢰도: {basic_best_object['confidence']:.3f})")
            print(f" - 기본 모델 인식률: {basic_best_object['confidence']*100:.1f}%")
            
            # 검출된 모든 객체 정보 표시
            print(f"   - 기본 모델 검출 객체 수: {len(basic_detected_objects)}개")
            if len(basic_detected_objects) > 1:
                print(f"   - 기본 모델 상위 3개 검출 결과:")
                for i, obj in enumerate(basic_detected_objects[:3]):
                    print(f"     {i+1}. {obj['class_name']} (신뢰도: {obj['confidence']:.3f})")
            
            # 결과가 다른 경우 True 반환
            if current_class != basic_class:
                return True
            else:
                return False
                
        except Exception as e:
            return True  # 오류 발생 시 다른 결과로 간주
