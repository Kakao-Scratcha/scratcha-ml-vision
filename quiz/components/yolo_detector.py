"""
YOLO 객체 검출 모듈 (TensorFlow 버전)
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict
from config.settings import VALID_CLASSES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD


class YOLODetector:
    """YOLO 객체 검출 클래스"""
    
    def __init__(self, model_path: str, basic_model_path: str):
        """
        초기화
        
        Args:
            model_path: TensorFlow YOLO 모델 경로 (문제 생성용)
            basic_model_path: 기본 TensorFlow YOLO 모델 경로 (검증용)
        """
        print("TensorFlow YOLO 검출기 초기화 중...")
        
        # TensorFlow YOLO 모델 로딩
        try:
            # SavedModel 형식으로 로딩
            self.model = tf.saved_model.load(model_path)
            self.predict_fn = self.model.signatures['serving_default']
            print(f"✓ TensorFlow YOLO 모델 로딩 성공: {model_path}")
        except Exception as e:
            print(f"✗ TensorFlow YOLO 모델 로딩 실패: {e}")
            raise
            
        try:
            # 기본 TensorFlow YOLO 모델 로딩 (검증용)
            self.basic_model = tf.saved_model.load(basic_model_path)
            self.basic_predict_fn = self.basic_model.signatures['serving_default']
            print(f"✓ 기본 TensorFlow YOLO 모델 로딩 성공: {basic_model_path}")
        except Exception as e:
            print(f"✗ 기본 TensorFlow YOLO 모델 로딩 실패: {e}")
            raise
        
        # GPU 사용 가능 여부 확인
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU 사용 가능: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            # GPU 메모리 성장 설정
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU 메모리 설정 오류: {e}")
        else:
            print("✓ CPU 모드로 실행")
        
        # COCO 클래스 이름 매핑 (YOLO 표준)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        print("TensorFlow YOLO 검출기 초기화 완료!")
    
    def _preprocess_image(self, image_bytes: bytes) -> tf.Tensor:
        """
        이미지 전처리
        
        Args:
            image_bytes: 이미지 바이트 데이터
            
        Returns:
            tf.Tensor: 전처리된 이미지 텐서
        """
        # 바이트 데이터를 numpy 배열로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("이미지를 읽을 수 없습니다.")
        
        # BGR to RGB 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 조정 (640x640)
        image = cv2.resize(image, (640, 640))
        
        # 정규화 (0-255 -> 0-1)
        image = image.astype(np.float32) / 255.0
        
        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)
        
        return tf.constant(image)
    
    def _postprocess_predictions(self, predictions, image_shape) -> List[Dict]:
        """
        TensorFlow YOLO 예측 결과 후처리
        
        Args:
            predictions: 모델 예측 결과
            image_shape: 원본 이미지 크기
            
        Returns:
            List[Dict]: 검출된 객체 목록
        """
        detected_objects = []
        
        try:
            # TensorFlow YOLO 출력: 'output_0' 키로 (1, num_classes+4, 8400) 형태
            output_key = 'output_0'
            if output_key not in predictions:
                output_key = list(predictions.keys())[0]
            
            # 출력 텐서: (1, features, anchors)
            output = predictions[output_key].numpy()[0]  # 배치 차원 제거 -> (features, 8400)
            
            # 출력 차원 확인
            num_features, num_anchors = output.shape
            num_classes = num_features - 4  # 4는 bbox 좌표 (x, y, w, h)
            
            # 각 앵커 포인트에 대해 처리
            for i in range(num_anchors):
                anchor_data = output[:, i]  # 한 앵커의 모든 feature
                
                # 바운딩 박스 좌표 (중심점 x, y, 너비, 높이)
                cx, cy, w, h = anchor_data[:4]
                
                # 클래스 확률들
                class_probs = anchor_data[4:4+num_classes]
                
                # 최고 확률 클래스 찾기
                max_class_idx = np.argmax(class_probs)
                max_confidence = float(class_probs[max_class_idx])
                
                # 신뢰도 임계값 확인
                if max_confidence >= CONFIDENCE_THRESHOLD:
                    # 클래스 이름 가져오기
                    if max_class_idx < len(self.coco_classes):
                        class_name = self.coco_classes[max_class_idx]
                        
                        # VALID_CLASSES에 포함된 클래스만 처리
                        if class_name in VALID_CLASSES:
                            # 중심점과 크기를 좌상단/우하단 좌표로 변환
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            
                            # 좌표를 원본 이미지 크기에 맞게 스케일링
                            height, width = image_shape[:2]
                            x1_scaled = int(x1 * width / 640)
                            y1_scaled = int(y1 * height / 640)
                            x2_scaled = int(x2 * width / 640)
                            y2_scaled = int(y2 * height / 640)
                            
                            # 좌표 범위 제한
                            x1_scaled = max(0, min(x1_scaled, width))
                            y1_scaled = max(0, min(y1_scaled, height))
                            x2_scaled = max(0, min(x2_scaled, width))
                            y2_scaled = max(0, min(y2_scaled, height))
                            
                            detected_objects.append({
                                'class_name': class_name,
                                'class_id': max_class_idx,
                                'confidence': max_confidence,
                                'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                                'area': int((x2_scaled - x1_scaled) * (y2_scaled - y1_scaled))
                            })
                            
        except Exception as e:
            print(f"후처리 중 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
        
        # 신뢰도 순으로 정렬
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        # NMS (Non-Maximum Suppression) 적용
        return self._apply_nms(detected_objects, IOU_THRESHOLD)
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Non-Maximum Suppression 적용
        
        Args:
            detections: 검출된 객체 목록
            iou_threshold: IoU 임계값
            
        Returns:
            List[Dict]: NMS 적용된 객체 목록
        """
        if not detections:
            return []
        
        # 바운딩 박스와 신뢰도 추출
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # TensorFlow NMS 적용
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=len(detections), iou_threshold=iou_threshold
        )
        
        # 선택된 인덱스에 해당하는 검출 결과만 반환
        return [detections[i] for i in selected_indices.numpy()]
    
    def detect_objects(self, image_bytes: bytes) -> List[Dict]:
        """
        TensorFlow YOLO 모델로 객체 검출
        
        Args:
            image_bytes: 이미지 바이트 데이터
            
        Returns:
            List[Dict]: 검출된 객체 목록
        """
        try:
            # 원본 이미지 크기 저장
            nparr = np.frombuffer(image_bytes, np.uint8)
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if original_image is None:
                raise ValueError("이미지를 읽을 수 없습니다.")
            
            # 이미지 전처리
            input_tensor = self._preprocess_image(image_bytes)
            
            # 모델 추론 (입력 이름을 'images'로 지정)
            predictions = self.predict_fn(images=input_tensor)
            
            # 후처리
            detected_objects = self._postprocess_predictions(predictions, original_image.shape)
            
            return detected_objects
            
        except Exception as e:
            print(f"TensorFlow YOLO 객체 검출 실패: {e}")
            return []
    
    def detect_objects_with_basic_model(self, image_bytes: bytes) -> List[Dict]:
        """
        기본 TensorFlow YOLO11x 모델로 객체 검출 (검증용)
        
        Args:
            image_bytes: 이미지 바이트 데이터
            
        Returns:
            List[Dict]: 검출된 객체 목록
        """
        try:
            # 원본 이미지 크기 저장
            nparr = np.frombuffer(image_bytes, np.uint8)
            original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if original_image is None:
                raise ValueError("이미지를 읽을 수 없습니다.")
            
            # 이미지 전처리
            input_tensor = self._preprocess_image(image_bytes)
            
            # 기본 모델 추론 (입력 이름을 'images'로 지정)
            predictions = self.basic_predict_fn(images=input_tensor)
            
            # 후처리
            detected_objects = self._postprocess_predictions(predictions, original_image.shape)
            
            return detected_objects
            
        except Exception as e:
            print(f"기본 TensorFlow YOLO 모델 객체 검출 실패: {e}")
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
