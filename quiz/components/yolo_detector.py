"""
YOLO 객체 검출 모듈 (TensorFlow 버전)
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import List, Dict
from config.settings import VALID_CLASSES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from .image_preprocessor import ImagePreprocessor


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
        
        # GPU 설정을 모델 로딩 전에 먼저 수행
        self._configure_gpu()
        
        # 모델 경로 검증
        self._validate_model_paths(model_path, basic_model_path)
        
        # TensorFlow YOLO 모델 로딩
        try:
            # SavedModel 형식으로 로딩 (로컬 디바이스에서 로딩)
            load_options = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost'
            )
            self.model = tf.saved_model.load(model_path, options=load_options)
            self.predict_fn = self.model.signatures['serving_default']
            print(f"✓ TensorFlow YOLO 모델 로딩 성공: {model_path}")
        except Exception as e:
            print(f"✗ TensorFlow YOLO 모델 로딩 실패: {e}")
            raise
            
        try:
            # 기본 TensorFlow YOLO 모델 로딩 (검증용)
            load_options = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost'
            )
            self.basic_model = tf.saved_model.load(basic_model_path, options=load_options)
            self.basic_predict_fn = self.basic_model.signatures['serving_default']
            print(f"✓ 기본 TensorFlow YOLO 모델 로딩 성공: {basic_model_path}")
        except Exception as e:
            print(f"✗ 기본 TensorFlow YOLO 모델 로딩 실패: {e}")
            raise
        
        # train_tf 모델의 실제 16개 클래스 매핑 (커스텀 학습 모델)
        self.train_tf_classes = [
            'backpack', 'bear', 'bed', 'bird', 'boat', 'bottle', 'car', 'cat',
            'chair', 'clock', 'cow', 'cup', 'dog', 'elephant', 'refrigerator', 'sheep'
        ]
        
        # COCO 클래스 이름 매핑 (yolo11x_tf 모델용 - YOLO 표준 80개 클래스)
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
        
        # 디노이징을 위한 ImagePreprocessor 인스턴스
        self.image_preprocessor = ImagePreprocessor()
        
        print("TensorFlow YOLO 검출기 초기화 완료!")
    
    def _configure_gpu(self):
        """
        GPU 메모리 설정 (모델 로딩 전에 실행되어야 함)
        """
        try:
            # GPU 사용 가능 여부 확인
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✓ GPU 사용 가능: {len(gpus)}개")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu}")
                
                # GPU 메모리 성장 설정 (물리적 장치 초기화 전에 설정)
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✓ GPU 메모리 성장 설정 완료")
            else:
                print("✓ CPU 모드로 실행")
                
        except RuntimeError as e:
            # 이미 초기화된 경우의 오류 메시지를 더 명확하게 표시
            if "Physical devices cannot be modified" in str(e):
                print("⚠ GPU가 이미 초기화되어 메모리 성장 설정을 건너뜁니다.")
                print("  (이는 정상적인 동작이며 성능에 영향을 주지 않습니다)")
            else:
                print(f"GPU 설정 중 오류: {e}")
        except Exception as e:
            print(f"GPU 설정 중 예상치 못한 오류: {e}")
    
    def _preprocess_image(self, image_bytes: bytes):
        """
        이미지 전처리 (비율 유지 + 패딩)
        
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
        
        # 비율 유지하면서 640x640에 맞게 리사이즈 + 패딩
        image = self._resize_with_padding(image, (640, 640))
        
        # 정규화 (0-255 -> 0-1)
        image = image.astype(np.float32) / 255.0
        
        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)
        
        return tf.constant(image)
    
    def _resize_with_padding(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        비율을 유지하면서 리사이즈 후 패딩 적용
        
        Args:
            image: 입력 이미지
            target_size: 목표 크기 (width, height)
            
        Returns:
            np.ndarray: 리사이즈 및 패딩된 이미지
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # 비율을 유지하면서 리사이즈할 크기 계산
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h))
        
        # 패딩 계산
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        # 패딩 적용 (회색으로)
        padded = cv2.copyMakeBorder(
            resized, 
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, 
            value=(114, 114, 114)  # YOLO 표준 패딩 색상
        )
        
        return padded
    
    def _convert_bbox_coordinates(self, x1: float, y1: float, x2: float, y2: float,
                                orig_width: int, orig_height: int,
                                model_width: int, model_height: int) -> tuple:
        """
        패딩을 고려한 정확한 좌표 변환
        
        Args:
            x1, y1, x2, y2: YOLO 모델 출력 좌표 (640x640 기준)
            orig_width, orig_height: 원본 이미지 크기
            model_width, model_height: 모델 입력 크기 (640x640)
            
        Returns:
            tuple: 원본 이미지 좌표계로 변환된 (x1, y1, x2, y2)
        """
        # 원본 이미지의 비율 계산
        scale = min(model_width / orig_width, model_height / orig_height)
        
        # 리사이즈된 이미지 크기
        resized_w = int(orig_width * scale)
        resized_h = int(orig_height * scale)
        
        # 패딩 계산
        pad_left = (model_width - resized_w) // 2
        pad_top = (model_height - resized_h) // 2
        
        # 패딩을 제거한 좌표 계산
        x1_no_pad = x1 - pad_left
        y1_no_pad = y1 - pad_top
        x2_no_pad = x2 - pad_left
        y2_no_pad = y2 - pad_top
        
        # 원본 이미지 좌표계로 변환
        x1_orig = int(x1_no_pad / scale)
        y1_orig = int(y1_no_pad / scale)
        x2_orig = int(x2_no_pad / scale)
        y2_orig = int(y2_no_pad / scale)
        
        return x1_orig, y1_orig, x2_orig, y2_orig
    
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
                    # train_tf 모델용 클래스 이름 가져오기 (16개 클래스)
                    if max_class_idx < len(self.train_tf_classes):
                        class_name = self.train_tf_classes[max_class_idx]
                        
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
    
    def _postprocess_predictions_basic(self, predictions, image_shape) -> List[Dict]:
        """
        yolo11x_tf 모델의 예측 결과 후처리 (80개 COCO 클래스)
        
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
                    # yolo11x_tf 모델용 클래스 이름 가져오기 (80개 COCO 클래스)
                    if max_class_idx < len(self.coco_classes):
                        class_name = self.coco_classes[max_class_idx]
                        
                        # VALID_CLASSES에 포함된 클래스만 처리
                        if class_name in VALID_CLASSES:
                            # 중심점과 크기를 좌상단/우하단 좌표로 변환
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            
                            # 패딩을 고려한 좌표 변환
                            orig_h, orig_w = image_shape[:2]
                            x1_orig, y1_orig, x2_orig, y2_orig = self._convert_bbox_coordinates(
                                x1, y1, x2, y2, orig_w, orig_h, 640, 640
                            )
                            
                            # 좌표값 유효성 검사
                            if (x2_orig > x1_orig and y2_orig > y1_orig and 
                                x1_orig >= 0 and y1_orig >= 0 and 
                                x2_orig <= orig_w and y2_orig <= orig_h):
                                
                                # 영역 계산
                                area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
                                
                                detected_objects.append({
                                    'class_name': class_name,
                                    'class_id': max_class_idx,
                                    'confidence': max_confidence,
                                    'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                                    'area': area
                                })
        
        except Exception as e:
            print(f"yolo11x_tf 모델 후처리 오류: {e}")
            import traceback
            traceback.print_exc()
        
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
            
            # 후처리 (yolo11x_tf 모델용 - 80개 COCO 클래스)
            detected_objects = self._postprocess_predictions_basic(predictions, original_image.shape)
            
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
    
    def validate_with_hybrid_denoising(self, processed_image_bytes: bytes, current_answer: Dict, 
                                     denoise_strength: str = 'medium') -> Dict:
        """
        하이브리드 디노이징을 적용한 후 기본 모델로 검증
        
        Args:
            processed_image_bytes: 노이즈 처리된 이미지 바이트 데이터
            current_answer: 현재 모델이 선택한 정답 객체
            denoise_strength: 디노이징 강도 ('light', 'medium', 'strong')
            
        Returns:
            Dict: 검증 결과 정보
        """
        try:
            print(f" 하이브리드 디노이징 검증 시작 (강도: {denoise_strength})")
            
            # 1. 노이즈 이미지를 numpy 배열로 변환
            nparr = np.frombuffer(processed_image_bytes, np.uint8)
            noisy_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if noisy_image is None:
                raise ValueError("노이즈 이미지를 읽을 수 없습니다.")
            
            # 2. 하이브리드 디노이징 적용
            denoised_image = self.image_preprocessor.hybridDenoising(noisy_image, denoise_strength)
            
            # 3. 디노이징된 이미지를 바이트로 변환
            success, encoded_image = cv2.imencode('.jpg', denoised_image)
            if not success:
                raise ValueError("디노이징된 이미지 인코딩에 실패했습니다.")
            denoised_image_bytes = encoded_image.tobytes()
            
            # 4. 기본 모델로 디노이징된 이미지 검출
            denoised_detected_objects = self.detect_objects_with_basic_model(denoised_image_bytes)
            
            # 5. 원본 노이즈 이미지로도 검출 (비교용)
            noisy_detected_objects = self.detect_objects_with_basic_model(processed_image_bytes)
            
            # 6. 결과 분석
            result = {
                'is_different_from_current': False,
                'denoising_improved': False,
                'current_answer': current_answer,
                'noisy_detection': None,
                'denoised_detection': None,
                'confidence_improvement': 0.0,
                'denoise_strength': denoise_strength
            }
            
            # 노이즈 이미지 검출 결과
            if noisy_detected_objects:
                noisy_best = max(noisy_detected_objects, key=lambda x: x['confidence'])
                result['noisy_detection'] = noisy_best
            
            # 디노이징 이미지 검출 결과
            if denoised_detected_objects:
                denoised_best = max(denoised_detected_objects, key=lambda x: x['confidence'])
                result['denoised_detection'] = denoised_best
                
                # 신뢰도 임계값 확인
                if denoised_best['confidence'] >= CONFIDENCE_THRESHOLD:
                    # 현재 모델과 다른 결과인지 확인
                    if denoised_best['class_name'] != current_answer['class_name']:
                        result['is_different_from_current'] = True
                    
                    # 디노이징이 개선되었는지 확인
                    if result['noisy_detection']:
                        confidence_diff = denoised_best['confidence'] - result['noisy_detection']['confidence']
                        result['confidence_improvement'] = confidence_diff
                        if confidence_diff > 0.05:  # 5% 이상 개선
                            result['denoising_improved'] = True
                
                # 결과 출력
                print(f" 디노이징 검증 결과:")
                print(f" - 현재 모델 정답: {current_answer['class_name']} (신뢰도: {current_answer['confidence']:.3f})")
                
                if result['noisy_detection']:
                    print(f" - 노이즈 이미지 검출: {result['noisy_detection']['class_name']} "
                          f"(신뢰도: {result['noisy_detection']['confidence']:.3f})")
                
                print(f" - 디노이징 이미지 검출: {denoised_best['class_name']} "
                      f"(신뢰도: {denoised_best['confidence']:.3f})")
                print(f" - 신뢰도 개선: {result['confidence_improvement']:+.3f}")
                print(f" - 디노이징 효과: {'있음' if result['denoising_improved'] else '없음'}")
                print(f" - 현재 모델과 다른 결과: {'예' if result['is_different_from_current'] else '아니오'}")
            
            return result
            
        except Exception as e:
            print(f"하이브리드 디노이징 검증 중 오류: {e}")
            return {
                'is_different_from_current': True,
                'denoising_improved': False,
                'current_answer': current_answer,
                'noisy_detection': None,
                'denoised_detection': None,
                'confidence_improvement': 0.0,
                'denoise_strength': denoise_strength,
                'error': str(e)
            }
    
    def validate_with_adaptive_denoising(self, processed_image_bytes: bytes, current_answer: Dict) -> Dict:
        """
        적응형 디노이징을 적용한 후 기본 모델로 검증
        
        Args:
            processed_image_bytes: 노이즈 처리된 이미지 바이트 데이터
            current_answer: 현재 모델이 선택한 정답 객체
            
        Returns:
            Dict: 검증 결과 정보
        """
        try:
            print(f" 적응형 디노이징 검증 시작")
            
            # 1. 노이즈 이미지를 numpy 배열로 변환
            nparr = np.frombuffer(processed_image_bytes, np.uint8)
            noisy_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if noisy_image is None:
                raise ValueError("노이즈 이미지를 읽을 수 없습니다.")
            
            # 2. 적응형 디노이징 적용
            denoised_image = self.image_preprocessor.adaptiveDenoising(noisy_image)
            
            # 3. 디노이징된 이미지를 바이트로 변환
            success, encoded_image = cv2.imencode('.jpg', denoised_image)
            if not success:
                raise ValueError("디노이징된 이미지 인코딩에 실패했습니다.")
            denoised_image_bytes = encoded_image.tobytes()
            
            # 4. 하이브리드 디노이징 검증과 동일한 로직
            return self.validate_with_hybrid_denoising(denoised_image_bytes, current_answer, 'adaptive')
            
        except Exception as e:
            print(f"적응형 디노이징 검증 중 오류: {e}")
            return {
                'is_different_from_current': True,
                'denoising_improved': False,
                'current_answer': current_answer,
                'noisy_detection': None,
                'denoised_detection': None,
                'confidence_improvement': 0.0,
                'denoise_strength': 'adaptive',
                'error': str(e)
            }
    
    def comprehensive_validation(self, processed_image_bytes: bytes, current_answer: Dict) -> Dict:
        """
        종합 검증: 기존 검증 + 하이브리드 디노이징 검증 비교
        
        Args:
            processed_image_bytes: 노이즈 처리된 이미지 바이트 데이터
            current_answer: 현재 모델이 선택한 정답 객체
            
        Returns:
            Dict: 종합 검증 결과
        """
        print("\n=== 종합 검증 시작 ===")
        
        # 1. 기존 검증 (노이즈 이미지 그대로)
        print("\n[1단계] 기존 검증 (노이즈 이미지)")
        basic_validation = self.validate_with_basic_model(processed_image_bytes, current_answer)
        
        # 2. 하이브리드 디노이징 검증 (여러 강도)
        denoising_results = {}
        denoise_strengths = ['light', 'medium', 'strong']
        
        for strength in denoise_strengths:
            print(f"\n[2단계] 하이브리드 디노이징 검증 - {strength}")
            denoising_results[strength] = self.validate_with_hybrid_denoising(
                processed_image_bytes, current_answer, strength
            )
        
        # 3. 적응형 디노이징 검증
        print(f"\n[3단계] 적응형 디노이징 검증")
        adaptive_result = self.validate_with_adaptive_denoising(processed_image_bytes, current_answer)
        denoising_results['adaptive'] = adaptive_result
        
        # 4. 결과 분석
        analysis = self._analyze_comprehensive_results(
            basic_validation, denoising_results, current_answer
        )
        
        # 5. 종합 결과 반환
        comprehensive_result = {
            'basic_validation': basic_validation,
            'denoising_results': denoising_results,
            'analysis': analysis,
            'recommendation': self._get_validation_recommendation(analysis)
        }
        
        print("\n=== 종합 검증 완료 ===")
        self._print_comprehensive_summary(comprehensive_result)
        
        return comprehensive_result
    
    def _analyze_comprehensive_results(self, basic_validation: bool, denoising_results: Dict, 
                                     current_answer: Dict) -> Dict:
        """종합 검증 결과 분석"""
        analysis = {
            'basic_different': basic_validation,
            'denoising_different_count': 0,
            'best_denoising_method': None,
            'max_confidence_improvement': 0.0,
            'denoising_effectiveness': {},
            'consistency_score': 0.0
        }
        
        # 디노이징 결과 분석
        different_methods = []
        improvements = []
        
        for method, result in denoising_results.items():
            if result.get('is_different_from_current', False):
                analysis['denoising_different_count'] += 1
                different_methods.append(method)
            
            improvement = result.get('confidence_improvement', 0.0)
            improvements.append(improvement)
            analysis['denoising_effectiveness'][method] = {
                'improved': result.get('denoising_improved', False),
                'confidence_improvement': improvement,
                'different_from_current': result.get('is_different_from_current', False)
            }
        
        # 최고 개선 방법 찾기
        if improvements:
            max_improvement = max(improvements)
            analysis['max_confidence_improvement'] = max_improvement
            
            for method, result in denoising_results.items():
                if result.get('confidence_improvement', 0.0) == max_improvement:
                    analysis['best_denoising_method'] = method
                    break
        
        # 일관성 점수 계산 (0-100)
        total_methods = len(denoising_results) + 1  # +1 for basic
        different_count = analysis['denoising_different_count'] + (1 if basic_validation else 0)
        analysis['consistency_score'] = (different_count / total_methods) * 100
        
        return analysis
    
    def _get_validation_recommendation(self, analysis: Dict) -> str:
        """검증 결과에 따른 권장사항"""
        consistency = analysis['consistency_score']
        best_method = analysis['best_denoising_method']
        max_improvement = analysis['max_confidence_improvement']
        
        if consistency >= 80:
            return f"높은 일관성 ({consistency:.1f}%) - 퀴즈 난이도가 적절합니다."
        elif consistency >= 50:
            if best_method and max_improvement > 0.1:
                return f"중간 일관성 ({consistency:.1f}%) - {best_method} 디노이징으로 {max_improvement:.3f} 개선됨"
            else:
                return f"중간 일관성 ({consistency:.1f}%) - 추가 노이즈 조정 권장"
        else:
            return f"낮은 일관성 ({consistency:.1f}%) - 노이즈 파라미터 재조정 필요"
    
    def _print_comprehensive_summary(self, result: Dict):
        """종합 검증 결과 요약 출력"""
        print("\n=== 종합 검증 결과 요약 ===")
        
        analysis = result['analysis']
        print(f"기본 검증 결과: {'다름' if analysis['basic_different'] else '같음'}")
        print(f"디노이징 검증 중 다른 결과: {analysis['denoising_different_count']}/4개 방법")
        print(f"최고 개선 방법: {analysis['best_denoising_method']}")
        print(f"최대 신뢰도 개선: {analysis['max_confidence_improvement']:+.3f}")
        print(f"일관성 점수: {analysis['consistency_score']:.1f}%")
        print(f"권장사항: {result['recommendation']}")
        
        print("\n=== 디노이징 효과 세부사항 ===")
        for method, effectiveness in analysis['denoising_effectiveness'].items():
            print(f"{method:>10}: 개선={effectiveness['improved']}, "
                  f"신뢰도={effectiveness['confidence_improvement']:+.3f}, "
                  f"다름={effectiveness['different_from_current']}")
        print("=" * 35)
    
    def _validate_model_paths(self, model_path: str, basic_model_path: str):
        """
        모델 경로와 필요한 파일들이 존재하는지 검증
        
        Args:
            model_path: 훈련된 모델 경로
            basic_model_path: 기본 모델 경로
        """
        import os
        
        print(f"모델 경로 검증 중...")
        print(f"  - 훈련된 모델: {model_path}")
        print(f"  - 기본 모델: {basic_model_path}")
        
        # 훈련된 모델 경로 검증
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"훈련된 모델 경로가 존재하지 않습니다: {model_path}")
        
        # 기본 모델 경로 검증
        if not os.path.exists(basic_model_path):
            raise FileNotFoundError(f"기본 모델 경로가 존재하지 않습니다: {basic_model_path}")
        
        # SavedModel 파일 검증
        saved_model_files = [
            os.path.join(model_path, "saved_model.pb"),
            os.path.join(basic_model_path, "saved_model.pb")
        ]
        
        for file_path in saved_model_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"SavedModel 파일이 존재하지 않습니다: {file_path}")
        
        print("✓ 모델 경로 검증 완료!")
