#!/usr/bin/env python3
"""
다중 모델 객체 검출 비교 테스트
- 원본 이미지: train_tf 모델
- 노이즈 이미지: YOLO11x, EfficientDet, Faster R-CNN
- 디노이징 이미지: YOLO11x, EfficientDet, Faster R-CNN
"""

import sys
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import requests
from io import BytesIO

# 한글 폰트 설정
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Windows에서 한글 폰트 설정
try:
    # Windows에서 사용 가능한 한글 폰트 목록
    font_list = [
        'Malgun Gothic',      # 맑은 고딕 (Windows 기본)
        'NanumGothic',        # 나눔고딕
        'NanumBarunGothic',   # 나눔바른고딕
        'AppleGothic',        # 맥용
        'Noto Sans CJK KR',   # 구글 폰트
        'DejaVu Sans'         # 기본 폰트
    ]
    
    # 시스템에 설치된 폰트 목록 가져오기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 한글 폰트 찾기
    korean_font_found = False
    for font_name in font_list:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 한글 폰트 설정: {font_name}")
            korean_font_found = True
            break
    
    if not korean_font_found:
        # 폰트를 찾지 못한 경우 Windows 기본 폰트 강제 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️ 시스템 폰트를 찾을 수 없어 Malgun Gothic을 강제 설정합니다.")
        
except Exception as e:
    print(f"⚠️ 폰트 설정 중 오류: {e}")
    # 오류 발생 시 Windows 기본 폰트로 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

# quiz 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'quiz'))

# 한글 클래스명 매핑
KOREAN_CLASS_NAMES = {
    # COCO 80개 클래스 (YOLO11x)
    'person': '사람', 'bicycle': '자전거', 'car': '자동차', 'motorcycle': '오토바이', 'airplane': '비행기',
    'bus': '버스', 'train': '기차', 'truck': '트럭', 'boat': '보트', 'traffic light': '신호등',
    'fire hydrant': '소화전', 'stop sign': '정지 표지판', 'parking meter': '주차 미터기', 'bench': '벤치',
    'bird': '새', 'cat': '고양이', 'dog': '강아지', 'horse': '말', 'sheep': '양', 'cow': '소',
    'elephant': '코끼리', 'bear': '곰', 'zebra': '얼룩말', 'giraffe': '기린', 'backpack': '배낭',
    'umbrella': '우산', 'handbag': '핸드백', 'tie': '넥타이', 'suitcase': '여행가방', 'frisbee': '프리스비',
    'skis': '스키', 'snowboard': '스노보드', 'sports ball': '스포츠 공', 'kite': '연', 'baseball bat': '야구 배트',
    'baseball glove': '야구 글러브', 'skateboard': '스케이트보드', 'surfboard': '서핑보드', 'tennis racket': '테니스 라켓',
    'bottle': '병', 'wine glass': '와인잔', 'cup': '컵', 'fork': '포크', 'knife': '칼', 'spoon': '숟가락',
    'bowl': '그릇', 'banana': '바나나', 'apple': '사과', 'sandwich': '샌드위치', 'orange': '오렌지',
    'broccoli': '브로콜리', 'carrot': '당근', 'hot dog': '핫도그', 'pizza': '피자', 'donut': '도넛',
    'cake': '케이크', 'chair': '의자', 'couch': '소파', 'potted plant': '화분', 'bed': '침대',
    'dining table': '식탁', 'toilet': '화장실', 'tv': 'TV', 'laptop': '노트북', 'mouse': '마우스',
    'remote': '리모컨', 'keyboard': '키보드', 'cell phone': '휴대폰', 'microwave': '전자레인지',
    'oven': '오븐', 'toaster': '토스터', 'sink': '싱크대', 'refrigerator': '냉장고', 'book': '책',
    'clock': '시계', 'vase': '꽃병', 'scissors': '가위', 'teddy bear': '곰인형', 'hair drier': '헤어드라이어',
    'toothbrush': '칫솔',
    
    # 추가 일반적인 객체들
    'bag': '가방', 'box': '상자', 'table': '테이블', 'door': '문', 'window': '창문',
    'tree': '나무', 'flower': '꽃', 'leaf': '잎', 'grass': '잔디', 'sky': '하늘',
    'mountain': '산', 'river': '강', 'lake': '호수', 'sea': '바다', 'beach': '해변',
    'house': '집', 'building': '건물', 'bridge': '다리', 'road': '도로', 'street': '거리',
    'car': '자동차', 'truck': '트럭', 'bus': '버스', 'train': '기차', 'plane': '비행기',
    'ship': '배', 'boat': '보트', 'bike': '자전거', 'motorcycle': '오토바이'
}

class MultiModelDetector:
    """다중 모델 객체 검출기"""
    
    def __init__(self):
        """초기화"""
        print("=== 다중 모델 객체 검출기 초기화 ===")
        
        # 1. 기존 YOLO 모델들 로드
        from quiz.components.model_manager import ModelManager
        from quiz.components import YOLODetector
        
        model_manager = ModelManager()
        model_paths = model_manager.get_model_paths()
        
        self.yolo_detector = YOLODetector(model_paths['train_model'], model_paths['basic_model'])
        
        # 2. EfficientDet 모델 로드
        self.efficientdet_model = self._load_efficientdet()
        
        # 3. Faster R-CNN 모델 로드
        self.faster_rcnn_model = self._load_faster_rcnn()
        
        print("✓ 모든 모델 로드 완료\n")
    
    def _load_efficientdet(self):
        """EfficientDet 모델 로드 (TensorFlow Hub)"""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            # EfficientDet-D0 모델 로드 (COCO 사전학습)
            model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
            print("✓ EfficientDet 모델 로드 완료 (TensorFlow Hub)")
            return model
        except Exception as e:
            print(f"⚠️ EfficientDet 모델 로드 실패: {e}")
            # 대안으로 RetinaNet 사용 (EfficientDet와 유사한 구조)
            try:
                model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
                model.eval()
                print("✓ RetinaNet 모델 로드 완료 (EfficientDet 대신)")
                return model
            except Exception as e2:
                print(f"⚠️ RetinaNet 모델 로드도 실패: {e2}")
                return None
    
    def _load_faster_rcnn(self):
        """Faster R-CNN 모델 로드"""
        try:
            # Faster R-CNN 모델 로드 (COCO 데이터셋)
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            print("✓ Faster R-CNN 모델 로드 완료")
            return model
        except Exception as e:
            print(f"⚠️ Faster R-CNN 모델 로드 실패: {e}")
            return None
    
    def detect_with_train_tf(self, image_bytes):
        """train_tf 모델로 검출"""
        try:
            detected_objects = self.yolo_detector.detect_objects(image_bytes)
            return detected_objects
        except Exception as e:
            print(f"train_tf 검출 실패: {e}")
            return []
    
    def detect_with_yolo11x(self, image_array):
        """YOLO11x 모델로 검출"""
        try:
            # numpy 배열을 바이트로 변환
            if isinstance(image_array, np.ndarray):
                # BGR을 RGB로 변환
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
                
                # numpy 배열을 바이트로 인코딩
                success, encoded_image = cv2.imencode('.jpg', image_rgb)
                if success:
                    image_bytes = encoded_image.tobytes()
                else:
                    print("YOLO11x: 이미지 인코딩 실패")
                    return []
            else:
                # 이미 바이트인 경우
                image_bytes = image_array
            
            detected_objects = self.yolo_detector.detect_objects_with_basic_model(image_bytes)
            return detected_objects
        except Exception as e:
            print(f"YOLO11x 검출 실패: {e}")
            return []
    
    def detect_with_efficientdet(self, image_array):
        """EfficientDet 모델로 검출 (TensorFlow Hub)"""
        if self.efficientdet_model is None:
            return []
        
        try:
            import tensorflow as tf
            import numpy as np
            
            # 이미지 전처리
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # EfficientDet 입력 크기로 리사이즈 (512x512)
            image_resized = image_pil.resize((512, 512))
            img_np = np.array(image_resized)  # uint8 타입으로 유지
            img_tensor = tf.convert_to_tensor([img_np], dtype=tf.uint8)
            
            # 추론
            result = self.efficientdet_model(img_tensor)
            boxes = result["detection_boxes"].numpy()[0]  # [N, 4] (y1, x1, y2, x2)
            scores = result["detection_scores"].numpy()[0]  # [N]
            classes = result["detection_classes"].numpy()[0]  # [N]
            
            detected_objects = []
            h, w = image_array.shape[:2]
            
            # COCO 클래스명 매핑
            coco_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
                if score > 0.5:  # 신뢰도 임계값
                    # EfficientDet는 (y1, x1, y2, x2) 형식이므로 (x1, y1, x2, y2)로 변환
                    y1, x1, y2, x2 = box
                    
                    # 원본 이미지 크기로 좌표 변환
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                    
                    class_id = int(class_id)
                    class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"
                    
                    detected_objects.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(score),
                        'class_name': class_name,
                        'korean_name': class_name  # 영어 그대로 사용
                    })
            
            return detected_objects
            
        except Exception as e:
            print(f"EfficientDet 검출 실패: {e}")
            return []
    
    def detect_with_faster_rcnn(self, image_array):
        """Faster R-CNN 모델로 검출"""
        if self.faster_rcnn_model is None:
            return []
        
        try:
            # 이미지 전처리
            image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
            input_tensor = transform(image)
            
            with torch.no_grad():
                outputs = self.faster_rcnn_model([input_tensor])
            
            # 결과 파싱
            detected_objects = []
            boxes = outputs[0]['boxes'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            
            # COCO 클래스명 매핑
            coco_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score > 0.5:  # 신뢰도 임계값
                    x1, y1, x2, y2 = box
                    class_name = coco_classes[label] if label < len(coco_classes) else f"class_{label}"
                    
                    detected_objects.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(score),
                        'class_name': class_name,
                        'korean_name': class_name  # 영어 그대로 사용
                    })
            
            return detected_objects
            
        except Exception as e:
            print(f"Faster R-CNN 검출 실패: {e}")
            return []
    
    def get_korean_name(self, class_name):
        """영어 클래스명을 한글로 변환"""
        return KOREAN_CLASS_NAMES.get(class_name, class_name)

class MultiModelComparisonTest:
    """다중 모델 비교 테스트"""
    
    def __init__(self):
        """초기화"""
        self.detector = MultiModelDetector()
        from quiz.components.storage_manager import StorageManager
        from quiz.components.image_handler import ImageHandler
        from quiz.components.image_preprocessor import ImagePreprocessor
        
        self.storage_manager = StorageManager()
        self.image_handler = ImageHandler()
        self.preprocessor = ImagePreprocessor()
    
    def run_comparison_test(self, difficulty='medium'):
        """비교 테스트 실행"""
        print(f"=== 다중 모델 비교 테스트 ({difficulty.upper()} 난이도) ===")
        
        # 1. 원본 이미지 가져오기
        image_key, image_bytes = self.storage_manager.get_random_original_image("images")
        print(f"원본 이미지: {image_key}")
        
        # 2. 원본 이미지에서 train_tf 모델로 검출
        print("원본 이미지에서 train_tf 모델 검출 중...")
        original_detections = self.detector.detect_with_train_tf(image_bytes)
        
        # 3. 노이즈 이미지 생성
        intensity, alpha = self.image_handler.get_random_noise_params(difficulty)
        processed_image_array = self.image_handler.process_image_with_noise(image_bytes, intensity=intensity, alpha=alpha)
        
        # 4. 디노이징 이미지 생성
        denoised_image_array = self.preprocessor.adaptiveDenoising(processed_image_array)
        
        # 5. 각 이미지에서 다양한 모델로 검출
        print("노이즈 이미지에서 다중 모델 검출 중...")
        noisy_yolo_detections = self.detector.detect_with_yolo11x(processed_image_array)
        noisy_efficientdet_detections = self.detector.detect_with_efficientdet(processed_image_array)
        noisy_faster_rcnn_detections = self.detector.detect_with_faster_rcnn(processed_image_array)
        
        print("디노이징 이미지에서 다중 모델 검출 중...")
        denoised_yolo_detections = self.detector.detect_with_yolo11x(denoised_image_array)
        denoised_efficientdet_detections = self.detector.detect_with_efficientdet(denoised_image_array)
        denoised_faster_rcnn_detections = self.detector.detect_with_faster_rcnn(denoised_image_array)
        
        # 6. 결과 시각화
        self.create_comparison_visualization(
            image_bytes, processed_image_array, denoised_image_array,
            original_detections, noisy_yolo_detections, noisy_efficientdet_detections, noisy_faster_rcnn_detections,
            denoised_yolo_detections, denoised_efficientdet_detections, denoised_faster_rcnn_detections,
            intensity, alpha, difficulty
        )
        
        return {
            'original_detections': original_detections,
            'noisy_yolo': noisy_yolo_detections,
            'noisy_efficientdet': noisy_efficientdet_detections,
            'noisy_faster_rcnn': noisy_faster_rcnn_detections,
            'denoised_yolo': denoised_yolo_detections,
            'denoised_efficientdet': denoised_efficientdet_detections,
            'denoised_faster_rcnn': denoised_faster_rcnn_detections
        }
    
    def create_comparison_visualization(self, original_bytes, noisy_array, denoised_array,
                                      original_detections, noisy_yolo, noisy_efficientdet, noisy_faster_rcnn,
                                      denoised_yolo, denoised_efficientdet, denoised_faster_rcnn,
                                      intensity, alpha, difficulty):
        """비교 결과 시각화"""
        
        # 원본 이미지 처리
        nparr = np.frombuffer(original_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # 원본 이미지 크기 저장 (bbox 좌표 변환용)
        original_height, original_width = original_image.shape[:2]
        
        # 노이즈 이미지
        noisy_image = cv2.cvtColor(noisy_array, cv2.COLOR_BGR2RGB)
        
        # 디노이징 이미지
        denoised_image = cv2.cvtColor(denoised_array, cv2.COLOR_BGR2RGB)
        
        # 모든 이미지를 같은 크기로 맞춤 (노이즈 이미지 크기 기준)
        target_height, target_width = noisy_image.shape[:2]
        original_image = cv2.resize(original_image, (target_width, target_height))
        denoised_image = cv2.resize(denoised_image, (target_width, target_height))
        
        # 원본 bbox 좌표를 리사이즈된 크기에 맞게 변환
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        for obj in original_detections:
            x1, y1, x2, y2 = obj['bbox']
            obj['bbox'] = [
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ]
        
        # 원본 검출 결과에서 클래스명 추출 (비교용)
        original_classes = set()
        for obj in original_detections:
            class_name = obj.get('korean_name', obj['class_name'])
            original_classes.add(class_name)
        
        def draw_detections(image, detections, title_prefix="", max_detections=3, is_original=False):
            """이미지에 검출 결과 그리기"""
            result_image = image.copy()
            
            if detections:
                for i, obj in enumerate(detections[:max_detections]):
                    x1, y1, x2, y2 = obj['bbox']
                    confidence = obj['confidence']
                    class_name = obj.get('korean_name', obj['class_name'])
                    
                    # bbox 좌표 클램핑
                    h, w = result_image.shape[:2]
                    x1 = max(0, min(int(x1), w-1))
                    y1 = max(0, min(int(y1), h-1))
                    x2 = max(0, min(int(x2), w-1))
                    y2 = max(0, min(int(y2), h-1))
                    
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue
                    
                    # bbox 색상 결정
                    if is_original:
                        # 원본 이미지는 기본 색상
                        color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.6 else (255, 0, 0)
                    else:
                        # 다른 이미지들은 원본과 비교하여 색상 결정
                        if class_name in original_classes:
                            # 원본에서도 검출된 객체는 빨간색으로 강조
                            color = (0, 0, 255)  # 빨간색 (BGR)
                        else:
                            # 원본에서 검출되지 않은 객체는 기본 색상
                            color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.6 else (255, 0, 0)
                    
                    # bbox 그리기
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 라벨 텍스트
                    label = f"{class_name}: {confidence:.2f}"
                    if not is_original and class_name in original_classes:
                        label += " [MATCH]"  # 원본과 일치하는 경우 표시
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    # 라벨 배경
                    cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # 라벨 텍스트
                    cv2.putText(result_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return result_image
        
        # 그래프 설정 (3x3 그리드)
        fig = plt.figure(figsize=(18, 15))
        
        # 첫 번째 행: 노이즈 이미지들
        # 노이즈 이미지 - YOLO11x
        ax1 = plt.subplot(3, 3, 1)
        noisy_yolo_with_bbox = draw_detections(noisy_image, noisy_yolo, is_original=False)
        ax1.imshow(noisy_yolo_with_bbox)
        ax1.set_title(f'노이즈 이미지 (YOLO11x)\n검출: {len(noisy_yolo)}개', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 노이즈 이미지 - EfficientDet
        ax2 = plt.subplot(3, 3, 2)
        noisy_eff_with_bbox = draw_detections(noisy_image, noisy_efficientdet, is_original=False)
        ax2.imshow(noisy_eff_with_bbox)
        ax2.set_title(f'노이즈 이미지 (EfficientDet)\n검출: {len(noisy_efficientdet)}개', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 노이즈 이미지 - Faster R-CNN
        ax3 = plt.subplot(3, 3, 3)
        noisy_rcnn_with_bbox = draw_detections(noisy_image, noisy_faster_rcnn, is_original=False)
        ax3.imshow(noisy_rcnn_with_bbox)
        ax3.set_title(f'노이즈 이미지 (Faster R-CNN)\n검출: {len(noisy_faster_rcnn)}개', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 두 번째 행: 디노이징 이미지들
        # 디노이징 이미지 - YOLO11x
        ax4 = plt.subplot(3, 3, 4)
        denoised_yolo_with_bbox = draw_detections(denoised_image, denoised_yolo, is_original=False)
        ax4.imshow(denoised_yolo_with_bbox)
        ax4.set_title(f'디노이징 이미지 (YOLO11x)\n검출: {len(denoised_yolo)}개', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 디노이징 이미지 - EfficientDet
        ax5 = plt.subplot(3, 3, 5)
        denoised_eff_with_bbox = draw_detections(denoised_image, denoised_efficientdet, is_original=False)
        ax5.imshow(denoised_eff_with_bbox)
        ax5.set_title(f'디노이징 이미지 (EfficientDet)\n검출: {len(denoised_efficientdet)}개', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # 디노이징 이미지 - Faster R-CNN
        ax6 = plt.subplot(3, 3, 6)
        denoised_rcnn_with_bbox = draw_detections(denoised_image, denoised_faster_rcnn, is_original=False)
        ax6.imshow(denoised_rcnn_with_bbox)
        ax6.set_title(f'디노이징 이미지 (Faster R-CNN)\n검출: {len(denoised_faster_rcnn)}개', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        # 세 번째 행: 원본 이미지 + 차트 + 정보
        # 원본 이미지 (train_tf)
        ax7 = plt.subplot(3, 3, 7)
        original_with_bbox = draw_detections(original_image, original_detections, is_original=True)
        ax7.imshow(original_with_bbox)
        ax7.set_title(f'원본 이미지 (train_tf)\n검출: {len(original_detections)}개', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # 검출 결과 비교 차트
        ax8 = plt.subplot(3, 3, 8)
        models = ['train_tf\n(원본)', 'YOLO11x\n(노이즈)', 'EfficientDet\n(노이즈)', 'Faster R-CNN\n(노이즈)',
                 'YOLO11x\n(디노이징)', 'EfficientDet\n(디노이징)', 'Faster R-CNN\n(디노이징)']
        detection_counts = [len(original_detections), len(noisy_yolo), len(noisy_efficientdet), len(noisy_faster_rcnn),
                           len(denoised_yolo), len(denoised_efficientdet), len(denoised_faster_rcnn)]
        
        colors = ['blue', 'red', 'green', 'orange', 'red', 'green', 'orange']
        bars = ax8.bar(range(len(models)), detection_counts, color=colors, alpha=0.7)
        ax8.set_xlabel('모델별 검출 결과')
        ax8.set_ylabel('검출된 객체 수')
        ax8.set_title('모델별 검출 객체 수 비교')
        ax8.set_xticks(range(len(models)))
        ax8.set_xticklabels(models, rotation=45, ha='right')
        ax8.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, count in zip(bars, detection_counts):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 상세 정보 텍스트
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # 검출된 객체들 정보 수집 (색상 정보 포함)
        def get_detection_info_with_colors(detections, model_name, is_original=False):
            if not detections:
                return f"{model_name}: 검출 실패", []
            
            objects = []
            colors = []
            for obj in detections[:3]:  # 최대 3개만
                korean_name = obj.get('korean_name', obj['class_name'])
                confidence = obj['confidence']
                
                # 원본과 일치하는지 확인
                if not is_original and korean_name in original_classes:
                    objects.append(f"{korean_name}({confidence:.2f})[MATCH]")
                    colors.append('red')
                else:
                    objects.append(f"{korean_name}({confidence:.2f})")
                    colors.append('black')
            
            return f"{model_name}: {', '.join(objects)}", colors
        
        # 기본 정보 텍스트
        basic_info = f"""다중 모델 검출 결과 비교

난이도: {difficulty.upper()}
노이즈 강도: {intensity*100:.0f}%
알파 블랜드: {alpha*100:.0f}%

검출 결과:"""
        
        # 기본 정보 표시
        ax9.text(0.05, 0.95, basic_info, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', color='black')
        
        # 검출 결과를 간단하게 표시
        y_pos = 0.70
        detection_results = [
            (original_detections, 'train_tf (원본)', True),
            (noisy_yolo, 'YOLO11x (노이즈)', False),
            (noisy_efficientdet, 'EfficientDet (노이즈)', False),
            (noisy_faster_rcnn, 'Faster R-CNN (노이즈)', False),
            (denoised_yolo, 'YOLO11x (디노이징)', False),
            (denoised_efficientdet, 'EfficientDet (디노이징)', False),
            (denoised_faster_rcnn, 'Faster R-CNN (디노이징)', False)
        ]
        
        for detections, model_name, is_original in detection_results:
            if not detections:
                text = f"{model_name}: 검출 실패"
                ax9.text(0.05, y_pos, text, transform=ax9.transAxes, 
                        fontsize=8, verticalalignment='top', color='black')
            else:
                # 모델명 표시
                ax9.text(0.05, y_pos, f"{model_name}: ", transform=ax9.transAxes, 
                        fontsize=8, verticalalignment='top', color='black', fontweight='bold')
                
                # 객체 정보를 색상과 함께 표시
                x_offset = 0.25
                for i, obj in enumerate(detections[:3]):  # 최대 3개만
                    korean_name = obj.get('korean_name', obj['class_name'])
                    confidence = obj['confidence']
                    
                    # 원본과 일치하는지 확인
                    if not is_original and korean_name in original_classes:
                        obj_text = f"{korean_name}({confidence:.2f})[MATCH]"
                        color = 'red'
                    else:
                        obj_text = f"{korean_name}({confidence:.2f})"
                        color = 'black'
                    
                    # 쉼표 추가 (첫 번째가 아닌 경우)
                    if i > 0:
                        ax9.text(0.05 + x_offset, y_pos, ", ", transform=ax9.transAxes, 
                                fontsize=8, verticalalignment='top', color='black')
                        x_offset += 0.02
                    
                    # 객체 텍스트 표시
                    ax9.text(0.05 + x_offset, y_pos, obj_text, transform=ax9.transAxes, 
                            fontsize=8, verticalalignment='top', color=color)
                    x_offset += len(obj_text) * 0.008
            
            y_pos -= 0.08
        
        # 모델 정보
        model_info = """
모델 정보:
• train_tf: 16개 클래스 (파인튜닝)
• YOLO11x: 80개 클래스 (COCO)
• EfficientDet: 80개 클래스 (COCO)
• Faster R-CNN: 80개 클래스 (COCO)"""
        
        ax9.text(0.05, y_pos, model_info, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', color='black')
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"multi_model_detection_comparison_{timestamp}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 다중 모델 비교 결과 저장: {output_path}")

def main():
    """메인 함수"""
    print("다중 모델 객체 검출 비교 테스트 시작")
    
    test = MultiModelComparisonTest()
    results = test.run_comparison_test('medium')
    
    print("\n🎉 다중 모델 비교 테스트 완료!")
    print(f"원본 검출: {len(results['original_detections'])}개")
    print(f"노이즈 YOLO11x: {len(results['noisy_yolo'])}개")
    print(f"노이즈 EfficientDet: {len(results['noisy_efficientdet'])}개")
    print(f"노이즈 Faster R-CNN: {len(results['noisy_faster_rcnn'])}개")
    print(f"디노이징 YOLO11x: {len(results['denoised_yolo'])}개")
    print(f"디노이징 EfficientDet: {len(results['denoised_efficientdet'])}개")
    print(f"디노이징 Faster R-CNN: {len(results['denoised_faster_rcnn'])}개")

if __name__ == "__main__":
    main()
