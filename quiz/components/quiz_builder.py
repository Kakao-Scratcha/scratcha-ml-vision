"""
퀴즈 옵션 생성 및 정답 선택 모듈
"""

import random
from typing import List, Dict, Optional
from config.settings import (
    CONFIDENCE_THRESHOLD, 
    IOU_THRESHOLD, 
    VALID_CLASSES, 
    CLASS_NAME_TO_KOREAN, 
    SIMILAR_OBJECT_PAIRS
)


class QuizBuilder:
    """퀴즈 생성 및 정답 선택 클래스"""
    
    def __init__(self):
        """초기화"""
        print("QuizBuilder 초기화 완료")
    
    def select_correct_answer(self, detected_objects: List[Dict], 
                            confidence_threshold: float = CONFIDENCE_THRESHOLD, 
                            iou_threshold: float = IOU_THRESHOLD) -> Optional[Dict]:
        """
        검출된 객체 중 조건을 만족하는 가장 정확도 높은 것을 정답으로 선택
        
        Args:
            detected_objects: 검출된 객체 목록
            confidence_threshold: 신뢰도 임계값 (기본값: 0.6)
            iou_threshold: IoU 임계값 (기본값: 0.5)
            
        Returns:
            Optional[Dict]: 정답 객체 (없으면 None)
        """
        if not detected_objects:
            return None
        
        # 조건을 만족하는 객체들 필터링
        valid_objects = []
        
        for obj in detected_objects:
            # 신뢰도 조건 확인
            if obj['confidence'] < confidence_threshold:
                continue
            
            # IoU 조건 확인 (다른 객체들과의 겹침 정도)
            max_iou = 0.0
            for other_obj in detected_objects:
                if obj != other_obj:
                    iou = self.calculate_iou(obj['bbox'], other_obj['bbox'])
                    max_iou = max(max_iou, iou)
            
            # IoU가 임계값보다 낮으면 (겹침이 적으면) 유효한 객체로 간주
            if max_iou < iou_threshold:
                valid_objects.append(obj)
        
        # 조건을 만족하는 객체가 없으면 None 반환
        if not valid_objects:
            print(f"신뢰도 ≥ {confidence_threshold} 및 IoU < {iou_threshold} 조건을 만족하는 객체가 없습니다.")
            return None
        
        # 조건을 만족하는 객체 중 가장 높은 신뢰도를 가진 객체 선택
        best_object = max(valid_objects, key=lambda x: x['confidence'])
        print(f"선택된 정답: {best_object['class_name']} (신뢰도: {best_object['confidence']:.3f})")
        
        return best_object
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        두 바운딩 박스 간의 IoU(Intersection over Union) 계산
        
        Args:
            bbox1: 첫 번째 바운딩 박스 [x1, y1, x2, y2]
            bbox2: 두 번째 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            float: IoU 값 (0.0 ~ 1.0)
        """
        # 바운딩 박스 좌표 추출
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역 계산
        x1_intersect = max(x1_1, x1_2)
        y1_intersect = max(y1_1, y1_2)
        x2_intersect = min(x2_1, x2_2)
        y2_intersect = min(y2_1, y2_2)
        
        # 교집합이 없는 경우
        if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
            return 0.0
        
        # 교집합 면적
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
        
        # 각 바운딩 박스의 면적
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 합집합 면적
        union_area = area1 + area2 - intersection_area
        
        # IoU 계산
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def generate_quiz_options(self, correct_answer: Dict, detected_objects: List[Dict]) -> List[Dict]:
        """
        퀴즈 옵션 생성 (정답 1개 + 다른 객체 1개 + 랜덤 객체 2개)
        모든 옵션이 서로 다른 클래스가 되도록 보장
        
        Args:
            correct_answer: 정답 객체
            detected_objects: 검출된 객체 목록
            
        Returns:
            List[Dict]: 4개의 퀴즈 옵션 (한글 클래스명 사용, 모두 다른 클래스)
        """
        options = []
        used_classes = set()  # 이미 사용된 클래스 추적
        
        # 1. 정답 추가 (한글 클래스명 사용)
        correct_korean_name = CLASS_NAME_TO_KOREAN.get(correct_answer['class_name'], correct_answer['class_name'])
        options.append({
            'class_name': correct_korean_name,
            'is_correct': True,
            'confidence': correct_answer['confidence']
        })
        used_classes.add(correct_korean_name)
        
        # 2. 유사 객체 오답 우선 생성 (정답과 유사한 객체들)
        correct_korean_name = CLASS_NAME_TO_KOREAN.get(correct_answer['class_name'], correct_answer['class_name'])
        
        # 유사 객체 오답 조합에서 선택
        if correct_korean_name in SIMILAR_OBJECT_PAIRS:
            similar_wrong_answers = SIMILAR_OBJECT_PAIRS[correct_korean_name]
            # 유사 객체 중에서 아직 사용되지 않은 것들 선택
            available_similar = [obj for obj in similar_wrong_answers if obj not in used_classes]
            
            if available_similar:
                # 유사 객체를 우선적으로 오답으로 사용
                similar_wrong = random.choice(available_similar)
                options.append({
                    'class_name': similar_wrong,
                    'is_correct': False,
                    'confidence': 0.0
                })
                used_classes.add(similar_wrong)
            else:
                self._add_other_object_option(options, used_classes, detected_objects, correct_answer)
        else:
            self._add_other_object_option(options, used_classes, detected_objects, correct_answer)
        
        # 3. 랜덤 객체 2개 추가 (모든 옵션이 서로 다른 클래스가 되도록)
        remaining_count = 4 - len(options)
        
        for _ in range(remaining_count):
            # 아직 사용되지 않은 클래스들 중에서 선택
            available_classes = [c for c in VALID_CLASSES 
                               if CLASS_NAME_TO_KOREAN.get(c, c) not in used_classes]
            
            if available_classes:
                random_class = random.choice(available_classes)
                random_korean_name = CLASS_NAME_TO_KOREAN.get(random_class, random_class)
                options.append({
                    'class_name': random_korean_name,
                    'is_correct': False,
                    'confidence': 0.0
                })
                used_classes.add(random_korean_name)
            else:
                # 모든 클래스를 사용한 경우 (이론적으로는 발생하지 않아야 함)
                print("경고 : 모든 클래스를 사용했습니다. 중복을 허용합니다.")
                available_classes = [c for c in VALID_CLASSES if c != correct_answer['class_name']]
                random_class = random.choice(available_classes)
                random_korean_name = CLASS_NAME_TO_KOREAN.get(random_class, random_class)
                options.append({
                    'class_name': random_korean_name,
                    'is_correct': False,
                    'confidence': 0.0
                })
        
        # 옵션 순서 섞기
        random.shuffle(options)
        
        # 검증: 모든 옵션이 서로 다른 클래스인지 확인
        option_classes = [opt['class_name'] for opt in options]
        if len(option_classes) != len(set(option_classes)):
            print(f"경고 : 중복된 클래스가 발견되었습니다: {option_classes}")
        
        return options
    
    def _add_other_object_option(self, options: List[Dict], used_classes: set, 
                               detected_objects: List[Dict], correct_answer: Dict):
        """다른 검출된 객체에서 오답 옵션 추가"""
        other_objects = [obj for obj in detected_objects 
                        if obj['class_name'] != correct_answer['class_name']]
        
        if other_objects:
            other_obj = random.choice(other_objects)
            other_korean_name = CLASS_NAME_TO_KOREAN.get(other_obj['class_name'], other_obj['class_name'])
            
            if other_korean_name not in used_classes:
                options.append({
                    'class_name': other_korean_name,
                    'is_correct': False,
                    'confidence': other_obj['confidence']
                })
                used_classes.add(other_korean_name)
            else:
                # 이미 사용된 클래스라면 다른 객체 찾기
                available_other_objects = [obj for obj in other_objects 
                                         if CLASS_NAME_TO_KOREAN.get(obj['class_name'], obj['class_name']) not in used_classes]
                if available_other_objects:
                    other_obj = random.choice(available_other_objects)
                    other_korean_name = CLASS_NAME_TO_KOREAN.get(other_obj['class_name'], other_obj['class_name'])
                    options.append({
                        'class_name': other_korean_name,
                        'is_correct': False,
                        'confidence': other_obj['confidence']
                    })
                    used_classes.add(other_korean_name)
                else:
                    self._add_random_class_option(options, used_classes, correct_answer)
        else:
            self._add_random_class_option(options, used_classes, correct_answer)
    
    def _add_random_class_option(self, options: List[Dict], used_classes: set, correct_answer: Dict):
        """랜덤 클래스에서 오답 옵션 추가"""
        available_classes = [c for c in VALID_CLASSES 
                           if CLASS_NAME_TO_KOREAN.get(c, c) not in used_classes]
        if available_classes:
            random_class = random.choice(available_classes)
            random_korean_name = CLASS_NAME_TO_KOREAN.get(random_class, random_class)
            options.append({
                'class_name': random_korean_name,
                'is_correct': False,
                'confidence': 0.0
            })
            used_classes.add(random_korean_name)
