#!/usr/bin/env python3
"""
CAPTCHA 퀴즈 생성기
"""

import asyncio
import uuid
import cv2
from datetime import datetime
from typing import Dict, List

# 모듈화된 컴포넌트 import
from components import DatabaseManager, YOLODetector, ImageHandler, QuizBuilder
from config.settings import (
    MODEL_PATH, 
    BASIC_MODEL_PATH, 
    IMAGE_FOLDER, 
    DIFFICULTY_CONFIGS
)


class ObjectDetectionQuizGenerator:
    """
    각 컴포넌트를 조합하여 퀴즈를 생성합니다.
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        초기화 - 각 컴포넌트 인스턴스 생성
        
        Args:
            model_path: YOLO 모델 경로
        """
        print("CAPTCHA 퀴즈 생성기 초기화 중...")
        
        # 컴포넌트 초기화
        self.db_manager = DatabaseManager()
        self.yolo_detector = YOLODetector(model_path, BASIC_MODEL_PATH)
        self.image_handler = ImageHandler()
        self.quiz_builder = QuizBuilder()
        
        print("모든 컴포넌트 초기화 완료!")
    
    def generate_quiz_with_difficulty(self, difficulty: str, image_folder: str = IMAGE_FOLDER) -> Dict:
        """
        특정 난이도로 퀴즈 생성
        
        Args:
            difficulty: 난이도 ('high', 'middle', 'low')
            image_folder: 이미지를 가져올 폴더 경로
            
        Returns:
            Dict: 생성된 퀴즈 데이터
        """
        if difficulty not in DIFFICULTY_CONFIGS:
            raise ValueError(f"지원하지 않는 난이도: {difficulty}. 지원 난이도: {list(DIFFICULTY_CONFIGS.keys())}")
        
        try:
            # 1. 오브젝트 스토리지에서 랜덤 이미지 가져오기
            image_key, image_bytes = self.image_handler.get_random_image_from_storage(image_folder)
            
            # 2. YOLO 객체 검출
            detected_objects = self.yolo_detector.detect_objects(image_bytes)
            
            if not detected_objects:
                return self.generate_quiz_with_difficulty(difficulty, image_folder)
            
            # 3. 정답 선택 (조건: confidence ≥ 0.6, IoU < 0.5)
            correct_answer = self.quiz_builder.select_correct_answer(detected_objects, 
                                                                  confidence_threshold=0.6, 
                                                                  iou_threshold=0.5)
            
            if correct_answer is None:
                return self.generate_quiz_with_difficulty(difficulty, image_folder)
            
            # 4. 퀴즈 옵션 생성
            options = self.quiz_builder.generate_quiz_options(correct_answer, detected_objects)
            
            # 5. 이미지 노이즈 처리 (난이도별 랜덤 설정 적용)
            quiz_id = str(uuid.uuid4())
            
            # 난이도별 랜덤 노이즈 파라미터 생성
            intensity, alpha = self.image_handler.get_random_noise_params(difficulty)
            processed_image_array = self.image_handler.process_image_with_noise(image_bytes, 
                                                                              intensity=intensity, 
                                                                              alpha=alpha)
            
            # 6. 기본 모델로 검증 (노이즈 처리된 이미지 배열을 바이트로 변환)
            success, encoded_image = cv2.imencode('.jpg', processed_image_array)
            if success:
                processed_image_bytes = encoded_image.tobytes()
            else:
                raise ValueError("이미지 인코딩에 실패했습니다.")
            
            is_valid = self.yolo_detector.validate_with_basic_model(processed_image_bytes, correct_answer)
            
            if not is_valid:
                print("검증 실패: 기본 모델과 결과가 같습니다. 다른 이미지로 재시도합니다.")
                return self.generate_quiz_with_difficulty(difficulty, image_folder)
            
            print("검증 성공: 기본 모델과 결과가 다릅니다.")
            
            # 기본 모델 인식률 요약 정보 추가
            basic_detected_objects = self.yolo_detector.detect_objects_with_basic_model(processed_image_bytes)
            if basic_detected_objects:
                basic_best = max(basic_detected_objects, key=lambda x: x['confidence'])
                print(f" 노이즈 효과 요약 [{difficulty.upper()}]:")
                print(f" - 현재 모델 신뢰도: {correct_answer['confidence']*100:.1f}%")
                print(f" - 기본 모델 신뢰도: {basic_best['confidence']*100:.1f}%")
                print(f" - 신뢰도 감소율: {(correct_answer['confidence'] - basic_best['confidence'])/correct_answer['confidence']*100:.1f}%")
                print(f" - 노이즈 설정: 강도 {intensity*100:.0f}%, 알파 {alpha*100:.0f}%")
            
            # 7. 오브젝트 스토리지에 저장 (난이도별 폴더)
            storage_key = self.image_handler.save_image_array_to_object_storage(processed_image_array, quiz_id, difficulty=difficulty)
            
            # 난이도별 프롬프트 구성 - 정확한 값 사용
            prompt_text = f"스크래치 후 정답을 선택하세요. 노이즈 {intensity*100:.0f}% 알파블랜드 {alpha*100:.0f}%"
            
            # 8. 퀴즈 데이터 구성
            quiz_data = {
                'quiz_id': quiz_id,
                'image_url': storage_key,
                'correct_answer': correct_answer['class_name'],
                'options': options,
                'detected_objects': detected_objects,
                'original_image_path': image_key,
                'prompt': prompt_text,
                'difficulty': difficulty,
                'noise_intensity_pct': f"{intensity*100:.0f}%",
                'alpha_pct': f"{alpha*100:.0f}%"
            }
            
            # 9. MySQL 데이터베이스에 저장
            self.db_manager.save_quiz_to_database(quiz_data)
            
            print(f"퀴즈 생성 완료! [난이도: {difficulty.upper()}]")
            return quiz_data
            
        except Exception as e:
            print(f"퀴즈 생성 중 오류 발생: {e}")
            raise
    
    async def generate_quizzes_by_difficulty_async(self, image_folder: str = IMAGE_FOLDER, 
                                                 max_concurrent: int = 3) -> Dict[str, List[Dict]]:
        """
        3가지 난이도별로 정해진 수량만큼 퀴즈를 비동기 병렬로 생성
        
        Args:
            image_folder: 이미지를 가져올 폴더 경로
            max_concurrent: 최대 동시 실행 개수
            
        Returns:
            Dict[str, List[Dict]]: 난이도별 퀴즈 리스트
        """
        all_quizzes = {}
        total_count = 0
        
        print(f"\n난이도별 퀴즈 비동기 병렬 생성 시작...")
        
        # 각 난이도별로 동시 실행
        difficulty_tasks = []
        for difficulty, cfg in DIFFICULTY_CONFIGS.items():
            count = cfg['count']
            print(f"  - {difficulty.upper()}: {count}개 생성 예정")
            task = self._generate_difficulty_quizzes_async(difficulty, count, image_folder, max_concurrent)
            difficulty_tasks.append((difficulty, task))
        
        # 모든 난이도 동시 실행
        results = await asyncio.gather(*[task for _, task in difficulty_tasks], return_exceptions=True)
        
        # 결과 정리
        for i, (difficulty, _) in enumerate(difficulty_tasks):
            if isinstance(results[i], Exception):
                print(f"경고: {difficulty.upper()} 난이도 생성 실패: {results[i]}")
                all_quizzes[difficulty] = []
            else:
                all_quizzes[difficulty] = results[i]
                total_count += len(results[i])
                expected = DIFFICULTY_CONFIGS[difficulty]['count']
                print(f"{difficulty.upper()} 난이도 완료: {len(results[i])}/{expected}개")
        
        print(f"\n전체 퀴즈 생성 완료!")
        print(f"총 생성된 퀴즈: {total_count}개")
        for difficulty, quizzes in all_quizzes.items():
            expected = DIFFICULTY_CONFIGS[difficulty]['count']
            print(f"  - {difficulty.upper()}: {len(quizzes)}/{expected}개")
        
        return all_quizzes
    
    async def _generate_difficulty_quizzes_async(self, difficulty: str, count: int, 
                                               image_folder: str, max_concurrent: int = 3) -> List[Dict]:
        """
        특정 난이도의 퀴즈들을 비동기 병렬로 생성
        
        Args:
            difficulty: 난이도
            count: 생성할 개수
            image_folder: 이미지 폴더
            max_concurrent: 최대 동시 실행 개수
            
        Returns:
            List[Dict]: 생성된 퀴즈 리스트
        """
        print(f"\n{difficulty.upper()} 난이도 퀴즈 {count}개 비동기 생성 시작...")
        
        # 동기 함수를 비동기로 실행
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.generate_quiz_with_difficulty, difficulty, image_folder)
            for _ in range(count)
        ]
        
        # 모든 작업 실행 및 결과 수집
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 결과만 필터링
        successful_quizzes = []
        for i, result in enumerate(results):
            if result is not None and not isinstance(result, Exception):
                successful_quizzes.append(result)
                print(f"  {difficulty.upper()} {len(successful_quizzes)}/{count} 완료")
            else:
                print(f"경고: {difficulty.upper()} {i+1}번째 실패: {result}")
        
        return successful_quizzes


async def main():
    """메인 함수"""
    # 퀴즈 생성기 초기화
    generator = ObjectDetectionQuizGenerator()
    
    try:
        print(f"\n 난이도별 퀴즈 생성 시작...")
        print(f"   - HIGH: {DIFFICULTY_CONFIGS['high']['count']}개 (노이즈 {DIFFICULTY_CONFIGS['high']['intensity_pct']}%, 알파 {DIFFICULTY_CONFIGS['high']['alpha_pct']}%)")
        print(f"   - MIDDLE: {DIFFICULTY_CONFIGS['middle']['count']}개 (노이즈 {DIFFICULTY_CONFIGS['middle']['intensity_pct']}%, 알파 {DIFFICULTY_CONFIGS['middle']['alpha_pct']}%)")
        print(f"   - LOW: {DIFFICULTY_CONFIGS['low']['count']}개 (노이즈 {DIFFICULTY_CONFIGS['low']['intensity_pct']}%, 알파 {DIFFICULTY_CONFIGS['low']['alpha_pct']}%)")
        
        all_quizzes = await generator.generate_quizzes_by_difficulty_async("images/")
        
        # 생성된 퀴즈 요약 정보 출력
        if all_quizzes:
            print(f"\n생성된 퀴즈 클래스별 분석:")
            all_class_counts = {}
            difficulty_class_counts = {}
            
            for difficulty, quiz_list in all_quizzes.items():
                difficulty_class_counts[difficulty] = {}
                for quiz in quiz_list:
                    class_name = quiz['correct_answer']
                    all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1
                    difficulty_class_counts[difficulty][class_name] = difficulty_class_counts[difficulty].get(class_name, 0) + 1
            
            print(f"\n전체 클래스별 분포:")
            for class_name, count in sorted(all_class_counts.items()):
                print(f"  - {class_name}: {count}개")
            
            print(f"\n난이도별 클래스 분포:")
            for difficulty, class_counts in difficulty_class_counts.items():
                print(f"  {difficulty.upper()}:")
                for class_name, count in sorted(class_counts.items()):
                    print(f"    - {class_name}: {count}개")
        
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")


def run_async():
    """비동기 실행 래퍼 함수"""
    asyncio.run(main())


if __name__ == "__main__":
    print("비동기 병렬 모드로 실행합니다.")
    run_async()
