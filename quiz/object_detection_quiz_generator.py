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
from components.storage_manager import StorageManager
from components.model_manager import ModelManager
from config.settings import (
    ORIGINAL_IMAGE_FOLDER, 
    QUIZ_IMAGE_FOLDER, 
    DIFFICULTY_CONFIGS
)


class ObjectDetectionQuizGenerator:
    """
    각 컴포넌트를 조합하여 퀴즈를 생성합니다.
    """
    
    def __init__(self):
        """
        초기화 - 각 컴포넌트 인스턴스 생성
        """
        print("\n=== CAPTCHA 퀴즈 생성기 초기화 시작 ===")
        
        # 1단계: 스토리지 관리자 초기화
        print("1단계: StorageManager 초기화 중...")
        self.storage_manager = StorageManager()
        print("✓ StorageManager 초기화 완료")
        
        # 2단계: 모델 관리자 초기화
        print("\n2단계: ModelManager 초기화 중...")
        self.model_manager = ModelManager()
        print("✓ ModelManager 초기화 완료")
        
        # 3단계: 모델 준비 및 다운로드
        print("\n3단계: ML 모델 준비 중...")
        if not self.model_manager.ensure_models_available():
            raise RuntimeError("ML 모델을 준비할 수 없습니다.")
        print("✓ ML 모델 준비 완료")
        
        # 4단계: 데이터베이스 관리자 초기화
        print("\n4단계: DatabaseManager 초기화 중...")
        self.db_manager = DatabaseManager()
        print("✓ DatabaseManager 초기화 완료")
        
        # 5단계: YOLO 검출기 초기화
        print("\n5단계: YOLO 검출기 초기화 중...")
        model_paths = self.model_manager.get_model_paths()
        print(f"  - 훈련된 모델 경로: {model_paths['train_model']}")
        print(f"  - 기본 모델 경로: {model_paths['basic_model']}")
        self.yolo_detector = YOLODetector(model_paths['train_model'], model_paths['basic_model'])
        print("✓ YOLO 검출기 초기화 완료")
        
        # 6단계: 이미지 핸들러 초기화
        print("\n6단계: ImageHandler 초기화 중...")
        self.image_handler = ImageHandler()
        print("✓ ImageHandler 초기화 완료")
        
        # 7단계: 퀴즈 빌더 초기화
        print("\n7단계: QuizBuilder 초기화 중...")
        self.quiz_builder = QuizBuilder()
        print("✓ QuizBuilder 초기화 완료")
        
        print("\n=== CAPTCHA 퀴즈 생성기 초기화 완료 ===")
        print("🎉 모든 컴포넌트가 성공적으로 초기화되었습니다!")
    
    def generate_quiz_with_difficulty(self, difficulty: str, image_folder: str = ORIGINAL_IMAGE_FOLDER) -> Dict:
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
            # 1. 새 버킷에서 랜덤 원본 이미지 가져오기
            image_key, image_bytes = self.storage_manager.get_random_original_image(image_folder)
            
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
            
            # 6. 이중 검증: 기본 노이즈 이미지 + 디노이징된 이미지 모두 체크
            success, encoded_image = cv2.imencode('.webp', processed_image_array)
            if success:
                processed_image_bytes = encoded_image.tobytes()
            else:
                raise ValueError("이미지 인코딩에 실패했습니다.")
            
            print("\n=== 이중 검증 시작 ===")
            print(f"train_tf 모델 정답: {correct_answer['class_name']} (신뢰도: {correct_answer['confidence']:.3f})")
            
            # 6-1. 기본 노이즈 이미지로 검증
            print("\n1단계: 기본 노이즈 이미지 검증")
            basic_validation = self.yolo_detector.validate_with_basic_model(processed_image_bytes, correct_answer)
            
            # 기본 노이즈 이미지 검출 결과 상세 로그
            basic_detected_objects = self.yolo_detector.detect_objects_with_basic_model(processed_image_bytes)
            if basic_detected_objects:
                basic_best = max(basic_detected_objects, key=lambda x: x['confidence'])
                print(f"노이즈 이미지 결과: {basic_best['class_name']} (신뢰도: {basic_best['confidence']:.3f})")
                print(f"   → train_tf와 비교: {'다름 ✓' if basic_validation else '같음 ❌'}")
            else:
                print("노이즈 이미지 결과: 검출 실패")
                print("   → train_tf와 비교: 다름 ✓ (검출 실패)")
            
            if not basic_validation:
                print("❌ 1단계 실패: 기본 노이즈 이미지에서 기본 모델과 결과가 같습니다. 다른 이미지로 재시도합니다.")
                return self.generate_quiz_with_difficulty(difficulty, image_folder)
            
            print("✅ 1단계 통과: 기본 노이즈 이미지에서 다른 결과")
            
            # 6-2. 하이브리드 디노이징 후 검증
            print("\n2단계: 하이브리드 디노이징 후 검증")
            denoising_validation = self.yolo_detector.validate_with_hybrid_denoising(
                processed_image_bytes, correct_answer, denoise_strength='medium'
            )
            
            # 디노이징 이미지 검출 결과 상세 로그
            denoised_detection = denoising_validation.get('denoised_detection')
            if denoised_detection:
                print(f"🔧 디노이징 이미지 결과: {denoised_detection['class_name']} (신뢰도: {denoised_detection['confidence']:.3f})")
            else:
                print("🔧 디노이징 이미지 결과: 검출 실패")
            
            is_denoising_valid = denoising_validation.get('is_different_from_current', False)
            print(f"     train_tf와 비교: {'다름' if is_denoising_valid else '같음'}")
            
            if not is_denoising_valid:
                print(" 2단계 실패: 하이브리드 디노이징 후에도 기본 모델과 결과가 같습니다. 다른 이미지로 재시도합니다.")
                return self.generate_quiz_with_difficulty(difficulty, image_folder)
            
            print(" 2단계 통과: 하이브리드 디노이징 후에도 다른 결과")
            
            # 종합 결과 로그
            print("\n 이중 검증 성공: 두 조건 모두 만족!")
            print("=" * 50)
            print(f" 검증 결과 요약:")
            print(f"    train_tf 정답:     {correct_answer['class_name']} (신뢰도: {correct_answer['confidence']:.3f})")
            if basic_detected_objects:
                basic_best = max(basic_detected_objects, key=lambda x: x['confidence'])
                print(f"    노이즈 이미지:      {basic_best['class_name']} (신뢰도: {basic_best['confidence']:.3f})")
            else:
                print(f"    노이즈 이미지:      검출 실패")
            if denoised_detection:
                print(f"    디노이징 이미지:    {denoised_detection['class_name']} (신뢰도: {denoised_detection['confidence']:.3f})")
            else:
                print(f"    디노이징 이미지:    검출 실패")
            print(f"    디노이징 개선 효과: {denoising_validation.get('denoising_improved', False)}")
            print(f"    신뢰도 개선:        {denoising_validation.get('confidence_improvement', 0.0):.3f}")
            print("=" * 50)
            
            # 7. 노이즈 처리된 이미지를 기존 버킷에 저장 (난이도별 폴더)
            # 이미지 배열을 바이트로 변환
            success, encoded_image = cv2.imencode('.jpg', processed_image_array)
            if success:
                processed_image_bytes = encoded_image.tobytes()
                storage_key = self.storage_manager.save_quiz_image(processed_image_bytes, quiz_id, difficulty)
            else:
                raise ValueError("노이즈 처리된 이미지 인코딩에 실패했습니다.")
            
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
    
    async def generate_quizzes_by_difficulty_async(self, image_folder: str = ORIGINAL_IMAGE_FOLDER, 
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
    
    async def generate_scheduled_quizzes(self, target_counts=None):
        """        
        Args:
            target_counts: 난이도별 생성할 개수 (None이면 설정 파일 사용)
        """
        if target_counts is None:
            # 기본 설정값 사용 (난이도별 동일한 수량)
            from config.settings import SCHEDULED_QUIZ_COUNT
            target_counts = {
                'high': SCHEDULED_QUIZ_COUNT,
                'middle': SCHEDULED_QUIZ_COUNT,
                'low': SCHEDULED_QUIZ_COUNT
            }
        
        print(f"\n=== 스케줄된 퀴즈 생성 시작 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
        print(f"목표 생성 수량:")
        for difficulty, count in target_counts.items():
            print(f"  - {difficulty.upper()}: {count}개")
        
        total_generated = 0
        
        try:
            # 각 난이도별로 순차 생성
            for difficulty, target_count in target_counts.items():
                print(f"\n{difficulty.upper()} 난이도 퀴즈 {target_count}개 생성 시작...")
                
                generated_count = 0
                for i in range(target_count):
                    try:
                        print(f"  {difficulty.upper()} {i+1}/{target_count} 생성 중...")
                        quiz = self.generate_quiz_with_difficulty(difficulty)
                        
                        if quiz:
                            generated_count += 1
                            total_generated += 1
                            print(f"  ✓ {difficulty.upper()} {generated_count}/{target_count} 완료 - 정답: {quiz['correct_answer']}")
                        else:
                            print(f"  ✗ {difficulty.upper()} {i+1}번째 생성 실패")
                            
                    except Exception as e:
                        print(f"  ✗ {difficulty.upper()} {i+1}번째 생성 중 오류: {e}")
                        continue
                
                print(f"{difficulty.upper()} 난이도 완료: {generated_count}/{target_count}개 생성")
            
            print(f"\n=== 스케줄된 퀴즈 생성 완료 ===")
            print(f"총 생성된 퀴즈: {total_generated}개")
            print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return total_generated
            
        except Exception as e:
            print(f"✗ 퀴즈 생성 중 치명적 오류: {e}")
            raise


async def main():
    """메인 함수"""
    # 퀴즈 생성기 초기화
    generator = ObjectDetectionQuizGenerator()
    
    try:
        print(f"\n 난이도별 퀴즈 생성 시작...")
        print(f"   - HIGH: {DIFFICULTY_CONFIGS['high']['count']}개 (노이즈 {DIFFICULTY_CONFIGS['high']['intensity_pct']}%, 알파 {DIFFICULTY_CONFIGS['high']['alpha_pct']}%)")
        print(f"   - MIDDLE: {DIFFICULTY_CONFIGS['middle']['count']}개 (노이즈 {DIFFICULTY_CONFIGS['middle']['intensity_pct']}%, 알파 {DIFFICULTY_CONFIGS['middle']['alpha_pct']}%)")
        print(f"   - LOW: {DIFFICULTY_CONFIGS['low']['count']}개 (노이즈 {DIFFICULTY_CONFIGS['low']['intensity_pct']}%, 알파 {DIFFICULTY_CONFIGS['low']['alpha_pct']}%)")
        
        all_quizzes = await generator.generate_quizzes_by_difficulty_async(ORIGINAL_IMAGE_FOLDER)
        
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
