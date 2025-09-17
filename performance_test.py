#!/usr/bin/env python3
"""
퀴즈 생성 성능 테스트 (소요시간 및 시도 횟수 측정) - 병렬 처리 버전
"""
import sys
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import asyncio
import concurrent.futures
from typing import List, Dict, Any

# 한글 폰트 설정
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Windows에서 한글 폰트 사용 가능한 경우
try:
    # Windows 기본 한글 폰트 시도
    font_list = [
        'Malgun Gothic',      # 맑은 고딕
        'NanumGothic',        # 나눔고딕
        'NanumBarunGothic',   # 나눔바른고딕
        'AppleGothic',        # 맥용
        'DejaVu Sans'         # 기본 폰트
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in font_list:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            print(f"✓ 한글 폰트 설정: {font_name}")
            break
    else:
        print("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
        
except Exception as e:
    print(f"⚠️ 폰트 설정 중 오류: {e}")
    plt.rcParams['font.family'] = 'DejaVu Sans'

# quiz 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'quiz'))

# ==============================================
# 테스트 설정 상수
# ==============================================

# 퀴즈 난이도 설정
# - 'low': 낮은 노이즈 강도 (쉬운 퀴즈)
# - 'medium': 중간 노이즈 강도 (보통 퀴즈)  
# - 'high': 높은 노이즈 강도 (어려운 퀴즈)
QUIZ_DIFFICULTY = 'medium'

# 병렬 테스트 횟수
PARALLEL_TEST_COUNT = 10

# 최대 시도 횟수 (이 횟수만큼 시도해도 성공하지 못하면 포기)
MAX_ATTEMPTS = 100

# 객체 검출 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.6  # 60% 이상의 신뢰도만 인정
IOU_THRESHOLD = 0.5         # IoU 50% 이상만 인정

# 디노이징 방식 설정
# - 'hybrid': 하이브리드 디노이징 (고정 강도)
# - 'adaptive': 적응형 디노이징 (자동 강도 조절)
DENOISE_METHOD = 'adaptive'
DENOISE_STRENGTH = 'medium'  # 하이브리드 디노이징용 (적응형에서는 무시됨)

# 병렬 처리 설정
MAX_WORKERS = 4  # 동시 실행할 최대 워커 수

# ==============================================

class PerformanceTestGenerator:
    """성능 테스트용 퀴즈 생성기 (DB 저장 제외) - 병렬 처리 버전"""
    
    def __init__(self):
        """초기화"""
        from quiz.components import YOLODetector, ImageHandler
        from quiz.components.storage_manager import StorageManager
        from quiz.components.quiz_builder import QuizBuilder
        from quiz.components.model_manager import ModelManager
        
        print("=== 성능 테스트용 퀴즈 생성기 초기화 (병렬 처리) ===")
        
        # 1. 스토리지 관리자
        self.storage_manager = StorageManager()
        
        # 2. 모델 관리자
        model_manager = ModelManager()
        model_paths = model_manager.get_model_paths()
        
        # 3. YOLO 검출기
        self.yolo_detector = YOLODetector(model_paths['train_model'], model_paths['basic_model'])
        
        # 4. 이미지 핸들러
        self.image_handler = ImageHandler()
        
        # 5. 퀴즈 빌더
        self.quiz_builder = QuizBuilder()
        
        print("✓ 초기화 완료\n")
    
    def generate_single_quiz(self, test_id: int, difficulty: str = QUIZ_DIFFICULTY) -> Dict[str, Any]:
        """단일 퀴즈 생성 (병렬 처리용)"""
        
        print(f"[테스트 {test_id}] {difficulty.upper()} 난이도 퀴즈 생성 시작")
        
        start_time = time.time()
        attempt_count = 0
        successful_attempts = []
        failed_attempts = []
        
        while True:
            attempt_count += 1
            attempt_start = time.time()
            
            try:
                # 1. 원본 이미지 가져오기
                image_key, image_bytes = self.storage_manager.get_random_original_image("images")
                
                # 2. train_tf 모델로 검출
                detected_objects = self.yolo_detector.detect_objects(image_bytes)
                
                if not detected_objects:
                    attempt_time = time.time() - attempt_start
                    failed_attempts.append({
                        'attempt': attempt_count,
                        'reason': 'Original Detection Failed',
                        'reason_ko': '원본 이미지 검출 실패',
                        'time': attempt_time
                    })
                    continue
                
                # 3. 정답 선택
                correct_answer = self.quiz_builder.select_correct_answer(detected_objects, 
                                                                      confidence_threshold=CONFIDENCE_THRESHOLD, 
                                                                      iou_threshold=IOU_THRESHOLD)
                
                if correct_answer is None:
                    attempt_time = time.time() - attempt_start
                    failed_attempts.append({
                        'attempt': attempt_count,
                        'reason': 'No Suitable Answer',
                        'reason_ko': '적절한 정답 없음',
                        'time': attempt_time
                    })
                    continue
                
                # 4. 노이즈 추가
                intensity, alpha = self.image_handler.get_random_noise_params(difficulty)
                processed_image_array = self.image_handler.process_image_with_noise(image_bytes, 
                                                                                  intensity=intensity, 
                                                                                  alpha=alpha)
                
                # 5. 이미지를 바이트로 변환
                success, encoded_image = cv2.imencode('.webp', processed_image_array)
                if not success:
                    attempt_time = time.time() - attempt_start
                    failed_attempts.append({
                        'attempt': attempt_count,
                        'reason': 'Image Encoding Failed',
                        'reason_ko': '이미지 인코딩 실패',
                        'time': attempt_time
                    })
                    continue
                    
                processed_image_bytes = encoded_image.tobytes()
                
                # 6. 이중 검증
                validation_start = time.time()
                
                # 6-1. 기본 노이즈 이미지 검증
                basic_validation = self.yolo_detector.validate_with_basic_model(processed_image_bytes, correct_answer)
                
                if not basic_validation:
                    attempt_time = time.time() - attempt_start
                    failed_attempts.append({
                        'attempt': attempt_count,
                        'reason': 'Stage 1 Validation Failed',
                        'reason_ko': '1단계 검증 실패',
                        'time': attempt_time,
                        'validation_time': time.time() - validation_start
                    })
                    continue
                
                # 6-2. 적응형 디노이징 후 검증
                denoising_validation = self.yolo_detector.validate_with_adaptive_denoising(
                    processed_image_bytes, correct_answer
                )
                
                is_denoising_valid = denoising_validation.get('is_different_from_current', False)
                
                if not is_denoising_valid:
                    attempt_time = time.time() - attempt_start
                    failed_attempts.append({
                        'attempt': attempt_count,
                        'reason': 'Stage 2 Validation Failed',
                        'reason_ko': '2단계 검증 실패',
                        'time': attempt_time,
                        'validation_time': time.time() - validation_start
                    })
                    continue
                
                # 성공!
                total_time = time.time() - start_time
                attempt_time = time.time() - attempt_start
                validation_time = time.time() - validation_start
                
                successful_attempts.append({
                    'attempt': attempt_count,
                    'time': attempt_time,
                    'validation_time': validation_time
                })
                
                print(f"[테스트 {test_id}] ✅ 성공! (시도: {attempt_count}회, 시간: {total_time:.2f}초)")
                
                return {
                    'test_id': test_id,
                    'success': True,
                    'total_time': total_time,
                    'attempt_count': attempt_count,
                    'correct_answer': correct_answer['class_name'],
                    'intensity': intensity,
                    'alpha': alpha,
                    'successful_attempts': successful_attempts,
                    'failed_attempts': failed_attempts,
                    'image_key': image_key,
                    'original_image_bytes': image_bytes,  # 원본 이미지 바이트 추가
                    'detected_objects': detected_objects,
                    'processed_image_array': processed_image_array,
                    'basic_validation': basic_validation,
                    'denoising_validation': denoising_validation
                }
                
            except Exception as e:
                attempt_time = time.time() - attempt_start
                failed_attempts.append({
                    'attempt': attempt_count,
                    'reason': f'Error: {str(e)}',
                    'reason_ko': f'오류: {str(e)}',
                    'time': attempt_time
                })
                
                # 너무 많은 실패 시 중단
                if attempt_count >= MAX_ATTEMPTS:
                    print(f"[테스트 {test_id}] ❌ 최대 시도 횟수 초과 ({MAX_ATTEMPTS}회)")
                    return {
                        'test_id': test_id,
                        'success': False,
                        'total_time': time.time() - start_time,
                        'attempt_count': attempt_count,
                        'failed_attempts': failed_attempts
                    }
                continue
    
    def run_parallel_tests(self, difficulty: str = QUIZ_DIFFICULTY) -> List[Dict[str, Any]]:
        """병렬로 여러 테스트 실행"""
        
        print(f"=== {difficulty.upper()} 난이도 병렬 퀴즈 생성 테스트 ({PARALLEL_TEST_COUNT}회) ===")
        print(f"병렬 처리 워커 수: {MAX_WORKERS}")
        
        start_time = time.time()
        results = []
        
        # ThreadPoolExecutor를 사용한 병렬 처리
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 모든 테스트 작업 제출
            future_to_test_id = {
                executor.submit(self.generate_single_quiz, i, difficulty): i 
                for i in range(1, PARALLEL_TEST_COUNT + 1)
            }
            
            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_test_id):
                test_id = future_to_test_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[테스트 {test_id}] 완료 - {'성공' if result['success'] else '실패'}")
                except Exception as e:
                    print(f"[테스트 {test_id}] 예외 발생: {e}")
                    results.append({
                        'test_id': test_id,
                        'success': False,
                        'error': str(e)
                    })
        
        total_time = time.time() - start_time
        
        # 결과 분석
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        print("\n" + "=" * 80)
        print("📊 병렬 테스트 결과 요약")
        print("=" * 80)
        print(f"총 테스트 수: {PARALLEL_TEST_COUNT}회")
        print(f"성공: {len(successful_results)}회")
        print(f"실패: {len(failed_results)}회")
        print(f"성공률: {len(successful_results)/PARALLEL_TEST_COUNT*100:.1f}%")
        print(f"총 소요시간: {total_time:.2f}초")
        print(f"평균 테스트 시간: {total_time/PARALLEL_TEST_COUNT:.2f}초")
        
        if successful_results:
            avg_success_time = np.mean([r['total_time'] for r in successful_results])
            avg_attempts = np.mean([r['attempt_count'] for r in successful_results])
            print(f"성공한 테스트 평균 시간: {avg_success_time:.2f}초")
            print(f"성공한 테스트 평균 시도 횟수: {avg_attempts:.1f}회")
        
        print("=" * 80)
        
        # 각 성공한 테스트에 대해 개별 시각화 생성
        if successful_results:
            for i, success_result in enumerate(successful_results):
                print(f"시각화 생성 중... ({i+1}/{len(successful_results)})")
                self.create_parallel_performance_visualization(
                    success_result, results, total_time, test_index=i+1
                )
        
        return results
    
    def create_parallel_performance_visualization(self, sample_result: Dict[str, Any], 
                                                all_results: List[Dict[str, Any]], 
                                                total_time: float, test_index: int = 1):
        """병렬 테스트 결과 시각화 (3x2 그리드)"""
        
        # 샘플 결과에서 데이터 추출
        image_key = sample_result.get('image_key', '')
        processed_image_array = sample_result.get('processed_image_array', None)
        correct_answer = sample_result.get('correct_answer', '')
        detected_objects = sample_result.get('detected_objects', [])
        basic_validation = sample_result.get('basic_validation', False)
        denoising_validation = sample_result.get('denoising_validation', {})
        
        # 원본 이미지 처리 (이미 가지고 있는 image_bytes 사용)
        try:
            # 샘플 결과에서 원본 이미지 바이트 추출
            original_bytes = sample_result.get('original_image_bytes', None)
            if original_bytes is None:
                # 기본 이미지 생성
                original_image = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(original_image, "No Original Image", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                nparr = np.frombuffer(original_bytes, np.uint8)
                original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"⚠️ 원본 이미지 로딩 실패: {e}")
            # 기본 이미지 생성
            original_image = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.putText(original_image, "Image Load Failed", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 노이즈 이미지
        noisy_image = cv2.cvtColor(processed_image_array, cv2.COLOR_BGR2RGB)
        
        # 디노이징 이미지 생성 (적응형 디노이징 사용)
        from quiz.components.image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        denoised_image = preprocessor.adaptiveDenoising(processed_image_array)
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
        
        # 노이즈 이미지와 디노이징 이미지에서 yolo11x 모델로 검출
        # 노이즈 이미지 검출
        success, encoded_noisy = cv2.imencode('.webp', processed_image_array)
        noisy_image_bytes = encoded_noisy.tobytes()
        noisy_detected_objects = self.yolo_detector.detect_objects_with_basic_model(noisy_image_bytes)
        
        # 디노이징 이미지 검출
        success, encoded_denoised = cv2.imencode('.webp', denoised_image)
        denoised_image_bytes = encoded_denoised.tobytes()
        denoised_detected_objects = self.yolo_detector.detect_objects_with_basic_model(denoised_image_bytes)
        
        # bbox 그리기 함수
        def draw_bboxes(image, detected_objects, title_prefix=""):
            """이미지에 bbox 그리기"""
            result_image = image.copy()
            
            if detected_objects:
                for i, obj in enumerate(detected_objects[:3]):  # 최대 3개만 표시
                    x1, y1, x2, y2 = obj['bbox']
                    confidence = obj['confidence']
                    class_name = obj['class_name']
                    
                    # bbox 좌표가 이미지 범위를 벗어나지 않도록 클램핑
                    h, w = result_image.shape[:2]
                    x1 = max(0, min(int(x1), w-1))
                    y1 = max(0, min(int(y1), h-1))
                    x2 = max(0, min(int(x2), w-1))
                    y2 = max(0, min(int(y2), h-1))
                    
                    # bbox가 유효한지 확인 (크기가 너무 작으면 스킵)
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue
                    
                    # bbox 색상 (신뢰도에 따라)
                    color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.6 else (255, 0, 0)
                    
                    # bbox 그리기
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 라벨 텍스트
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    # 라벨 배경
                    cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # 라벨 텍스트
                    cv2.putText(result_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return result_image
        
        # 그래프 설정 (3x2 그리드)
        fig = plt.figure(figsize=(16, 12))
        
        # 상단: 이미지들 (bbox 포함)
        ax1 = plt.subplot(2, 3, 1)
        original_with_bbox = draw_bboxes(original_image, detected_objects or [])
        ax1.imshow(original_with_bbox)
        ax1.set_title(f'원본 이미지 (train_tf 모델)\n정답: {correct_answer}')
        ax1.axis('off')
        
        # 노이즈 이미지에서 검출된 객체 이름 추출
        noisy_detected_names = []
        if noisy_detected_objects:
            for obj in noisy_detected_objects[:3]:  # 최대 3개만
                noisy_detected_names.append(f"{obj['class_name']} ({obj['confidence']:.2f})")
        noisy_title = f'노이즈 이미지 (yolo11x 모델)\n노이즈: {sample_result.get("intensity", 0)*100:.0f}%\n검출: {", ".join(noisy_detected_names) if noisy_detected_names else "검출 실패"}'
        
        ax2 = plt.subplot(2, 3, 2)
        noisy_with_bbox = draw_bboxes(noisy_image, noisy_detected_objects or [])
        ax2.imshow(noisy_with_bbox)
        ax2.set_title(noisy_title)
        ax2.axis('off')
        
        # 디노이징 이미지에서 검출된 객체 이름 추출
        denoised_detected_names = []
        if denoised_detected_objects:
            for obj in denoised_detected_objects[:3]:  # 최대 3개만
                denoised_detected_names.append(f"{obj['class_name']} ({obj['confidence']:.2f})")
        denoised_title = f'적응형 디노이징 이미지 (yolo11x 모델)\n(자동 강도 조절)\n검출: {", ".join(denoised_detected_names) if denoised_detected_names else "검출 실패"}'
        
        ax3 = plt.subplot(2, 3, 3)
        denoised_with_bbox = draw_bboxes(denoised_image, denoised_detected_objects or [])
        ax3.imshow(denoised_with_bbox)
        ax3.set_title(denoised_title)
        ax3.axis('off')
        
        # 하단: 성능 차트들
        
        # 시도별 소요시간 (샘플 결과의 시도들)
        ax4 = plt.subplot(2, 3, 4)
        all_attempts = sample_result.get('failed_attempts', []) + sample_result.get('successful_attempts', [])
        all_attempts.sort(key=lambda x: x['attempt'])
        
        attempt_nums = [a['attempt'] for a in all_attempts]
        attempt_times = [a['time'] for a in all_attempts]
        colors = ['red' if a in sample_result.get('failed_attempts', []) else 'green' for a in all_attempts]
        
        ax4.bar(attempt_nums, attempt_times, color=colors, alpha=0.7)
        ax4.set_xlabel('시도 횟수')
        ax4.set_ylabel('소요시간 (초)')
        ax4.set_title('시도별 소요시간')
        ax4.grid(True, alpha=0.3)
        
        # 실패 원인별 분석
        ax5 = plt.subplot(2, 3, 5)
        failure_reasons = {}
        
        # 실패 원인별 카운트 계산
        for fail in sample_result.get('failed_attempts', []):
            reason = fail['reason']
            if reason not in failure_reasons:
                failure_reasons[reason] = 0
            failure_reasons[reason] += 1
        
        if failure_reasons:
            reasons = list(failure_reasons.keys())
            counts = list(failure_reasons.values())
            
            # 영어 레이블을 한글로 변환하는 딕셔너리
            reason_translation = {
                'Original Detection Failed': '원본 검출\n실패',
                'No Suitable Answer': '적절한 정답\n없음',
                'Image Encoding Failed': '이미지 인코딩\n실패', 
                'Stage 1 Validation Failed': '1단계 검증\n실패',
                'Stage 2 Validation Failed': '2단계 검증\n실패'
            }
            
            # 레이블을 한글로 변환
            korean_labels = []
            for reason in reasons:
                if reason.startswith('Error:'):
                    korean_labels.append('오류')
                else:
                    korean_labels.append(reason_translation.get(reason, reason))
            
            # 파이 차트 생성
            colors = plt.cm.Set3(range(len(reasons)))
            wedges, texts, autotexts = ax5.pie(counts, labels=korean_labels, autopct='%1.1f%%', 
                                               colors=colors, startangle=90)
            
            # 폰트 크기 조정
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            ax5.set_title('실패 원인별 분석', fontsize=10, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, '실패 없음', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12, fontweight='bold')
            ax5.set_title('실패 원인별 분석', fontsize=10, fontweight='bold')
        
        # 성능 요약
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        successful_count = len([r for r in all_results if r['success']])
        failed_count = len([r for r in all_results if not r['success']])
        success_rate = successful_count / len(all_results) * 100
        
        summary_text = f"""병렬 테스트 성능 요약

총 테스트 수: {len(all_results)}회
성공: {successful_count}회
실패: {failed_count}회
성공률: {success_rate:.1f}%

시간 정보:
• 총 소요시간: {total_time:.2f}초
• 평균 테스트 시간: {total_time/len(all_results):.2f}초

성공한 테스트:
• 평균 소요시간: {np.mean([r['total_time'] for r in all_results if r['success']]):.2f}초
• 평균 시도 횟수: {np.mean([r['attempt_count'] for r in all_results if r['success']]):.1f}회

퀴즈 상세 정보:
• 정답: {correct_answer}
• 노이즈강도: {sample_result.get("intensity", 0)*100:.0f}%
• 알파블랜드: {sample_result.get("alpha", 0)*100:.0f}%

모델 정보:
• train_tf: 16개 클래스 (파인튜닝)
• yolo11x: 80개 클래스 (COCO)"""
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top')
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"parallel_performance_test_result_{timestamp}_test{test_index:02d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 테스트 {test_index} 시각화 저장: {output_path}")

def main():
    """메인 함수"""
    
    # 가상환경 체크
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ 가상환경 활성화됨\n")
    else:
        print("❌ 가상환경이 활성화되지 않았습니다.")
        return
    
    generator = PerformanceTestGenerator()
    results = generator.run_parallel_tests(QUIZ_DIFFICULTY)
    
    successful_count = len([r for r in results if r['success']])
    print(f"\n🎉 병렬 테스트 완료!")
    print(f"성공: {successful_count}/{len(results)}회")

if __name__ == "__main__":
    main()
