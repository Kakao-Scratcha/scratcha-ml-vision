#!/usr/bin/env python3
"""
모델 정확도 테스트
- train_tf 모델로 origin/ 폴더의 이미지들을 검출
- 검출되지 않으면 no/ 폴더로 복사
- 검출되면 시각화하여 detect/ 폴더에 저장
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from pathlib import Path
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

# 한글 폰트 설정
import matplotlib.font_manager as fm

# Windows에서 한글 폰트 설정
try:
    font_list = [
        'Malgun Gothic',      # 맑은 고딕 (Windows 기본)
        'NanumGothic',        # 나눔고딕
        'NanumBarunGothic',   # 나눔바른고딕
        'AppleGothic',        # 맥용
        'Noto Sans CJK KR',   # 구글 폰트
        'DejaVu Sans'         # 기본 폰트
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    korean_font_found = False
    for font_name in font_list:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 한글 폰트 설정: {font_name}")
            korean_font_found = True
            break
    
    if not korean_font_found:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        print("⚠️ 시스템 폰트를 찾을 수 없어 Malgun Gothic을 강제 설정합니다.")
        
except Exception as e:
    print(f"⚠️ 폰트 설정 중 오류: {e}")
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

# quiz 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'quiz'))

# 한글 클래스명 매핑
KOREAN_CLASS_NAMES = {
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
    'toothbrush': '칫솔'
}

class ModelAccuracyTester:
    """모델 정확도 테스터 (비동기 처리)"""
    
    def __init__(self, max_workers=4):
        """초기화"""
        print("=== 모델 정확도 테스터 초기화 (비동기 처리) ===")
        
        # 경로 설정
        self.origin_dir = Path("imagetest/origin")
        self.detect_dir = Path("imagetest/detect")
        self.no_dir = Path("imagetest/no")
        
        # 폴더 생성
        self.detect_dir.mkdir(exist_ok=True)
        self.no_dir.mkdir(exist_ok=True)
        
        # 모델 로드
        from quiz.components.model_manager import ModelManager
        from quiz.components import YOLODetector
        
        model_manager = ModelManager()
        model_paths = model_manager.get_model_paths()
        
        self.yolo_detector = YOLODetector(model_paths['train_model'], model_paths['basic_model'])
        
        # 스레드 풀 설정
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        print("✓ 모델 로드 완료")
        print(f"✓ 원본 이미지 폴더: {self.origin_dir}")
        print(f"✓ 검출 결과 폴더: {self.detect_dir}")
        print(f"✓ 미검출 폴더: {self.no_dir}")
        print(f"✓ 최대 워커 수: {max_workers}\n")
    
    async def detect_objects_async(self, image_path):
        """비동기 이미지에서 객체 검출"""
        try:
            # 비동기 파일 읽기
            async with aiofiles.open(image_path, 'rb') as f:
                image_bytes = await f.read()
            
            # 스레드 풀에서 동기 검출 함수 실행
            loop = asyncio.get_event_loop()
            detected_objects = await loop.run_in_executor(
                self.executor, self._detect_objects_sync, image_bytes
            )
            
            return detected_objects
            
        except Exception as e:
            print(f"검출 실패 ({image_path}): {e}")
            return []
    
    def _detect_objects_sync(self, image_bytes):
        """동기 객체 검출 (스레드 풀에서 실행)"""
        try:
            # train_tf 모델로 검출
            detected_objects = self.yolo_detector.detect_objects(image_bytes)
            return detected_objects
        except Exception as e:
            print(f"동기 검출 실패: {e}")
            return []
    
    async def visualize_detection_async(self, image_path, detected_objects, output_path):
        """비동기 검출 결과 시각화"""
        try:
            # 스레드 풀에서 동기 시각화 함수 실행
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor, self._visualize_detection_sync, 
                image_path, detected_objects, output_path
            )
            return success
        except Exception as e:
            print(f"비동기 시각화 실패 ({image_path}): {e}")
            return False
    
    def _visualize_detection_sync(self, image_path, detected_objects, output_path):
        """동기 검출 결과 시각화 (스레드 풀에서 실행)"""
        try:
            # 이미지 읽기
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 검출 결과 그리기
            result_image = image_rgb.copy()
            
            for obj in detected_objects:
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
                
                # bbox 색상 (신뢰도에 따라)
                if confidence > 0.8:
                    color = (0, 255, 0)  # 초록색
                elif confidence > 0.6:
                    color = (255, 255, 0)  # 노란색
                else:
                    color = (255, 0, 0)  # 빨간색
                
                # bbox 그리기
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                
                # 라벨 텍스트
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 라벨 배경
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # 라벨 텍스트
                cv2.putText(result_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 결과 저장
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), result_image_bgr)
            
            return True
            
        except Exception as e:
            print(f"동기 시각화 실패 ({image_path}): {e}")
            return False
    
    async def process_single_image(self, image_path, index, total):
        """단일 이미지 비동기 처리"""
        print(f"[{index}/{total}] 처리 중: {image_path.name}")
        
        try:
            # 객체 검출
            detected_objects = await self.detect_objects_async(image_path)
            
            if detected_objects:
                # 검출된 경우 - 시각화하여 detect 폴더에 저장
                output_path = self.detect_dir / image_path.name
                success = await self.visualize_detection_async(image_path, detected_objects, output_path)
                
                if success:
                    print(f"✓ 검출됨 ({len(detected_objects)}개) → {output_path}")
                    return {'status': 'detected', 'count': len(detected_objects)}
                else:
                    print(f"❌ 시각화 실패 → {self.no_dir / image_path.name}")
                    # 파일 이동 (동기)
                    shutil.move(str(image_path), str(self.no_dir / image_path.name))
                    return {'status': 'visualization_failed'}
            else:
                # 검출되지 않은 경우 - no 폴더로 이동
                shutil.move(str(image_path), str(self.no_dir / image_path.name))
                print(f"✗ 검출 안됨 → {self.no_dir / image_path.name}")
                return {'status': 'not_detected'}
                
        except Exception as e:
            print(f"❌ 처리 실패 ({image_path.name}): {e}")
            # 오류 발생 시 no 폴더로 이동
            try:
                shutil.move(str(image_path), str(self.no_dir / image_path.name))
            except:
                pass
            return {'status': 'error', 'error': str(e)}
    
    async def run_accuracy_test_async(self, max_images=None, batch_size=10):
        """비동기 정확도 테스트 실행"""
        print("=== 모델 정확도 테스트 시작 (비동기 처리) ===")
        
        # 원본 이미지 목록 가져오기
        image_files = list(self.origin_dir.glob("*.JPEG"))
        if not image_files:
            print("❌ 원본 이미지를 찾을 수 없습니다.")
            return
        
        if max_images:
            image_files = image_files[:max_images]
        
        total_images = len(image_files)
        print(f"총 {total_images}개 이미지 처리 예정 (배치 크기: {batch_size})")
        
        # 통계 변수
        detected_count = 0
        not_detected_count = 0
        visualization_failed_count = 0
        error_count = 0
        
        # 배치별로 처리
        for i in range(0, total_images, batch_size):
            batch = image_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_images + batch_size - 1) // batch_size
            
            print(f"\n--- 배치 {batch_num}/{total_batches} 처리 중 ({len(batch)}개 이미지) ---")
            
            # 배치 내에서 비동기 처리
            tasks = []
            for j, image_path in enumerate(batch):
                task = self.process_single_image(image_path, i + j + 1, total_images)
                tasks.append(task)
            
            # 배치 완료 대기
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 집계
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                elif result['status'] == 'detected':
                    detected_count += 1
                elif result['status'] == 'not_detected':
                    not_detected_count += 1
                elif result['status'] == 'visualization_failed':
                    visualization_failed_count += 1
                elif result['status'] == 'error':
                    error_count += 1
            
            print(f"배치 {batch_num} 완료 - 검출: {detected_count}, 미검출: {not_detected_count}, 오류: {error_count}")
        
        # 결과 요약
        print(f"\n=== 테스트 완료 ===")
        print(f"총 이미지: {total_images}개")
        print(f"검출됨: {detected_count}개 ({detected_count/total_images*100:.1f}%)")
        print(f"미검출: {not_detected_count}개 ({not_detected_count/total_images*100:.1f}%)")
        print(f"시각화 실패: {visualization_failed_count}개")
        print(f"처리 오류: {error_count}개")
        print(f"검출 결과: {self.detect_dir}")
        print(f"미검출 결과: {self.no_dir}")
        
        return {
            'total': total_images,
            'detected': detected_count,
            'not_detected': not_detected_count,
            'visualization_failed': visualization_failed_count,
            'error': error_count,
            'detection_rate': detected_count/total_images*100
        }
    
    def run_accuracy_test(self, max_images=None, batch_size=10):
        """정확도 테스트 실행 (동기 래퍼)"""
        return asyncio.run(self.run_accuracy_test_async(max_images, batch_size))

    def __del__(self):
        """소멸자 - 스레드 풀 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

def main():
    """메인 함수"""
    print("모델 정확도 테스트 시작 (비동기 처리)")
    
    # 워커 수 설정 (CPU 코어 수에 따라 조정)
    import multiprocessing
    max_workers = min(4, multiprocessing.cpu_count())
    
    tester = ModelAccuracyTester(max_workers=max_workers)
    
    try:
        # 테스트 실행 (전체 이미지, 배치 크기 20)
        results = tester.run_accuracy_test(batch_size=20)
        
        print(f"\n🎉 테스트 완료!")
        print(f"검출률: {results['detection_rate']:.1f}%")
        print(f"처리된 이미지: {results['total']}개")
        print(f"검출 성공: {results['detected']}개")
        print(f"미검출: {results['not_detected']}개")
        if results.get('visualization_failed', 0) > 0:
            print(f"시각화 실패: {results['visualization_failed']}개")
        if results.get('error', 0) > 0:
            print(f"처리 오류: {results['error']}개")
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        # 스레드 풀 정리
        if hasattr(tester, 'executor'):
            tester.executor.shutdown(wait=True)

if __name__ == "__main__":
    main()
