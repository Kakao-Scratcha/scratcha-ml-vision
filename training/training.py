import torch
import gc
from ultralytics import YOLO

def optimize_memory():
    """GPU 메모리 최적화 함수"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def train_yolo11s_improved():
    print("학습 시작...")
    optimize_memory()

    # 모델 로딩
    model = YOLO('yolo11s.pt')

    train_args = {
        'data': './dataset.yaml',  # 데이터셋 설정 파일 경로
        'epochs': 150,  # 전체 학습 반복 횟수
        'imgsz': 640,  # 학습/검증 이미지 크기
        'batch': 32,  # 한 번에 처리할 이미지 개수
        'device': 0,  # GPU사용
        'workers': 4,  # 데이터 로딩을 위한 워커 프로세스 수
        'cache': True,  # 이미지를 메모리에 캐시하여 로딩 속도 향상
        'amp': True,  # 메모리 절약 및 속도 향상
        'patience': 30,  # Early stopping을 위한 성능 개선 대기 에포크 수
        'save': True,  # 모델 체크포인트 저장 여부
        'save_period': 5,  # 5 에포크마다 모델을 저장
        'project': 'runs/detect',  # 결과 저장 폴더명
        'name': 'test10',  # 실험 결과 폴더명
        'exist_ok': True,  # 기존 폴더가 있어도 덮어쓰기 허용
        'optimizer': 'AdamW',  # 최적화 알고리즘 (AdamW: 가중치 감쇠가 개선된 Adam)
        'lr0': 0.0005,  # 초기 학습률
        'weight_decay': 0.001,  # 가중치 감쇠 (과적합 방지)
        'cos_lr': True,  # Cosine Learning Rate 스케줄링 사용
        'warmup_epochs': 3,  # 학습률을 점진적으로 증가시키는 워밍업 에포크 수
        'mosaic': 0.8,  # Mosaic 증강 사용 확률 (4개 이미지를 합쳐서 학습)
        'mixup': 0.3,  # Mixup 증강 사용 확률 (두 이미지를 혼합)
        'copy_paste': 0.3,  # Copy-Paste 증강 사용 확률 (객체를 다른 이미지에 복사)
        'hsv_h': 0.04,  # HSV 색상 공간에서 Hue(색조) 변화 범위
        'hsv_s': 0.5,  # HSV 색상 공간에서 Saturation(채도) 변화 범위
        'hsv_v': 0.3,  # HSV 색상 공간에서 Value(명도) 변화 범위
        'fliplr': 0.6,  # 좌우 반전 증강 사용 확률
        'flipud': 0.3,  # 상하 반전 증강 사용 확률
        'single_cls': False,  # 단일 클래스 분류 모드 (False: 다중 클래스)
        'dropout': 0.05,  # Dropout 비율 (과적합 방지를 위한 뉴런 비활성화)
        'val': True,  # 검증 데이터로 모델 성능 평가 여부
        'plots': True,  # 학습 과정 그래프 및 결과 시각화 생성 여부
    }

    try:
        print(f"학습 시작 - 이미지 크기: {train_args['imgsz']}px")
        results = model.train(**train_args)
        print("학습 완료!")
        return results

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("메모리 부족! 더 작은 설정으로 재시도 중...")
            train_args.update({
                'imgsz': 512,
                'batch': 16,
                'workers': 2,
                'mosaic': 0.5,
                'mixup': 0.0,
                'copy_paste': 0.0,
            })
            optimize_memory()
            try:
                results = model.train(**train_args)
                print("재시도 학습 완료!")
                return results
            except Exception as e2:
                print(f"재시도 실패: {e2}")
                return None
        else:
            print(f"학습 중 오류: {e}")
            return None

def monitor_memory():
    """GPU 메모리 상태 모니터링"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU 총 메모리: {total:.1f} GB, 할당: {allocated:.1f} GB, 예약: {reserved:.1f} GB, 사용가능: {total-reserved:.1f} GB")

if __name__ == "__main__":
    monitor_memory()
    results = train_yolo11s_improved()

    if results:
        print("\n학습 결과 요약:")
        try:
            print(f"Box mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"Mask mAP50: {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
        except Exception as e:
            print(f"결과 출력 중 오류 발생: {e}")

        monitor_memory()
    else:
        print("학습 실패.")