#!/usr/bin/env python3
"""
크론잡용 CAPTCHA 퀴즈 생성기
일정 시간마다 실행되어 정해진 수량의 퀴즈를 생성하고 종료
"""

import asyncio
import sys
import os
from datetime import datetime
from object_detection_quiz_generator import ObjectDetectionQuizGenerator
from config.settings import DIFFICULTY_CONFIGS

class CronJobQuizGenerator:
    """크론잡용 퀴즈 생성기"""
    
    def __init__(self):
        """초기화"""
        self.generator = ObjectDetectionQuizGenerator()
        
    async def generate_scheduled_quizzes(self, target_counts=None):
        """
        스케줄된 퀴즈 생성
        
        Args:
            target_counts: 난이도별 생성할 개수 (None이면 설정 파일 사용)
        """
        if target_counts is None:
            # 기본 설정값의 1/3로 설정 (하루 3번 실행 가정)
            target_counts = {
                'high': max(1, DIFFICULTY_CONFIGS['high']['count'] // 3),
                'middle': max(1, DIFFICULTY_CONFIGS['middle']['count'] // 3),
                'low': max(1, DIFFICULTY_CONFIGS['low']['count'] // 3)
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
                        quiz = self.generator.generate_quiz_with_difficulty(difficulty)
                        
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
    print("크론잡 퀴즈 생성기 시작...")
    
    # 환경 변수에서 크론잡 모드 확인
    cronjob_mode = os.getenv('CRONJOB_MODE', 'false').lower() == 'true'
    
    if cronjob_mode:
        print("크론잡 모드로 실행됩니다.")
        
        # 환경 변수에서 생성 수량 설정 (선택사항)
        target_counts = {}
        try:
            target_counts['high'] = int(os.getenv('HIGH_COUNT', 3))
            target_counts['middle'] = int(os.getenv('MIDDLE_COUNT', 3)) 
            target_counts['low'] = int(os.getenv('LOW_COUNT', 3))
        except ValueError:
            target_counts = None  # 기본값 사용
    else:
        target_counts = None
    
    try:
        generator = CronJobQuizGenerator()
        total_generated = await generator.generate_scheduled_quizzes(target_counts)
        
        # 성공 시 exit code 0
        print(f"✓ 크론잡 완료: {total_generated}개 퀴즈 생성")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n크론잡이 중단되었습니다.")
        sys.exit(1)
        
    except Exception as e:
        print(f"✗ 크론잡 실행 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    print("크론잡 모드로 비동기 실행합니다.")
    asyncio.run(main())

