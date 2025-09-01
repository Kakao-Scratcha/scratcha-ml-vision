#!/usr/bin/env python3
"""
스케줄러 서비스
APScheduler를 사용하여 정기적으로 퀴즈를 생성합니다.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from object_detection_quiz_generator import ObjectDetectionQuizGenerator
from config.settings import (
    SCHEDULE_ENABLED, 
    SCHEDULE_INTERVAL_HOURS, 
    SCHEDULE_TIMEZONE,
    SCHEDULED_QUIZ_COUNTS
)

logger = logging.getLogger(__name__)

class SchedulerService:
    """스케줄러 서비스 클래스"""
    
    def __init__(self):
        """초기화"""
        self.scheduler = AsyncIOScheduler(timezone=SCHEDULE_TIMEZONE)
        self.quiz_generator = ObjectDetectionQuizGenerator()
        self.is_running = False
        self.last_execution = None
        self.next_execution = None
        self.execution_count = 0
        self.last_result = None
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def start_scheduler(self) -> Dict[str, Any]:
        """스케줄러 시작"""
        try:
            if self.is_running:
                return {
                    "status": "already_running",
                    "message": "스케줄러가 이미 실행 중입니다.",
                    "interval_hours": SCHEDULE_INTERVAL_HOURS
                }
            
            # 스케줄러 시작
            self.scheduler.start()
            
            # 주기적 작업 추가 (시간 간격 기반)
            job = self.scheduler.add_job(
                func=self._generate_scheduled_quizzes,
                trigger=IntervalTrigger(hours=SCHEDULE_INTERVAL_HOURS),
                id='quiz_generation_job',
                name='정기 퀴즈 생성',
                replace_existing=True,
                max_instances=1  # 중복 실행 방지
            )
            
            self.is_running = True
            self.next_execution = job.next_run_time
            
            logger.info(f"스케줄러 시작됨 - {SCHEDULE_INTERVAL_HOURS}시간마다 실행")
            
            return {
                "status": "started",
                "message": f"스케줄러가 시작되었습니다. {SCHEDULE_INTERVAL_HOURS}시간마다 실행됩니다.",
                "interval_hours": SCHEDULE_INTERVAL_HOURS,
                "next_execution": self.next_execution.isoformat() if self.next_execution else None,
                "timezone": SCHEDULE_TIMEZONE
            }
            
        except Exception as e:
            logger.error(f"스케줄러 시작 실패: {e}")
            return {
                "status": "error",
                "message": f"스케줄러 시작 실패: {str(e)}"
            }
    
    async def stop_scheduler(self) -> Dict[str, Any]:
        """스케줄러 중지"""
        try:
            if not self.is_running:
                return {
                    "status": "not_running",
                    "message": "스케줄러가 실행되고 있지 않습니다."
                }
            
            # 스케줄러 중지
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            self.next_execution = None
            
            logger.info("스케줄러 중지됨")
            
            return {
                "status": "stopped",
                "message": "스케줄러가 중지되었습니다.",
                "execution_count": self.execution_count,
                "last_execution": self.last_execution.isoformat() if self.last_execution else None
            }
            
        except Exception as e:
            logger.error(f"스케줄러 중지 실패: {e}")
            return {
                "status": "error",
                "message": f"스케줄러 중지 실패: {str(e)}"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        jobs = []
        if self.scheduler.running:
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                })
        
        return {
            "is_running": self.is_running,
            "scheduler_running": self.scheduler.running,
            "execution_count": self.execution_count,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "next_execution": self.next_execution.isoformat() if self.next_execution else None,
            "last_result": self.last_result,
            "interval_hours": SCHEDULE_INTERVAL_HOURS,
            "timezone": SCHEDULE_TIMEZONE,
            "enabled": SCHEDULE_ENABLED,
            "scheduled_counts": SCHEDULED_QUIZ_COUNTS,
            "jobs": jobs
        }
    
    async def execute_now(self) -> Dict[str, Any]:
        """즉시 퀴즈 생성 실행"""
        try:
            logger.info("수동 퀴즈 생성 실행 시작")
            
            start_time = datetime.now()
            total_generated = await self.quiz_generator.generate_scheduled_quizzes(SCHEDULED_QUIZ_COUNTS)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = {
                "status": "success",
                "message": f"퀴즈 생성 완료: {total_generated}개",
                "generated_count": total_generated,
                "execution_time_seconds": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "target_counts": SCHEDULED_QUIZ_COUNTS
            }
            
            logger.info(f"수동 퀴즈 생성 완료: {total_generated}개 ({execution_time:.1f}초)")
            
            return result
            
        except Exception as e:
            logger.error(f"수동 퀴즈 생성 실패: {e}")
            return {
                "status": "error",
                "message": f"퀴즈 생성 실패: {str(e)}",
                "generated_count": 0
            }
    
    async def _generate_scheduled_quizzes(self):
        """스케줄된 퀴즈 생성 (내부 메서드)"""
        try:
            logger.info("정기 퀴즈 생성 시작")
            
            start_time = datetime.now()
            total_generated = await self.quiz_generator.generate_scheduled_quizzes(SCHEDULED_QUIZ_COUNTS)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 실행 정보 업데이트
            self.execution_count += 1
            self.last_execution = start_time
            self.last_result = {
                "generated_count": total_generated,
                "execution_time_seconds": execution_time,
                "timestamp": start_time.isoformat()
            }
            
            # 다음 실행 시간 업데이트
            jobs = self.scheduler.get_jobs()
            if jobs:
                self.next_execution = jobs[0].next_run_time
            
            logger.info(f"정기 퀴즈 생성 완료: {total_generated}개 ({execution_time:.1f}초)")
            
        except Exception as e:
            logger.error(f"정기 퀴즈 생성 실패: {e}")
            self.last_result = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """리소스 정리"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        self.is_running = False
        logger.info("스케줄러 서비스 정리 완료")


# 글로벌 스케줄러 인스턴스
scheduler_service: Optional[SchedulerService] = None

def get_scheduler_service() -> SchedulerService:
    """스케줄러 서비스 싱글톤 인스턴스 반환"""
    global scheduler_service
    if scheduler_service is None:
        scheduler_service = SchedulerService()
    return scheduler_service

async def init_scheduler_service() -> SchedulerService:
    """스케줄러 서비스 초기화"""
    service = get_scheduler_service()
    
    # 자동 시작 설정이 활성화된 경우 스케줄러 시작
    if SCHEDULE_ENABLED:
        await service.start_scheduler()
        logger.info("스케줄러 서비스가 자동으로 시작되었습니다.")
    
    return service

async def cleanup_scheduler_service():
    """스케줄러 서비스 정리"""
    global scheduler_service
    if scheduler_service:
        await scheduler_service.cleanup()
        scheduler_service = None
