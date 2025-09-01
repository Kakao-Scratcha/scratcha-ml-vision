#!/usr/bin/env python3
"""
FastAPI 기반 퀴즈 생성 서비스
크론잡 대신 스케줄러를 사용하여 정기적으로 퀴즈를 생성합니다.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services import get_scheduler_service, init_scheduler_service, cleanup_scheduler_service
from config.settings import (
    API_HOST, 
    API_PORT, 
    SCHEDULE_ENABLED,
    SCHEDULE_INTERVAL_HOURS,
    SCHEDULED_QUIZ_COUNTS,
    SCHEDULED_QUIZ_COUNT
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic 모델들
class QuizGenerationRequest(BaseModel):
    """퀴즈 생성 요청 모델"""
    target_counts: Optional[Dict[str, int]] = Field(
        None,
        description="난이도별 생성할 퀴즈 개수 (None이면 기본값 사용)",
        example={"high": 3, "middle": 3, "low": 3}
    )

class SchedulerControlRequest(BaseModel):
    """스케줄러 제어 요청 모델"""
    action: str = Field(
        description="수행할 액션: start 또는 stop",
        example="start"
    )

class ApiResponse(BaseModel):
    """API 응답 기본 모델"""
    status: str = Field(description="응답 상태")
    message: str = Field(description="응답 메시지")
    data: Optional[Dict[str, Any]] = Field(None, description="응답 데이터")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# 애플리케이션 라이프사이클 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 컨텍스트 매니저"""
    # 시작 시
    logger.info("FastAPI 애플리케이션 시작 중...")
    
    try:
        # 스케줄러 서비스 초기화 (지연 로딩으로 빠르게 시작)
        await init_scheduler_service()
        logger.info("FastAPI 서버 시작 완료")
        logger.info("스케줄러 서비스 초기화 완료 (퀴즈 생성기는 필요시 로딩)")
        
        yield
        
    finally:
        # 종료 시
        logger.info("FastAPI 애플리케이션 종료 중...")
        await cleanup_scheduler_service()
        logger.info("스케줄러 서비스 정리 완료")

# FastAPI 앱 생성
app = FastAPI(
    title="퀴즈 생성 서비스",
    description="CAPTCHA 퀴즈를 정기적으로 생성하는 FastAPI 서비스",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=ApiResponse)
async def root():
    """루트 엔드포인트"""
    return ApiResponse(
        status="success",
        message="퀴즈 생성 서비스가 실행 중입니다.",
        data={
            "service": "quiz-generator",
            "version": "1.0.0",
            "schedule_enabled": SCHEDULE_ENABLED,
            "schedule_interval_hours": SCHEDULE_INTERVAL_HOURS
        }
    )

@app.get("/health", response_model=ApiResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    scheduler_service = get_scheduler_service()
    status = scheduler_service.get_status()
    
    return ApiResponse(
        status="healthy",
        message="서비스가 정상 작동 중입니다.",
        data={
            "service": "quiz-generator",
            "scheduler": status,
            "uptime": datetime.now().isoformat()
        }
    )

@app.get("/quiz/status", response_model=ApiResponse)
async def get_quiz_status():
    """퀴즈 생성 스케줄러 상태 조회"""
    try:
        scheduler_service = get_scheduler_service()
        status = scheduler_service.get_status()
        
        return ApiResponse(
            status="success",
            message="스케줄러 상태 조회 완료",
            data=status
        )
        
    except Exception as e:
        logger.error(f"스케줄러 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")

@app.post("/quiz/generate", response_model=ApiResponse)
async def generate_quiz_manual(
    request: QuizGenerationRequest,
    background_tasks: BackgroundTasks
):
    """수동 퀴즈 생성 엔드포인트"""
    try:
        scheduler_service = get_scheduler_service()
        
        # 요청된 생성 수량이 있으면 사용, 없으면 기본값 사용
        target_counts = request.target_counts or SCHEDULED_QUIZ_COUNTS
        
        logger.info(f"수동 퀴즈 생성 요청: {target_counts}")
        
        # 즉시 실행
        result = await scheduler_service.execute_now()
        
        if result["status"] == "success":
            return ApiResponse(
                status="success",
                message=result["message"],
                data={
                    "generated_count": result["generated_count"],
                    "execution_time_seconds": result["execution_time_seconds"],
                    "target_counts": result["target_counts"]
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"수동 퀴즈 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 실패: {str(e)}")

@app.post("/quiz/schedule/start", response_model=ApiResponse)
async def start_scheduler():
    """스케줄러 시작"""
    try:
        scheduler_service = get_scheduler_service()
        result = await scheduler_service.start_scheduler()
        
        if result["status"] in ["started", "already_running"]:
            return ApiResponse(
                status="success",
                message=result["message"],
                data=result
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"스케줄러 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=f"스케줄러 시작 실패: {str(e)}")

@app.post("/quiz/schedule/stop", response_model=ApiResponse)
async def stop_scheduler():
    """스케줄러 중지"""
    try:
        scheduler_service = get_scheduler_service()
        result = await scheduler_service.stop_scheduler()
        
        if result["status"] in ["stopped", "not_running"]:
            return ApiResponse(
                status="success",
                message=result["message"],
                data=result
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"스케줄러 중지 실패: {e}")
        raise HTTPException(status_code=500, detail=f"스케줄러 중지 실패: {str(e)}")

@app.get("/quiz/config", response_model=ApiResponse)
async def get_quiz_config():
    """퀴즈 생성 설정 조회"""
    return ApiResponse(
        status="success",
        message="퀴즈 생성 설정 조회 완료",
        data={
            "schedule_enabled": SCHEDULE_ENABLED,
            "schedule_interval_hours": SCHEDULE_INTERVAL_HOURS,
            "scheduled_quiz_count": SCHEDULED_QUIZ_COUNT,
            "scheduled_quiz_counts": SCHEDULED_QUIZ_COUNTS,
            "api_host": API_HOST,
            "api_port": API_PORT
        }
    )

# 전역 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리기"""
    logger.error(f"예상치 못한 오류 발생: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "서버 내부 오류가 발생했습니다.",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"퀴즈 생성 서비스 시작 - http://{API_HOST}:{API_PORT}")
    logger.info(f"스케줄링 활성화: {SCHEDULE_ENABLED}")
    logger.info(f"스케줄링 간격: {SCHEDULE_INTERVAL_HOURS}시간")
    logger.info(f"기본 생성 수량: {SCHEDULED_QUIZ_COUNTS}")
    
    uvicorn.run(
        "fastapi_app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
