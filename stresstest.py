import os
import json
import random
import time
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptchaUser(HttpUser):
    """
    캡챠 API 스트레스 테스트를 위한 사용자 클래스
    """
    wait_time = between(1, 3)  # 요청 간 1-3초 대기
    
    def on_start(self):
        """사용자 시작 시 초기화"""
        self.api_key = os.getenv('API_KEY', '1d8664541d81347522a137cf3f1adf390aff0c0a99ac534a380b0d1794c18375')
        self.headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key
        }
        logger.info(f"사용자 시작 - API Key: {self.api_key[:10]}...")
    
    @task(70)
    def request_captcha_problem(self):
        """캡챠 문제 요청 (70% 비중)"""
        try:
            # API 스펙에 맞게 빈 body로 요청
            payload = {}
            
            with self.client.post(
                "/api/captcha/problem",
                json=payload,
                headers=self.headers,
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    response_data = response.json()
                    if 'clientToken' in response_data:
                        # 클라이언트 토큰을 세션에 저장
                        self.client_token = response_data['clientToken']
                        response.success()
                        logger.info(f"캡챠 문제 요청 성공: {self.client_token}")
                    else:
                        response.failure("응답에 clientToken이 없습니다")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            logger.error(f"캡챠 문제 요청 실패: {str(e)}")
    
    @task(25)
    def verify_captcha_answer(self):
        """캡챠 답변 검증 (25% 비중)"""
        try:
            # 이전에 요청한 문제가 있으면 사용, 없으면 랜덤 생성
            client_token = getattr(self, 'client_token', f"test_token_{random.randint(1000, 9999)}")
            
            # 실제 API 스펙에 맞는 payload (추정)
            payload = {
                "clientToken": client_token,
                "answer": random.choice(["컵", "가방", "냉장고", "코끼리"])  # 실제 옵션 중에서 선택
            }
            
            with self.client.post(
                "/api/captcha/verify",
                json=payload,
                headers=self.headers,
                catch_response=True
            ) as response:
                if response.status_code in [200, 400, 422]:
                    response.success()
                    logger.info(f"캡챠 검증 완료: {client_token}")
                else:
                    response.failure(f"HTTP {response.status_code}")
                    
        except Exception as e:
            logger.error(f"캡챠 검증 실패: {str(e)}")
    
    @task(5)
    def request_without_api_key(self):
        """API 키 없이 요청 (5% 비중) - 에러 케이스 테스트"""
        try:
            # API 스펙에 맞게 빈 body로 요청
            payload = {}
            
            with self.client.post(
                "/api/captcha/problem",
                json=payload,
                headers={'Content-Type': 'application/json'},  # API 키 제외
                catch_response=True
            ) as response:
                if response.status_code == 401:
                    response.success()  # 예상된 에러
                    logger.info("API 키 없이 요청 - 예상된 401 에러")
                else:
                    response.failure(f"예상과 다른 응답: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"API 키 없이 요청 실패: {str(e)}")

class DashboardUser(HttpUser):
    """
    대시보드 API 스트레스 테스트를 위한 사용자 클래스
    """
    wait_time = between(2, 5)
    
    def on_start(self):
        """사용자 시작 시 로그인"""
        self.api_key = os.getenv('API_KEY', '1d8664541d81347522a137cf3f1adf390aff0c0a99ac534a380b0d1794c18375')
        self.headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key
        }
        
        # 로그인 시도
        login_payload = {
            "email": "admin@admin.com",
            "password": "admin1234"
        }
        
        try:
            response = self.client.post(
                "/api/dashboard/auth/login",
                json=login_payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                if 'accessToken' in token_data:
                    self.headers['Authorization'] = f"Bearer {token_data['accessToken']}"
                    logger.info("대시보드 사용자 로그인 성공")
                else:
                    logger.warning("로그인 응답에 accessToken이 없습니다")
            else:
                logger.warning(f"로그인 실패: {response.status_code}")
                
        except Exception as e:
            logger.error(f"로그인 중 오류: {str(e)}")
    
    @task(40)
    def get_user_info(self):
        """사용자 정보 조회"""
        with self.client.get(
            "/api/dashboard/users/me",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"사용자 정보 조회 실패: {response.status_code}")
    
    @task(30)
    def get_applications(self):
        """애플리케이션 목록 조회"""
        with self.client.get(
            "/api/dashboard/applications/all",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"애플리케이션 목록 조회 실패: {response.status_code}")
    
    @task(20)
    def get_statistics(self):
        """통계 조회"""
        # 필수 파라미터: periodType (daily, weekly, monthly, yearly)
        period_types = ["daily", "weekly", "monthly", "yearly"]
        period_type = random.choice(period_types)
        
        # 선택적 파라미터들
        params = {
            "periodType": period_type
        }
        
        # 50% 확률로 keyId 추가
        if random.random() < 0.5:
            params["keyId"] = random.randint(1, 10)
        
        # 30% 확률로 날짜 범위 추가
        if random.random() < 0.3:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=random.randint(1, 30))
            params["startDate"] = start_date.strftime("%Y-%m-%d")
            params["endDate"] = end_date.strftime("%Y-%m-%d")
        
        with self.client.get(
            "/api/dashboard/statistics/summary",
            params=params,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"통계 조회 실패: {response.status_code}")
    
    @task(10)
    def get_api_keys(self):
        """API 키 목록 조회"""
        with self.client.get(
            "/api/dashboard/api-keys/all",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"API 키 목록 조회 실패: {response.status_code}")

# 이벤트 리스너 - 테스트 시작/종료 시 로그
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    logger.info("스트레스 테스트 초기화 완료")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("스트레스 테스트 시작")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("스트레스 테스트 종료")

# 실행 방법:
# 1. pip install locust
# 2. export API_KEY="your_api_key_here"
# 3. locust -f stresstest.py --host=http://your-api-host.com
# 4. 웹 브라우저에서 http://localhost:8089 접속하여 테스트 설정

def print_usage():
    print("캡챠 API 스트레스 테스트")
    print("\n사용법:")
    print("1. 웹 UI 모드:")
    print("   python stresstest.py --host=https://api.scratcha.cloud")
    print("\n2. 헤드리스 모드:")
    print("   python stresstest.py --host=https://api.scratcha.cloud --api-key=your_key --users=100 --spawn-rate=10 --run-time=10m --headless")
    print("\n3. 결과 저장:")
    print("   python stresstest.py --host=https://api.scratcha.cloud --api-key=your_key --users=100 --spawn-rate=10 --run-time=10m --headless --csv=results.csv")
    print("\n환경변수 설정:")
    print("   export API_KEY=your_api_key_here")
    print("\n예시:")
    print("   python stresstest.py --host=https://api.scratcha.cloud --api-key=sk-1234567890abcdef --users=50 --spawn-rate=5 --run-time=5m --headless")

if __name__ == "__main__":
    """
    독립 실행을 위한 메인 함수
    """
    import sys
    import subprocess
    import os
    
    # 명령행 인수 파싱
    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        sys.exit(0)
    
    # locust 명령어 구성
    cmd = ["locust", "-f", __file__]
    
    # 인수 처리
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == "--host":
            if i + 1 < len(sys.argv):
                cmd.extend(["--host", sys.argv[i + 1]])
                i += 2
            else:
                print("오류: --host 뒤에 호스트 URL이 필요합니다")
                sys.exit(1)
        
        elif arg == "--api-key":
            if i + 1 < len(sys.argv):
                os.environ['API_KEY'] = sys.argv[i + 1]
                i += 2
            else:
                print("오류: --api-key 뒤에 API 키가 필요합니다")
                sys.exit(1)
        
        elif arg == "--users":
            if i + 1 < len(sys.argv):
                cmd.extend(["--users", sys.argv[i + 1]])
                i += 2
            else:
                print("오류: --users 뒤에 사용자 수가 필요합니다")
                sys.exit(1)
        
        elif arg == "--spawn-rate":
            if i + 1 < len(sys.argv):
                cmd.extend(["--spawn-rate", sys.argv[i + 1]])
                i += 2
            else:
                print("오류: --spawn-rate 뒤에 생성률이 필요합니다")
                sys.exit(1)
        
        elif arg == "--run-time":
            if i + 1 < len(sys.argv):
                cmd.extend(["--run-time", sys.argv[i + 1]])
                i += 2
            else:
                print("오류: --run-time 뒤에 실행 시간이 필요합니다")
                sys.exit(1)
        
        elif arg == "--headless":
            cmd.append("--headless")
            i += 1
        
        elif arg == "--csv":
            if i + 1 < len(sys.argv):
                cmd.extend(["--csv", sys.argv[i + 1]])
                i += 2
            else:
                print("오류: --csv 뒤에 파일명이 필요합니다")
                sys.exit(1)
        
        else:
            # 알 수 없는 인수는 locust에 전달
            cmd.append(arg)
            i += 1
    
    # 기본값 설정 - --host 인수 확인
    host_found = False
    for i in range(len(cmd)):
        if cmd[i] == "--host":
            host_found = True
            break
    
    if not host_found:
        print("오류: --host 인수가 필요합니다")
        print_usage()
        sys.exit(1)
    
    if "--headless" in cmd:
        # 헤드리스 모드에서는 필수 인수 확인
        required_args = ["--users", "--spawn-rate", "--run-time"]
        missing_args = [arg for arg in required_args if arg not in cmd]
        if missing_args:
            print(f"오류: 헤드리스 모드에서는 다음 인수가 필요합니다: {', '.join(missing_args)}")
            print_usage()
            sys.exit(1)
    
    try:
        print("스트레스 테스트 시작...")
        print(f"명령어: {' '.join(cmd)}")
        print()
        
        # locust 실행
        subprocess.run(cmd, check=True)
        print("\n스트레스 테스트 완료!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n스트레스 테스트 실행 중 오류 발생: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n사용자에 의해 테스트가 중단되었습니다.")
        sys.exit(0)
    except FileNotFoundError:
        print("\n오류: locust가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("pip install locust")
        sys.exit(1)
