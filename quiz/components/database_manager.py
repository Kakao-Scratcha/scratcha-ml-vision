"""
데이터베이스 관리 모듈
"""

import os
import pymysql
import threading
from datetime import datetime, timedelta
from typing import Dict
from config.settings import DB_CONFIG, DIFFICULTY_TO_NUMBER, PROMPT


class DatabaseManager:
    """MySQL 데이터베이스 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        self._lock = threading.Lock()  # 스레드 안전성을 위한 락
        print("DatabaseManager 초기화 완료")
    
    def get_db_connection(self):
        """
        PyMySQL을 사용한 MySQL 데이터베이스 연결 생성
        
        Returns:
            pymysql.Connection: 데이터베이스 연결 객체
        """
        try:
            # PyMySQL 연결 설정
            config = {
                'host': os.getenv('MYSQL_HOST'),
                'port': int(os.getenv('MYSQL_PORT', 3306)),
                'database': os.getenv('MYSQL_DATABASE'),
                'user': os.getenv('MYSQL_USER'),
                'password': os.getenv('MYSQL_PASSWORD'),
                'charset': 'utf8mb4',
                'autocommit': True,
                'connect_timeout': 60
            }
            

            
            connection = pymysql.connect(**config)
            print(f"PyMySQL 연결 성공 - 서버: {config['host']}:{config['port']}")
            return connection
            
        except Exception as e:
            print(f"PyMySQL 연결 실패: {e}")

            raise
    
    def init_database(self):
        """데이터베이스 테이블 초기화"""
        connection = None
        cursor = None
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            # captcha_problem 테이블 생성
            create_table_query = """
            CREATE TABLE IF NOT EXISTS captcha_problem (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_url TEXT NOT NULL,
                answer VARCHAR(20) NOT NULL,
                prompt VARCHAR(255) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NULL,
                difficulty INT NOT NULL DEFAULT 1 COMMENT '0: low, 1: middle, 2: high',
                wrong_answer_1 VARCHAR(20) NOT NULL,
                wrong_answer_2 VARCHAR(20) NOT NULL,
                wrong_answer_3 VARCHAR(20) NOT NULL,
                INDEX idx_difficulty (difficulty),
                INDEX idx_created_at (created_at),
                INDEX idx_expires_at (expires_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            cursor.execute(create_table_query)
            connection.commit()
            print("captcha_problem 테이블 생성/확인 완료")
            
        except Exception as e:
            print(f"데이터베이스 초기화 실패: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
    
    def save_quiz_to_database(self, quiz_data: Dict):
        """
        퀴즈 데이터를 MySQL 데이터베이스에 저장 (스레드 안전)
        
        Args:
            quiz_data: 퀴즈 데이터
        """
        with self._lock:  # 스레드 안전성을 위한 락
            connection = None
            cursor = None
            try:
                connection = self.get_db_connection()
                cursor = connection.cursor()
                
                # 옵션에서 정답과 오답 분리
                correct_option = None
                wrong_options = []
                
                for option in quiz_data['options']:
                    if option['is_correct']:
                        correct_option = option
                    else:
                        wrong_options.append(option)
                
                # 난이도를 숫자로 변환
                difficulty_str = quiz_data.get('difficulty', 'middle')  # 기본값은 middle
                difficulty_number = DIFFICULTY_TO_NUMBER.get(difficulty_str, 1)  # 기본값은 1 (middle)
                
                # MySQL에 저장할 데이터 준비 (id는 자동 생성)
                insert_query = """
                INSERT INTO captcha_problem 
                (image_url, answer, prompt, created_at, expires_at, difficulty, 
                 wrong_answer_1, wrong_answer_2, wrong_answer_3)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                quiz_record = (
                    quiz_data['image_url'],                    # image_url
                    correct_option['class_name'],              # answer
                    quiz_data.get('prompt', PROMPT),          # prompt
                    datetime.now(),                            # created_at
                    datetime.now() + timedelta(days=1),       # expires_at (1일 후)
                    difficulty_number,                         # difficulty
                    wrong_options[0]['class_name'] if len(wrong_options) > 0 else '',  # wrong_answer_1
                    wrong_options[1]['class_name'] if len(wrong_options) > 1 else '',  # wrong_answer_2
                    wrong_options[2]['class_name'] if len(wrong_options) > 2 else ''   # wrong_answer_3
                )
                
                cursor.execute(insert_query, quiz_record)
                connection.commit()
                
                # 자동 생성된 ID 가져오기
                generated_id = cursor.lastrowid
                print(f"✓ 퀴즈 데이터 MySQL DB 저장 완료: ID {generated_id}")
                
                return generated_id
                
            except Exception as e:
                print(f"MySQL 저장 실패: {e}")
                if connection:
                    connection.rollback()
                raise
            finally:
                if cursor:
                    cursor.close()
                if connection:
                    connection.close()
