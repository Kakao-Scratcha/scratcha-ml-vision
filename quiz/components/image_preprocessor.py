import cv2
import numpy as np
from typing import Tuple

# 기본 이미지 크기 설정
IMAGE_SIZE = 300



class ImagePreprocessor:
    """
    이미지 전처리 클래스
    """
    
    def __init__(self, target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)):
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray, frequency: float = 0.1, 
                  intensity: float = 0.1, alpha: float = 0.3, 
                  output_path: str = None) -> np.ndarray:
        """
        이미지 전처리 - 순차적 노이즈 적용 (랜덤 픽셀 → 솔트앤페퍼 → 고주파)
        
        Args:
            image: 원본 이미지
            frequency: 고주파 컬러 노이즈 강도
            intensity: 노이즈 강도
            alpha: 알파 블랜딩 강도
            output_path: 저장 경로
            
        Returns:
            전처리된 이미지
        """

        # 이미지 리사이즈
        resized = self.resizeImage(image)
        
        # 1단계: 랜덤 픽셀 노이즈 적용
        random_noise = self.randomColorNoise(resized, intensity)
        step1_image = self.alphaBlend(resized, random_noise, alpha)
        
        # 2단계: 솔트앤페퍼 노이즈 적용 (랜덤 노이즈 위에)
        salt_pepper_noise = self.saltAndPepperNoise(step1_image, intensity)
        step2_image = self.alphaBlend(step1_image, salt_pepper_noise, alpha)
        
        # 3단계: 고주파 노이즈 적용 (솔트앤페퍼 위에)
        high_frequency_noise = self.highFrequencyColorNoise(self.target_size, intensity, frequency)
        processed = self.alphaBlend(step2_image, high_frequency_noise, alpha)
        
        return processed
    
    def resizeImage(self, image: np.ndarray) -> np.ndarray:
        """비율을 유지하며 이미지 크기 조정"""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        aspect_ratio = w / h
        target_aspect_ratio = target_w / target_h
        
        if aspect_ratio > target_aspect_ratio:
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def highFrequencyColorNoise(self, size: Tuple[int, int], intensity: float = 0.1, 
                               frequency: float = 0.1) -> np.ndarray:
        """고주파 컬러 노이즈 생성"""
        h, w = size
        # 고주파 노이즈 생성 (더 강한 효과)
        base_noise = np.random.randn(h, w, 3)
        high_freq_noise = np.zeros_like(base_noise)
        
        for channel in range(3):
            channel_noise = base_noise[:, :, channel]
            # 더 작은 커널로 고주파 성분 추출
            low_freq = cv2.GaussianBlur(channel_noise, (5, 5), 1)
            high_freq = channel_noise - low_freq
            high_freq_noise[:, :, channel] = high_freq * frequency * 10 
        
        # 노이즈 정규화 및 스케일링
        noise_normalized = (high_freq_noise - high_freq_noise.min()) / (high_freq_noise.max() - high_freq_noise.min())
        noise_scaled = (noise_normalized * 255 * intensity).astype(np.uint8)
        return noise_scaled
    
    def randomColorNoise(self, image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """랜덤 컬러 노이즈 생성"""
        h, w = image.shape[:2]
        # 랜덤 노이즈 생성
        noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        # intensity에 따라 원본 이미지와 노이즈를 혼합
        if intensity > 0:
            # 원본 이미지와 노이즈를 알파 블렌딩
            image_float = image.astype(np.float32)
            noise_float = noise.astype(np.float32)
            blended = (1 - intensity) * image_float + intensity * noise_float
            return np.clip(blended, 0, 255).astype(np.uint8)
        else:
            return image
    
    def saltAndPepperNoise(self, image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """솔트 앤 페퍼 노이즈 생성"""
        h, w = image.shape[:2]
        noise = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 솔트(흰색) 노이즈
        salt_mask = np.random.random((h, w)) < intensity * 0.5
        noise[salt_mask] = 255
        
        # 페퍼(검은색) 노이즈
        pepper_mask = np.random.random((h, w)) < intensity * 0.5
        noise[pepper_mask] = 0
        
        return noise
    

    
    def alphaBlend(self, image: np.ndarray, noise: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """알파 블랜딩 (단일 노이즈용)"""
        image_float = image.astype(np.float32)
        noise_float = noise.astype(np.float32)
        
        blended = (1 - alpha) * image_float + alpha * noise_float
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def hybridDenoising(self, noisy_image: np.ndarray, denoise_strength: str = 'medium') -> np.ndarray:
        """
        하이브리드 디노이징 (가우시안 블러 + 바이레터럴 필터 + 메디안 필터 조합)
        
        Args:
            noisy_image: 노이즈가 있는 이미지
            denoise_strength: 디노이징 강도 ('light', 'medium', 'strong')
            
        Returns:
            디노이징된 이미지
        """
        # 강도별 파라미터 설정
        if denoise_strength == 'light':
            gaussian_kernel = (3, 3)
            gaussian_sigma = 0.5
            bilateral_d = 5
            bilateral_sigma = 30
            median_kernel = 3
        elif denoise_strength == 'medium':
            gaussian_kernel = (5, 5)
            gaussian_sigma = 1.0
            bilateral_d = 7
            bilateral_sigma = 50
            median_kernel = 3
        elif denoise_strength == 'strong':
            gaussian_kernel = (7, 7)
            gaussian_sigma = 1.5
            bilateral_d = 9
            bilateral_sigma = 75
            median_kernel = 5
        else:
            # 기본값 (medium)
            gaussian_kernel = (5, 5)
            gaussian_sigma = 1.0
            bilateral_d = 7
            bilateral_sigma = 50
            median_kernel = 3
        
        # 1단계: 가우시안 블러 (전체적인 노이즈 감소)
        step1 = cv2.GaussianBlur(noisy_image, gaussian_kernel, gaussian_sigma)
        
        # 2단계: 바이레터럴 필터 (엣지 보존하면서 노이즈 제거)
        step2 = cv2.bilateralFilter(step1, bilateral_d, bilateral_sigma, bilateral_sigma)
        
        # 3단계: 메디안 필터 (솔트앤페퍼 노이즈 제거)
        step3 = cv2.medianBlur(step2, median_kernel)
        
        return step3
    
    def adaptiveDenoising(self, noisy_image: np.ndarray) -> np.ndarray:
        """
        적응형 디노이징 (이미지 특성에 따라 자동 조정)
        
        Args:
            noisy_image: 노이즈가 있는 이미지
            
        Returns:
            디노이징된 이미지
        """
        # 이미지 노이즈 레벨 추정
        gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 노이즈 레벨에 따른 강도 결정
        if laplacian_var < 100:
            strength = 'light'
        elif laplacian_var < 500:
            strength = 'medium'
        else:
            strength = 'strong'
            
        print(f"   적응형 디노이징: 노이즈 레벨 {laplacian_var:.1f} -> {strength} 강도 적용")
        return self.hybridDenoising(noisy_image, strength)

