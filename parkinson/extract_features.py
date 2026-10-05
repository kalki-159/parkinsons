import os
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import cv2

# MediaPipe imports for v0.10+
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat


class HandFeatureExtractor:
    def __init__(
        self,
        model_path: str = None,
        min_detection_confidence: float = 0.15,
        min_tracking_confidence: float = 0.15,
        num_hands: int = 2
    ):
        """
        Initialize the hand feature extractor with MediaPipe Tasks API.
        Uses IMAGE mode for robust detection at any distance/size.
        
        Args:
            model_path: Path to the hand_landmarker.task model file
            min_detection_confidence: Minimum confidence for hand detection
            num_hands: Maximum number of hands to detect
        """
        # Default model path
        if model_path is None:
            model_path = Path(__file__).parent / "hand_landmarker.task"
        
        self.model_path = str(model_path)
        
        # Verify model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {self.model_path}\n"
                "Please download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        
        # Configure MediaPipe Hand Landmarker for robust detection
        base_options = BaseOptions(model_asset_path=self.model_path)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,  # Use IMAGE mode for better detection
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.1,  # Very low threshold
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # Keypoint indices
        self.WRIST = 0
        self.INDEX_TIP = 8
        
        # Upscaling threshold - videos smaller than this will be upscaled
        self.UPSCALE_MIN_HEIGHT = 480
        self.UPSCALE_TARGET_HEIGHT = 720  # Upscale to this for detection
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except Exception:
                pass
    
    def _convert_to_mp_image(self, frame: np.ndarray) -> Image:
        """Convert numpy array to MediaPipe Image"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    
    def _upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale small frames for better detection"""
        height, width = frame.shape[:2]
        
        if height < self.UPSCALE_MIN_HEIGHT:
            scale_factor = self.UPSCALE_TARGET_HEIGHT / height
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return frame
    
    def detect_hands(self, frame: np.ndarray) -> Any:
        """
        Detect hands in a single frame using IMAGE mode.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            
        Returns:
            HandLandmarkerResult with hand landmarks
        """
        # Upscale if needed
        frame = self._upscale_frame(frame)
        mp_image = self._convert_to_mp_image(frame)
        return self.landmarker.detect(mp_image)
    
    def extract_features_from_video(self, video_path: str) -> Dict[str, Any]:
        """
        Extract hand features from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with extracted features and metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                'success': False,
                'error': 'Could not open video file',
                'hand_detected': False
            }
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if video_fps <= 0:
            video_fps = 30.0
        
        # Storage for keypoint trajectories
        left_wrist_trajectory = []
        right_wrist_trajectory = []
        left_index_trajectory = []
        right_index_trajectory = []
        
        frame_count = 0
        frames_with_hands = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect hands using IMAGE mode
            try:
                result = self.detect_hands(frame)
                
                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    frames_with_hands += 1
                    
                    for idx, landmarks in enumerate(result.hand_landmarks):
                        # Determine hand type
                        if idx < len(result.handedness):
                            handedness = result.handedness[idx]
                            hand_type = handedness[0].category_name
                        else:
                            hand_type = "Left" if idx == 0 else "Right"
                        
                        # Extract wrist and index tip
                        wrist = landmarks[self.WRIST]
                        index_tip = landmarks[self.INDEX_TIP]
                        
                        if hand_type == 'Left':
                            left_wrist_trajectory.append([wrist.x, wrist.y, wrist.z])
                            left_index_trajectory.append([index_tip.x, index_tip.y, index_tip.z])
                        else:
                            right_wrist_trajectory.append([wrist.x, wrist.y, wrist.z])
                            right_index_trajectory.append([index_tip.x, index_tip.y, index_tip.z])
            except Exception:
                pass
            
            frame_count += 1
        
        cap.release()
        
        # Calculate detection ratio
        hand_detection_ratio = frames_with_hands / frame_count if frame_count > 0 else 0
        
        result = {
            'success': True,
            'video_fps': video_fps,
            'total_frames': total_frames,
            'frames_processed': frame_count,
            'frames_with_hands': frames_with_hands,
            'hand_detection_ratio': hand_detection_ratio,
            'hand_detected': frames_with_hands > 0,
            'left_hand_detected': len(left_wrist_trajectory) > 0,
            'right_hand_detected': len(right_wrist_trajectory) > 0,
            'video_duration_seconds': total_frames / video_fps if video_fps > 0 else 0,
            'video_resolution': f'{frame_width}x{frame_height}'
        }
        
        # Calculate features if hands detected
        if frames_with_hands > 0:
            tremor_features = self._calculate_tremor_features(
                left_wrist_trajectory,
                right_wrist_trajectory,
                left_index_trajectory,
                right_index_trajectory,
                video_fps
            )
            result.update(tremor_features)
        else:
            result['error'] = 'No hands detected in video'
        
        return result
    
    def _calculate_tremor_features(
        self,
        left_wrist: List[List[float]],
        right_wrist: List[List[float]],
        left_index: List[List[float]],
        right_index: List[List[float]],
        fps: float
    ) -> Dict[str, float]:
        """Calculate tremor features from keypoint trajectories"""
        
        # Use hand with more data
        wrist_data = left_wrist if len(left_wrist) > len(right_wrist) else right_wrist
        
        if len(wrist_data) < 10:
            return {'error': 'Insufficient frames with hands detected'}
        
        wrist_array = np.array(wrist_data)
        wrist_y = wrist_array[:, 1]
        
        # Calculate tremor metrics
        tremor_freq, freq_confidence = self._calculate_tremor_frequency(wrist_y, fps)
        amplitude = self._calculate_amplitude(wrist_y)
        rhythm = self._calculate_rhythm_consistency(wrist_y, fps, tremor_freq)
        smoothness = self._calculate_smoothness(wrist_y)
        velocity = self._calculate_velocity(wrist_y, fps)
        
        return {
            'tremor_frequency_hz': round(float(tremor_freq), 2),
            'frequency_confidence': round(float(freq_confidence), 2),
            'amplitude_mm': round(float(amplitude), 2),
            'rhythm_consistency': round(float(rhythm), 2),
            'movement_smoothness': round(float(smoothness), 2),
            'average_velocity': round(float(velocity), 4),
            'dominant_hand': 'left' if len(left_wrist) > len(right_wrist) else 'right',
            'hand_frames_used': len(wrist_data)
        }
    
    def _calculate_tremor_frequency(self, signal_data: np.ndarray, fps: float) -> Tuple[float, float]:
        """Calculate dominant tremor frequency using FFT"""
        n = len(signal_data)
        if n < 32:
            return 0.0, 0.0
        
        # Detrend and window
        signal_detrended = signal.detrend(signal_data)
        window = np.hanning(n)
        signal_windowed = signal_detrended * window
        
        # FFT
        fft_values = np.abs(fft(signal_windowed))
        freqs = fftfreq(n, 1/fps)
        
        # Positive frequencies only
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_fft = fft_values[positive_mask]
        
        # Focus on Parkinson's tremor range (4-8 Hz)
        tremor_range_mask = (positive_freqs >= 4) & (positive_freqs <= 8)
        
        if not np.any(tremor_range_mask):
            tremor_range_mask = positive_mask
        
        tremor_freqs = positive_freqs[tremor_range_mask]
        tremor_fft = positive_fft[tremor_range_mask]
        
        if len(tremor_fft) == 0:
            return 0.0, 0.0
        
        # Dominant frequency
        dominant_idx = np.argmax(tremor_fft)
        dominant_freq = float(tremor_freqs[dominant_idx])
        
        # Confidence
        mean_power = np.mean(tremor_fft)
        peak_power = tremor_fft[dominant_idx]
        confidence = min(100, (peak_power / mean_power) * 20) if mean_power > 0 else 0
        
        return dominant_freq, confidence
    
    def _calculate_amplitude(self, signal_data: np.ndarray) -> float:
        """Calculate tremor amplitude"""
        std_dev = np.std(signal_data)
        amplitude_mm = std_dev * 200
        return amplitude_mm
    
    def _calculate_rhythm_consistency(self, signal_data: np.ndarray, fps: float, tremor_freq: float) -> float:
        """Calculate rhythm consistency via autocorrelation"""
        n = len(signal_data)
        if n < 32 or tremor_freq <= 0:
            return 50.0
        
        expected_period = fps / tremor_freq
        signal_normalized = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)
        
        autocorr = np.correlate(signal_normalized, signal_normalized, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0]
        
        lag_idx = int(round(expected_period))
        if lag_idx < len(autocorr):
            rhythm_score = max(0, min(100, autocorr[lag_idx] * 100))
        else:
            rhythm_score = 50.0
        
        return rhythm_score
    
    def _calculate_smoothness(self, signal_data: np.ndarray) -> float:
        """Calculate movement smoothness (jerk-based)"""
        if len(signal_data) < 4:
            return 0.0
        
        velocity = np.diff(signal_data)
        acceleration = np.diff(velocity)
        jerk = np.diff(acceleration)
        
        signal_range = np.max(signal_data) - np.min(signal_data) + 1e-10
        smoothness = 100 / (1 + np.var(jerk) / (signal_range ** 2))
        
        return min(100, smoothness)
    
    def _calculate_velocity(self, signal_data: np.ndarray, fps: float) -> float:
        """Calculate average movement velocity"""
        if len(signal_data) < 2:
            return 0.0
        
        velocity = np.abs(np.diff(signal_data))
        return np.mean(velocity) * fps
    
    def get_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to model input vector"""
        feature_order = [
            'tremor_frequency_hz',
            'amplitude_mm',
            'rhythm_consistency',
            'movement_smoothness',
            'average_velocity',
            'frequency_confidence'
        ]
        
        vector = [features.get(key, 0.0) for key in feature_order]
        return np.array(vector).reshape(1, -1)


def extract_features(video_path: str) -> Dict[str, Any]:
    """Convenience function to extract features from a video"""
    extractor = HandFeatureExtractor()
    return extractor.extract_features_from_video(video_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Analyzing video: {video_path}")
        
        try:
            result = extract_features(video_path)
            
            print("\n=== RESULTS ===")
            for key, value in result.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python extract_features.py <video_path>")
