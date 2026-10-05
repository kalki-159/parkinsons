import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

# Initialize landmarker with lower confidence
base_options = BaseOptions(model_asset_path=r'C:\Users\Dell\OneDrive\Desktop\parkinson\hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,  # Lower threshold
    min_hand_presence_confidence=0.3,    # Lower threshold
    min_tracking_confidence=0.3          # Lower threshold
)
landmarker = HandLandmarker.create_from_options(options)

# Analyze mini1.mp4
cap = cv2.VideoCapture(r'C:\Users\Dell\OneDrive\Desktop\mini1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"=== Video Properties ===")
print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Total frames: {total_frames}")
print(f"Duration: {total_frames/fps:.2f}s")
print()

# Sample frames at different positions
sample_positions = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]

for pos in sample_positions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((pos / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands_detected = len(result.hand_landmarks) if result.hand_landmarks else 0
        print(f"Frame {pos}: {hands_detected} hands detected", end="")
        
        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                wrist = landmarks[0]
                print(f"  | Hand {i+1} wrist: ({wrist.x:.3f}, {wrist.y:.3f})", end="")
        print()

cap.release()

# Now test with full pipeline
print()
print("=== Testing Full Detection ===")
cap = cv2.VideoCapture(r'C:\Users\Dell\OneDrive\Desktop\mini1.mp4')
frame_count = 0
frames_with_hands = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((frame_count / fps) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        frames_with_hands += 1
    
    frame_count += 1

cap.release()
landmarker.close()

print(f"Total frames: {frame_count}")
print(f"Frames with hands: {frames_with_hands}")
print(f"Detection ratio: {frames_with_hands/frame_count*100:.1f}%")
