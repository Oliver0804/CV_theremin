import cv2
import numpy as np
import mediapipe as mp
import pyaudio
import math
import threading
from collections import deque

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to generate sine wave audio
def generate_wave(frequency, volume, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = volume * np.sin(2 * np.pi * frequency * t)
    audio = (wave * 32767).astype(np.int16).tobytes()
    return audio

def audio_callback(frequency_ref, volume_ref, sample_rate, chunk_size):
    def callback(in_data, frame_count, time_info, status):
        wave = generate_wave(frequency_ref[0], volume_ref[0], sample_rate, frame_count / sample_rate)
        return (wave, pyaudio.paContinue)
    return callback

# Initialize PyAudio
sample_rate = 44100
chunk_size = 1024
p = pyaudio.PyAudio()

# Start audio stream
frequency = [440]  # Initial frequency
volume = [0.5]     # Initial volume (range 0-1)

audio_stream = p.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=sample_rate,
                      output=True,
                      frames_per_buffer=chunk_size,
                      stream_callback=audio_callback(frequency, volume, sample_rate, chunk_size))

audio_stream.start_stream()

def detect_thumb_index_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + 
                         (thumb_tip.y - index_tip.y) ** 2 +
                         (thumb_tip.z - index_tip.z) ** 2)
    return distance < 0.05

# Trail settings
trail_length = 20
left_trail = deque(maxlen=trail_length)
right_trail = deque(maxlen=trail_length)

# Load and mirror PNG image
png_image = cv2.imread('play.png', cv2.IMREAD_UNCHANGED)
png_image = cv2.flip(png_image, 1)  # Mirror the image

# Function to overlay PNG image on background
def overlay_image(background, overlay):
    overlay_resized = cv2.resize(overlay, (background.shape[1], background.shape[0]))
    h, w = overlay_resized.shape[0], overlay_resized.shape[1]
    overlay_image = overlay_resized[:, :, :3]
    mask = overlay_resized[:, :, 3:] / 255.0
    background[:] = (1.0 - mask) * background + mask * overlay_image

# Start video capture
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        # Convert RGB back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check left or right hand
                handedness = results.multi_handedness[idx].classification[0].label
                hand_type = "Right" if handedness == "Right" else "Left"

                if detect_thumb_index_pinch(hand_landmarks):
                    # Get index finger tip position
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    finger_x = int(index_finger_tip.x * image.shape[1])
                    finger_y = int(index_finger_tip.y * image.shape[0])
                    
                    if hand_type == "Left":
                        left_trail.append((finger_x, finger_y))
                        frequency[0] = 1000 - (finger_x / image.shape[1] * 800)  # Left to right: high to low frequency
                    elif hand_type == "Right":
                        right_trail.append((finger_x, finger_y))
                        volume[0] = 1 - finger_y / image.shape[0]  # Top to bottom: high to low volume
                else:
                    if hand_type == "Left":
                        frequency[0] = 0
                    elif hand_type == "Right":
                        volume[0] = 0
        else:
            frequency[0] = 0
            volume[0] = 0

        # Draw left hand trail
        for i, (tx, ty) in enumerate(left_trail):
            alpha = (i + 1) / trail_length
            cv2.circle(image, (tx, ty), int(15 * alpha), (0, 0, 255), -1)

        # Draw right hand trail
        for i, (tx, ty) in enumerate(right_trail):
            alpha = (i + 1) / trail_length
            cv2.circle(image, (tx, ty), int(15 * alpha), (255, 0, 0), -1)

        # Overlay PNG image on the frame
        overlay_image(image, png_image)  # Stretch to cover the entire frame

        cv2.imshow('Theremin', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    audio_stream.stop_stream()
    audio_stream.close()
    p.terminate()
