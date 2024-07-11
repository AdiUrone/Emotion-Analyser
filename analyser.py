import cv2
import mediapipe as mp
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path to the video you want to analyze
video_path = '/content/Untitled design.mp4'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_frames_from_video(video_path, interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_timestamps = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
            frame_timestamps.append(frame_count / fps)
        frame_count += 1

    cap.release()
    return frames, frame_timestamps

def analyze_and_display_emotions_and_hands(frames, frame_timestamps):
    consolidated_results = []

    for idx, (frame, timestamp) in enumerate(zip(frames, frame_timestamps)):
        try:
            # Convert the frame to RGB (cv2 loads images in BGR format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand movements
            hand_results = hands.process(frame_rgb)
            hand_landmarks = []
            if hand_results.multi_hand_landmarks:
                for hand_landmark in hand_results.multi_hand_landmarks:
                    hand_landmarks.append([(lm.x, lm.y) for lm in hand_landmark.landmark])
            hand_movement = bool(hand_landmarks)

            # Analyzing the frame for emotion detection
            result = DeepFace.analyze(img_path=frame_rgb, actions=['emotion'])

            # Check if result is a list and retrieve emotions from the first element
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]  # Assuming the first result contains the desired data
                if 'emotion' in first_result:
                    emotions = first_result['emotion']
                    consolidated_results.append({
                        'frame_index': idx,
                        'timestamp': timestamp,
                        'emotions': emotions,
                        'hand_movement': hand_movement
                    })
                else:
                    print("Emotion key not found in the first result:", first_result)
            else:
                print("No valid results found for frame", idx)

        except Exception as e:
            print(f"Error analyzing face for frame {idx}: {e}")

    if consolidated_results:
        # Displaying consolidated frames with annotations
        plt.figure(figsize=(15, 10))
        for idx, result in enumerate(consolidated_results):
            frame_index = result['frame_index']
            emotions = result['emotions']
            timestamp = result['timestamp']
            hand_movement = result['hand_movement']

            # Convert frame back to RGB for display
            frame_rgb = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2RGB)

            plt.subplot(len(consolidated_results), 1, idx + 1)
            plt.imshow(frame_rgb)
            plt.axis('off')
            plt.title(f'Emotions and Hand Movements for Frame {frame_index} at {timestamp:.2f}s')

            # Add text annotations for each emotion detected
            for emotion, value in emotions.items():
                plt.text(10, 20 + list(emotions.keys()).index(emotion) * 20, f'{emotion}: {value:.2f}', fontsize=12, color='red', weight='bold')

            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * frame_rgb.shape[1]), int(lm.y * frame_rgb.shape[0])
                        cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), -1)
                    plt.imshow(frame_rgb)

        plt.tight_layout()
        plt.show()

        # Creating the classification table
        table_data = []
        hand_movement_counts = 0
        for result in consolidated_results:
            row = {'Frame Index': result['frame_index'], 'Timestamp (s)': result['timestamp'], 'Hand Movement': result['hand_movement']}
            hand_movement_counts += result['hand_movement']
            row.update(result['emotions'])
            table_data.append(row)

        df = pd.DataFrame(table_data)
        print("\nClassification Table:")
        print(df)

        # Creating the bar graph
        emotion_sums = df.drop(columns=['Frame Index', 'Timestamp (s)', 'Hand Movement']).sum()
        emotion_sums['Hand Movement'] = hand_movement_counts
        plt.figure(figsize=(12, 6))
        emotion_sums.plot(kind='bar', color='skyblue')
        plt.title('Total Emotion Scores and Hand Movements for All Frames')
        plt.xlabel('Emotion')
        plt.ylabel('Total Score / Count')
        plt.show()

        # Creating the line graph
        df.set_index('Frame Index', inplace=True)
        plt.figure(figsize=(12, 6))
        for emotion in df.columns:
            if emotion not in ['Timestamp (s)', 'Hand Movement']:
                plt.plot(df.index, df[emotion], label=emotion)
        plt.plot(df.index, df['Hand Movement'].astype(int) * df[emotion].max(), label='Hand Movement', linestyle='--')
        plt.title('Emotion Scores and Hand Movements Over Frames')
        plt.xlabel('Frame Index')
        plt.ylabel('Emotion Score / Hand Movement')
        plt.legend()
        plt.show()

# Extract frames from the video
frames, frame_timestamps = extract_frames_from_video(video_path)

# Analyze and display emotions and hand movements for all extracted frames
analyze_and_display_emotions_and_hands(frames, frame_timestamps)