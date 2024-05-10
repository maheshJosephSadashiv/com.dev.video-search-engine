import av
import logging
import numpy as np
from sklearn.cluster import KMeans
import cv2
from moviepy.editor import VideoFileClip

# Suppress av logging
av.logging.set_level(av.logging.ERROR)


def extract_i_frames(video_path):
    data = []
    container = av.open(video_path)
    stream = container.streams.video[0]  # Assuming the first stream is a video stream.

    frame_idx = 0
    i_frame_idx = 0

    for frame in container.decode(stream):
        if data:
            data[-1]["end_timestamp"] = frame.time
        if frame.key_frame:
            data.append({
                "image": frame.to_ndarray(format='bgr24'),
                "start_timestamp": frame.time,
                "id": frame_idx})
            i_frame_idx += 1
        frame_idx += 1
    if data:
        data[-1]["end_timestamp"] = None

    print(f"Total frames: {frame_idx}, I-frames extracted: {i_frame_idx}")
    return data


def extract_frames(video_path):
    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = True
    frames = []
    frame_rate = vidObj.get(cv2.CAP_PROP_FPS)
    while success:
        success, image = vidObj.read()
        frames.append({'image': image, 'start_timestamp': count / frame_rate, 'id': count})
        count += 1
    return frames


def find_dominant_colors(frame, k=5):
    data = np.reshape(frame, (-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(data)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors


def calculate_variance(frame):
    means, stds = cv2.meanStdDev(frame)
    variance = stds
    return variance.flatten()


def process_audio_from_video(video_file_path, audio_file_path):
    # Extract Audio
    print("Extracting audio from video: ", video_file_path)
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file_path)
    audio_clip.close()
    video_clip.close()
