import os
import time

from concurrent.futures import ThreadPoolExecutor

import cv2
import librosa
import numpy as np
import pandas as pd

from src.constants import OUTPUT_DIR
from src.preprocessing.utils import calculate_variance, find_dominant_colors, extract_frames
from src.utils.normalization import normalize_np_array


def extract_freq_vectors(img, block_size=8):
    # Convert image to YCrCb and extract the Y channel
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    h, w = img_y.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    img_y = img_y[:h, :w]

    # Initialize lists to store DC and AC coefficients
    dc_coefficients = []
    # ac_coefficients = []

    # Process each 8x8 block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_y[i:i + block_size, j:j + block_size]
            dct_block = cv2.dct(np.float32(block))
            dc_coefficients.append(dct_block[0, 0])
            # acs = dct_block.flatten()[1:]
            # ac_coefficients.append(acs)
    return dc_coefficients


def extract_color_features(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dominant_colors = find_dominant_colors(frame, k=5)
    variance = calculate_variance(frame)
    dom = dominant_colors.flatten()
    return dom, variance


def extract_audio_features(video_path, start_time_sec, end_time_sec=None):
    """
    Extracts the following audio features from a WAV file between the given start and end times in seconds:
    - Max amplitude
    - Max frequency
    - Min amplitude
    - Min frequency
    - MFCC mean
    - MFCC standard deviation
    - Audio activity threshold

    Parameters:
    video_path (str): Path to the video file (mp4 format).
    start_time_sec (float): The start time in seconds.
    end_time_sec (float): The end time in seconds. If not provided, extracts features until the end of the WAV file.

    Returns:
    numpy.ndarray: A 1D numpy array containing the requested audio features.
    """
    # Get audio file path
    audio_file_path = f'{OUTPUT_DIR}/{video_path.split("/")[-1].replace(".mp4", ".wav")}'

    # Load the audio file with a sampling rate of 44.1 kHz
    audio_time_series, sr = librosa.load(audio_file_path, sr=44100, mono=True)

    # Get the video frame rate (fps)
    fps = 30

    # Calculate the frame duration and hop length based on the frame rate
    frame_duration = 1 / fps
    hop_length = int(frame_duration * sr / 2)

    # Convert start time to sample index
    start_sample_idx = int(start_time_sec * sr)

    # If end time is not provided, use the end of the WAV file
    if end_time_sec is None:
        end_sample_idx = len(audio_time_series)
    else:
        end_sample_idx = int(end_time_sec * sr)

    # Extract the audio segment
    audio_segment = audio_time_series[start_sample_idx:end_sample_idx]

    # Get the MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio_segment, sr=sr, hop_length=hop_length)

    # Compute MFCC statistics
    mfcc_mean = np.mean(mfcc_features, axis=1)
    mfcc_std = np.std(mfcc_features, axis=1)
    mfcc_threshold = mfcc_mean - mfcc_std

    # Compute audio activity and energy
    audio_activity = np.where(mfcc_features > mfcc_threshold[:, None], 1, 0)
    audio_energy = np.sum(audio_activity, axis=1)

    # Extract the requested features
    max_amplitude = np.max(audio_segment)
    max_frequency = np.max(librosa.fft_frequencies(sr=sr, n_fft=len(audio_segment)))
    min_amplitude = np.min(audio_segment)
    min_frequency = np.min(librosa.fft_frequencies(sr=sr, n_fft=len(audio_segment)))

    return np.array([max_amplitude, max_frequency, min_amplitude, min_frequency, np.mean(mfcc_mean), np.mean(mfcc_std),
                     np.mean(mfcc_threshold)])


def compute_features(video_file, block_size=8, normalize=False):
    video_name = os.path.basename(video_file)
    frames = extract_frames(video_file)
    vectors = []
    max_frequency = 0
    min_frequency = 10 ** 4
    max_dominant_colors = 0
    min_dominant_colors = 10 ** 4
    max_variance = 0
    min_variance = 10 ** 4
    frequencies = []
    dominant_colors = []
    variances = []
    num_frames = len(frames) - 1
    for i in range(0, num_frames, 30):
        frame = frames[i]
        image = frame['image']
        if image is None:
            print(f'WARNING: Frame {i} has no image belonging to {video_name}')
        freq_vector = extract_freq_vectors(image, block_size=block_size)
        max_frequency = max(max_frequency, max(freq_vector))
        min_frequency = min(min_frequency, min(freq_vector))
        dom, variance = extract_color_features(image)
        max_dominant_colors = max(max_dominant_colors, max(dom))
        min_dominant_colors = min(min_dominant_colors, min(dom))
        max_variance = max(max_variance, max(variance))
        min_variance = max(min_variance, min(variance))
        frequencies.append(freq_vector)
        dominant_colors.append(dom)
        variances.append(variance)
    frequencies = normalize_np_array(frequencies, min_frequency, max_frequency)
    variances = normalize_np_array(variances, min_variance, max_variance)
    dominant_colors = normalize_np_array(dominant_colors, min_dominant_colors, max_dominant_colors)
    count = 0
    for i in range(0, num_frames, 30):
        frame = frames[i]
        start_timestamp = frame['start_timestamp']
        frame_id = frame['id']
        freq_vector = frequencies[count]
        variance = variances[count]
        dom = dominant_colors[count]
        embed = list(np.concatenate((freq_vector, variance, dom), axis=0))
        vectors.append([video_name, start_timestamp, frame_id, embed])
        count += 1
    pandas_df = pd.DataFrame(vectors, columns=['video_name', 'time_stamp', 'frame_num', 'embedding'])
    return pandas_df


def get_features_per_frame(video_name, frame_id, frame, block_size=8):
    image = frame['image']
    if image is None:
        print(f'WARNING: Frame {frame_id} has no image belonging to {video_name}')
        return []
    freq_vector = extract_freq_vectors(image, block_size=block_size)
    dom, variance = extract_color_features(image)
    embed = list(np.concatenate((freq_vector, variance, dom), axis=0))
    return [video_name, frame['start_timestamp'], frame['id'], embed]


def compute_features_optimized(video_file, block_size=8):
    time_start = time.time()
    video_name = os.path.basename(video_file)
    frames = extract_frames(video_file)
    time_end = time.time()
    print(f'Extracting frames took {time_end - time_start} seconds')
    time_start = time.time()
    vectors = process_video_frames(video_name, frames, block_size, 10)
    pandas_df = pd.DataFrame(vectors, columns=['video_name', 'time_stamp', 'frame_num', 'embedding'])
    time_end = time.time()
    print(f'Computing features took {time_end - time_start} seconds')
    return pandas_df


def process_frame(args):
    video_name, index, frame, block_size = args
    return get_features_per_frame(video_name, index, frame, block_size)


def process_video_frames(video_name, frames, block_size, num_threads):
    num_frames = len(frames) - 1
    args = [(video_name, i, frames[i], block_size) for i in range(0, num_frames, 30)]
    vectors = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        vectors = list(executor.map(process_frame, args))
    vectors = list(filter(lambda x: x != [], vectors))
    return vectors

