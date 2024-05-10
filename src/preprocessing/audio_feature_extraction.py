import os
import time

import librosa
import numpy as np
import pandas as pd
from scipy.fft import fft

from src.constants import OUTPUT_DIR
from src.preprocessing.utils import process_audio_from_video
from src.utils.normalization import normalize_np_array
from src.utils.np_util import append_zeros

audio_time_series_dictionary = {}


def load_audio(video_path):
    audio_file_path = video_path.replace(".mp4", ".wav")
    if video_path not in audio_time_series_dictionary.keys():
        audio_time_series, sr = librosa.load(audio_file_path, sr=44100, mono=True)
        audio_time_series_dictionary[video_path] = audio_time_series, sr
    else:
        audio_time_series, sr = audio_time_series_dictionary[video_path]
    return audio_time_series, sr


def extract_audio_features_mfcc(video_path):
    # Load the audio file with a sampling rate of 44.1 kHz
    audio_time_series, sr = load_audio(video_path)

    # Get the video frame rate (fps)
    fps = 30

    # Calculate the frame duration and hop length based on the frame rate
    frame_duration = 1 / fps
    hop_length = int(frame_duration * sr)
    np_array = []
    maxi = 0
    mini = 10 ** 9
    for start_sample_idx in range(0, len(audio_time_series), hop_length):
        audio_segment = audio_time_series[start_sample_idx:start_sample_idx + hop_length]
        mfcc_features = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=100)
        temp = []
        for mfcc in mfcc_features:
            mfcc_mean = np.mean(mfcc, axis=0)
            temp.append(mfcc_mean)
        maxi = max(maxi, np.max(temp))
        mini = min(mini, np.min(temp))
        np_array.append(temp)
    return np_array


def extract_fft(video_path):
    # Load the audio file with a sampling rate of 44.1 kHz
    audio_time_series, sr = load_audio(video_path)

    # Get the video frame rate (fps)
    fps = 30

    # Calculate the frame duration and hop length based on the frame rate
    frame_duration = 1 / fps
    hop_length = int(frame_duration * sr)
    np_array = []
    maxi = 0
    mini = 10 ** 9
    for start_sample_idx in range(0, len(audio_time_series), hop_length):
        # Extract the audio segment
        audio_segment = audio_time_series[start_sample_idx: start_sample_idx + hop_length]
        # fft_freq = fft(audio_segment, n=128) this is if we need less numb of data points
        # currently 1470 frequencies in 1/30th of a second
        fft_freq = fft(audio_segment)
        magnitude = np.abs(fft_freq)
        maxi = max(maxi, np.max(magnitude))
        mini = min(mini, np.min(magnitude))
        magnitude = append_zeros(np.array(magnitude), 1470)
        np_array.append(magnitude)

    # np_array = normalize_np_array(np_array, mini, maxi)

    return np_array


def compute_features(video_file):
    start = time.time()
    video_name = os.path.basename(video_file)
    vectors = []
    audio_file_path = video_file.replace(".mp4", ".wav")
    if not os.path.exists(audio_file_path):
        process_audio_from_video(video_file, audio_file_path)
    mfcc_coefficients = extract_audio_features_mfcc(video_file)
    fft_coefficients = extract_fft(video_file)
    for i in range(len(mfcc_coefficients)):
        start_timestamp = i * 1 / 30
        frame_id = i
        embed = list(np.concatenate((mfcc_coefficients[i], fft_coefficients[i]), axis=0))
        vectors.append([video_name, start_timestamp, frame_id, embed])
    pandas_df = pd.DataFrame(vectors, columns=['video_name', 'time_stamp', 'frame_num', 'embedding'])
    end = time.time()
    print(f"Time taken to compute features for {video_name}: {end - start} seconds")
    return pandas_df