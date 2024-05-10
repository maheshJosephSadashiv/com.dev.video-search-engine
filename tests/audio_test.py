import os

import librosa
import pywt
from scipy.fft import fft
from src.constants import OUTPUT_DIR
from src.preprocessing.utils import process_audio_from_video
import numpy as np

maxi = 0
mini = 10**9

def testing(video_file):
    video_name = os.path.basename(video_file)
    audio_file_path = f'{video_name.replace(".mp4", ".wav")}'
    if not os.path.exists(audio_file_path):
        process_audio_from_video(video_file, audio_file_path)
    # return extract_audio_features_frame_wise(audio_file_path)
    return extract_audio_features_mfcc(audio_file_path)

def dpcm_encode(signal):
    # Initialize encoded signal with first sample
    encoded_signal = [signal[0]]

    # Iterate over the remaining samples
    for i in range(1, len(signal)):
        # Calculate the prediction error (difference between current sample and previous prediction)
        prediction_error = signal[i] - encoded_signal[-1]

        # Encode the prediction error
        encoded_signal.append(prediction_error)

    return np.array(encoded_signal)

def extract_audio_features_frame_wise(video_path):
    # Get audio file path
    audio_file_path = f'{video_path.split("/")[-1].replace(".mp4", ".wav")}'

    # Load the audio file with a sampling rate of 44.1 kHz
    audio_time_series, sr = librosa.load(audio_file_path, sr=44100, mono=True)

    # Get the video frame rate (fps)
    fps = 30

    # Calculate the frame duration and hop length based on the frame rate
    frame_duration = 1 / fps
    hop_length = int(frame_duration * sr)
    idx = 0
    num_frame = int(len(audio_time_series)/sr) * fps
    np_array = []

    for start_sample_idx in range(0, len(audio_time_series), hop_length):
        # Extract the audio segment
        audio_segment = audio_time_series[start_sample_idx: start_sample_idx + hop_length]

        fft_freq = fft(audio_segment, n=128)
        # print(fft_freq)
        magnitude = np.abs(fft_freq)
        maxi = max(maxi, np.max(magnitude))
        mini = min(mini, np.min(magnitude))
        np_array.append(np.max(magnitude))

    # for i, lis in enumerate(np_array):
    #     np_array[i] = (lis - mini) / (maxi - mini)

    return np_array

def extract_audio_features_mfcc(video_path):
    audio_file_path = f'{video_path.split("/")[-1].replace(".mp4", ".wav")}'

    # Load the audio file with a sampling rate of 44.1 kHz
    audio_time_series, sr = librosa.load(audio_file_path, sr=44100, mono=True)

    # Get the video frame rate (fps)
    fps = 30

    # Calculate the frame duration and hop length based on the frame rate
    frame_duration = 1 / fps
    hop_length = int(frame_duration * sr)
    np_array = []
    for start_sample_idx in range(0, len(audio_time_series), hop_length):
        audio_segment = audio_time_series[start_sample_idx:start_sample_idx + hop_length]
        mfcc_features = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=10)
        temp = []
        for mfcc in mfcc_features:
            temp.append(np.mean(mfcc, axis=0))
        np_array.append(temp)
    return np_array

def extract_dwt_features(video_path):
    audio_file_path = f'{video_path.split("/")[-1].replace(".mp4", ".wav")}'
    audio_time_series, sr = librosa.load(audio_file_path, sr=44100, mono=True)

    total_samples = len(audio_time_series)
    frames = []
    dwt_features = []
    fps =30
    hop_length = int(sr / fps)

    for start_sample_idx in range(0, total_samples, hop_length):
        frame = audio_time_series[start_sample_idx:start_sample_idx + hop_length]
        coeffs = pywt.wavedec(frame, "sym5", level=5)
        feature_vector = np.concatenate([c.flatten() for c in coeffs[1:-1]])
        dwt_features.append(feature_vector)

    return dwt_features


if __name__ == '__main__':
    # 12240
    np_store = testing("video7.mp4")
    np_queries = testing("video7_1_modified.mp4")
    offset = 870
    for i in range(len(np_queries)):
        np_query = np_queries[i]
        np_array = np_store[i + offset + 10]
        diff = np.mean(np.array(np_query) - np.array(np_array))
        print(diff)
    # np_store = testing("video6.mp4")
    # np_queries = testing("video_6_1_filtered.mp4")
    # offset = 12240
    # for i in range(len(np_queries)):
    #     np_query = np.array(np_queries[i])
    #     np_array = np.array(np_store[i + offset - 1])
    #     print(np_query[0])
    #     # diff = np_query - np_array
    #     # print(np.mean(np.abs(diff), axis=0))
    # np_store = extract_dwt_features("video6.mp4")
    # np_queries = extract_dwt_features("video_6_1_filtered.mp4")
    # offset = 12240
    # for i in range(len(np_queries)):
    #     np_query = np_queries[i]
    #     np_array = np_store[i + 300]
    #     diff = np_query - np_array
    #     print(np.mean(np.abs(diff), axis=0))


