import time

import numpy as np
from src.db import video_client as vc
from src.db import audio_client as ac
from src.preprocessing import feature_extraction as fe
from src.preprocessing import audio_feature_extraction as afe
from src.constants import OUTPUT_DIR
import pandas as pd
import json
import os
from scipy import stats
from collections import defaultdict


def load_video_vectors(csv_file):
    df = pd.read_csv(csv_file)
    df["embedding"] = df["embedding"].astype(str)
    res = []
    for i in df["embedding"]:
        sd = json.loads(i)
        res.append(sd)
    temp_df = pd.DataFrame()
    temp_df["embedding"] = res
    df["embedding"] = temp_df["embedding"]
    size = len(df["embedding"][1])
    vc.createTable(size)
    vc.insertEmbedding(df)


def load_audio_vectors(csv_file):
    df = pd.read_csv(csv_file)
    df["embedding"] = df["embedding"].astype(str)
    res = []
    for i in df["embedding"]:
        sd = json.loads(i)
        res.append(sd)
    temp_df = pd.DataFrame()
    temp_df["embedding"] = res
    df["embedding"] = temp_df["embedding"]
    size = len(df["embedding"][1])
    ac.createTable(size)
    ac.insertEmbedding(df)


def search_video(video_file):
    start = time.time()
    features = fe.compute_features_optimized(video_file, block_size=4)
    end = time.time()
    print("**** Extracting features for {} took {} seconds".format(video_file, end - start))
    embeddings = features["embedding"]
    vector_detected_video = vc.get_video_name(embeddings)
    print("**** Detected video for {} : {}".format(video_file, vector_detected_video))
    return vector_detected_video


def search_audio(video_file, video_name):
    features = afe.compute_features(video_file)
    embeddings = features["embedding"]
    result = []
    res_mode = []
    frequency_distribution = defaultdict(int)
    for i, embedding in enumerate(embeddings):
        vector_detected_video = ac.get_top3_similar_docs(embedding, video_name)
        if len(vector_detected_video) == 0:
            continue
        frame_detected = vector_detected_video[0][2] - i
        res_mode.append(frame_detected)
        result.append(vector_detected_video)
        frequency_distribution[frame_detected] += 1

    lis = sorted(frequency_distribution, key=frequency_distribution.get, reverse=True)
    mode, count = stats.mode(np.array(res_mode))
    print(
        f"confidence score is : {frequency_distribution[lis[0]] / len(res_mode)},"
        f" {frequency_distribution[lis[1]] / len(res_mode)},"
        f" {frequency_distribution[lis[2]] / len(res_mode)},"
        f" {frequency_distribution[lis[3]] / len(res_mode)},"
        f" {frequency_distribution[lis[4]] / len(res_mode)}")
    return mode


def extract_video_features(video_file, store=False):
    features = fe.compute_features_optimized(video_file)
    vc.createTable(len(features["embedding"][0]))
    vc.insertEmbedding(features)
    output_file = os.path.join(OUTPUT_DIR,
                               "feature_vectors_video_{}.csv".format(os.path.basename(video_file).split('.')[0]))
    if store:
        features.to_csv(output_file, index=False)


def extract_audio_features(video_file, store=False):
    features = afe.compute_features(video_file)
    ac.createTable()
    ac.insertEmbedding(features)
    output_file = os.path.join(OUTPUT_DIR,
                               "feature_vectors_audio_{}.csv".format(os.path.basename(video_file).split('.')[0]))
    if store:
        features.to_csv(output_file, index=False)
