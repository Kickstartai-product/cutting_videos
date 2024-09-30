"""This module contains functions to analyze video frames and generate keep segments."""

import cv2
from object_detector import ObjectDetector
from tqdm import tqdm


def analyze_video_frames(
    video_path: str, predictor: ObjectDetector, sampling_rate: int = 1
):
    """
    Analyze video frames using the provided predictor.

    Args:
    video_path (str): Path to the input video file
    predictor (Predictor): An instance of a Predictor subclass
    sampling_rate (int): Analyze every nth frame (default: 1)

    Returns:
    list of tuples: List of (timestamp, has_objects) pairs for sampled video frames
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_data = []
    for i in tqdm(range(0, frame_count, sampling_rate)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = i / fps
        detected_objects = predictor.predict(frame, i)
        # insert timestamp and frame number into detected objects, this is useful for debugging
        for d in detected_objects:
            d['timestamp'] = timestamp
            d['frame_number'] = i
        has_objects = len(detected_objects) > 0
        frame_data.append((timestamp, has_objects, detected_objects))

    cap.release()
    return frame_data


def generate_keep_segments(frame_data, empty_frame_threshold):
    """
    Generate keep segments based on frame data and empty frame threshold.

    Args:
    frame_data (list of tuples): List of (timestamp, has_objects) pairs for sampled video frames
    empty_frame_threshold (int): Number of consecutive empty frames to trigger a cut

    Returns:
    list of tuples: List of (start_time, end_time) pairs for segments to keep
    """
    keep_segments = []
    start_time = 0
    empty_count = 0
    last_object_time = 0

    for i, (timestamp, has_objects, _) in enumerate(frame_data):
        if has_objects:
            if empty_count >= empty_frame_threshold:
                if start_time < last_object_time:
                    keep_segments.append((start_time, last_object_time))
                start_time = timestamp
            last_object_time = timestamp
            empty_count = 0
        else:
            empty_count += 1

        if i == len(frame_data) - 1:
            if empty_count < empty_frame_threshold:
                keep_segments.append((start_time, last_object_time))
            elif start_time < last_object_time:
                keep_segments.append((start_time, last_object_time))

    return keep_segments


def save_keep_segments(keep_segments, output_file):
    """
    Save keep segments to a csv file.

    Args:
    keep_segments (list of tuples): List of (start_time, end_time) pairs for segments to keep
    output_file (str): Path to the output text file
    """
    with open(output_file, "w") as f:
        # write header
        f.write("start_time,end_time\n")
        for start, end in keep_segments:
            f.write(f"{start},{end}\n")

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps