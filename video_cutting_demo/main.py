from object_detector import ObjectDetector
from video_processor import create_output_video
from frame_analyzer import (
    analyze_video_frames,
    generate_keep_segments,
    save_keep_segments,
)


def process_video(
    input_video,
    output_video,
    statistics_file,
    detector,
    sampling_rate=1,
    empty_frame_threshold=5,
):
    # Analyze video frames
    frame_data = analyze_video_frames(input_video, detector, sampling_rate)

    # Generate keep segments
    keep_segments = generate_keep_segments(frame_data, empty_frame_threshold)

    # Save keep segments to a file
    save_keep_segments(keep_segments, statistics_file)

    # Create output video
    create_output_video(keep_segments, input_video, output_video)


if __name__ == "__main__":
    INPUT_VIDEO = "input_videos/data_station_footage_Zandvoort_livestream_2024-08-04_11-37-06.mp4"
    OUTPUT_VIDEO = "output_videos/demo_short.mp4"
    OUTPUT_FRAMES_DIR = "output_frames"
    STATISTICS_FILE = "statistics/keep_segments.csv"
    FRAME_INTERVAL = 60 * 10  # Process every nth frame
    EMPTY_FRAME_THRESHOLD = 2  # Number of consecutive empty frames before cutting
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection

    detector = ObjectDetector(
        output_frames_dir=OUTPUT_FRAMES_DIR, confidence_threshold=CONFIDENCE_THRESHOLD, verbose=True
    )

    process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        STATISTICS_FILE,
        detector,
        sampling_rate=FRAME_INTERVAL,
        empty_frame_threshold=EMPTY_FRAME_THRESHOLD,
    )
