"""This module contains the function create_output_video, which processes a video file to keep segments with detected objects and remove empty segments, using FFmpeg with minimal re-encoding. It handles videos with or without audio."""

import subprocess
from moviepy.editor import VideoFileClip


def create_output_video(keep_segments, input_video, output_video):
    """
    Process a video file to keep segments with detected objects and remove empty segments,
    using FFmpeg with minimal re-encoding. Handles videos with or without audio.

    Args:
    keep_segments (list of tuples): List of (start_time, end_time) pairs for segments to keep
    input_video (str): Path to the input video file
    output_video (str): Path for the output video file
    """
    try:
        with VideoFileClip(input_video) as input_clip:
            has_audio = input_clip.audio is not None

        if not keep_segments:
            print(
                "No relevant objects detected in the video. The output will be the same as the input."
            )
            ffmpeg_cmd = ["ffmpeg", "-i", input_video, "-c", "copy", output_video]
        else:
            ffmpeg_cmd = ["ffmpeg"]
            filter_complex = []

            for i, (start, end) in enumerate(keep_segments):
                ffmpeg_cmd.extend(["-i", input_video])
                filter_complex.append(
                    f"[{i}:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
                )
                if has_audio:
                    filter_complex.append(
                        f"[{i}:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
                    )

            if has_audio:
                filter_complex.append(
                    "".join(f"[v{i}][a{i}]" for i in range(len(keep_segments)))
                    + f"concat=n={len(keep_segments)}:v=1:a=1[outv][outa]"
                )
            else:
                filter_complex.append(
                    "".join(f"[v{i}]" for i in range(len(keep_segments)))
                    + f"concat=n={len(keep_segments)}:v=1[outv]"
                )

            ffmpeg_cmd.extend(["-filter_complex", "".join(filter_complex)])
            ffmpeg_cmd.extend(["-map", "[outv]"])
            if has_audio:
                ffmpeg_cmd.extend(["-map", "[outa]"])

            ffmpeg_cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-r", "30"])
            if has_audio:
                ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "128k"])

            ffmpeg_cmd.append(output_video)

        subprocess.run(ffmpeg_cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
        raise
    except Exception as e:
        print(f"Error creating output video: {str(e)}")
        raise
