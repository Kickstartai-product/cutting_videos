This repository is used to cut parts of stations' camera recordings where there aren't
any objects of interest (e.g., humans or bags). This allows us to more efficiency look
for cases of unattended objects.

Make sure you have the following folder structure in this project:

.
├── input_videos
├── output_frames
├── output_videos
├── pyproject.toml
├── README.md
├── statistics
├── tests
└── video_cutting_demo

To run the code, make sure that there are input videos present in `input_videos`. 
The project can be installed using poetry. First run `poetry install`. Then, run main.py
using `poetry run video_cutting_demo/main.py`. 

Additionally, make sure that ffmpeg is installed to process the output video. Ffmpeg can be
installed on MAC OS systems with homebrew (`brew install ffmpeg`).