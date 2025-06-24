# Player Detection and Tracking with YOLOv8 + Deep SORT

This project demonstrates **player detection and tracking** in sports videos using **YOLOv8** for object detection and **Deep SORT** for tracking across frames.


##  Features

- Detects **players (persons)** in video frames using YOLOv8
- Tracks players using Deep SORT
- Labels each detected player as `Player X` with bounding box
- Saves the output video with bounding boxes and player IDs

---

##  Tech Stack

- `Python 3.10+`
- `Ultralytics YOLOv8` for detection
- `deep_sort_realtime` for tracking
- `OpenCV` for video processing

---

##  Installation

1. **Clone the repository:**
cmd
git clone https://github.com/your-username/player-tracking-yolov8.git
cd player-tracking-yolov8

2)Create and activate virtual environment:

python -m venv yolov8env
yolov8env\Scripts\activate  # Windows

3)Install required libraries:
pip install ultralytics opencv-python deep_sort_realtime

4)Ensure your video file is placed at:
videos/15sec_input_720p.mp4

5)How to Run
python detect_and_track.py


Project Structure

Copy code
├── detect_and_track.py         # Main script
├── videos/
│   └── 15sec_input_720p.mp4    # Input video
├── output_tracked.mp4          # Output video with player tracking
├── requirements.txt            # Required libraries
└── README.md                   # Documentation


