YOLO Object Detection & AI Proctoring
A Python-based computer vision project for object detection and
exam-proctoring experiments using YOLO (Ultralytics). The project
includes modules for audio voice-activity detection (VAD),
suspicious-activity evaluation, and logging.
Note: This repository contains the application source code.
Trained YOLO weights, datasets, generated detections, and the local
Python virtual environment are excluded from Git.

Features
- YOLO object detection: Integrate trained YOLO models into the
  Python application.
- Audio VAD module: audio/vad_detector.py contains the
  voice-activity detection component.
- Suspicion engine: core/suspicion_engine.py provides a
  dedicated module for proctoring-related decision logic.
- Logging: core/logger.py supports application logging.
- Modular codebase: Separate audio and core functionality for
  easier maintenance.
Actual detection classes, alert thresholds, and screenshot behavior
depend on your local model files and current implementation.
Repository structure
Object Detection/
├── .gitignore
├── README.md
├── main.py
├── audio/
│   ├── __init__.py
│   └── vad_detector.py
└── core/
    ├── __init__.py
    ├── logger.py
    └── suspicion_engine.py
Additional local directories may include dataset/, dataset2/,
Dataset3/, Dataset4/, runs/, Cheating_Detected/, logs/,
violations/, and yolov8_env/. These are intentionally not part of
the repository.
Requirements
- Python (use the version compatible with your installed Ultralytics
  and PyTorch packages)
- A webcam for live video detection, if enabled by the application
- Trained model weight files (.pt) expected by the code
- An NVIDIA GPU is optional; CPU execution may be slower
Installation
1. Clone the repository
git clone https://github.com/Prateekshrivastava26/Object_Detection.git
cd Object_Detection
2. Create and activate a virtual environment (Windows PowerShell)
py -m venv yolov8_env
.\yolov8_env\Scripts\Activate.ps1
3. Install dependencies
For basic YOLO and OpenCV functionality:
python -m pip install --upgrade pip
pip install ultralytics opencv-python
Other dependencies may be needed for the VAD and proctoring modules.
Install them according to the imports in the source files. Once you have
a working environment, you can create a reproducible dependency list:
python -m pip freeze > requirements.txt
Commit requirements.txt to the repository after reviewing it.
Model weights and datasets
Trained model weights such as best.pt are not included in this GitHub
repository. Obtain your trained weights separately and update the paths
in the application to match their actual location.
For example, a locally trained model may be stored at:
runs/detect/train/weights/best.pt
Datasets are also excluded from version control. Do not publish private
candidate recordings or exam materials.
Run the project
Start the Python application from the project root:
python main.py
The application may require model files, a working camera, and any paths
or settings referenced by main.py.
Test a trained YOLO model independently
If Ultralytics is installed and a model weight file is available:
yolo detect predict model="runs/detect/train/weights/best.pt" source=0 show=True
For an NVIDIA GPU supported by your PyTorch installation:
yolo detect predict model="runs/detect/train/weights/best.pt" source=0 device=0 show=True
source=0 uses the default webcam. These commands are for standalone
Ultralytics detection, not necessarily the complete proctoring pipeline.
Development workflow
git status
git add .
git commit -m "Describe your changes"
git push
The .gitignore excludes local environments, trained weights, datasets,
logs, and generated output. Verify ignored files before changing the
rules.
Privacy and responsible use
Use this software only in authorized testing or exam environments.
Obtain appropriate consent, protect recordings and screenshots, and
review automated detections rather than treating them as definitive
proof of misconduct.
Repository
GitHub ---
Prateekshrivastava26/Object_Detection
Project status
In development. Update this README with exact supported detection
classes, model download instructions, configuration options, and tested
dependencies as the implementation evolves.