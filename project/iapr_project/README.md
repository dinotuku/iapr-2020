Image Analysis and Pattern Recognition - Special Project
===

Group 5: Chun-Hung Yeh, Kuan Tung, Zhuoyue Wang

## Setup

- `Python version == 3.6.10`

### Install required packages

```bash
pip install -r requirements.txt
```


## Usage

```bash
python main.py --input [INPUT_VIDEO_PATH] --output [OUTPUT_VIDEO_PATH]
```


## Python files descriptions

- `classifications.py`: Classification functions.
- `detections.py`: Object detection (red arrow, digits and operators) functions.
- `main.py`: Find and solve the math problem in a video, the main script.
- `train.py`: Training the digits/operators classifier.
- `utilities.py`: Helper functions.
