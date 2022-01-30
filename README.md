# Scoreboard & text detection
Using U-Net architecture to segment scoreboard from tennis field and EAST or Pytessaract to extract Player names and who is serving indicator on video data of Tennis matches.

## Prerequisites

- python version >= 3.6

## Install required packages
```
$ pip install -r requirements.txt
```

## Run
- For testing U-Net segmenation run:
```
python train_test.py
```
- For extracting text / player names with EAST and Pytessaract:
```
python detect_text.py
```
