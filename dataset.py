from torch.utils.data import Dataset
import json
import cv2 as cv
import numpy as np
from utils import normalize

TRAIN_LEN = 700
TEST_LEN = 1500
WIDTH = 512
HEIGHT = 256

class VideoDataset(Dataset):
    def __init__(self, video_file, annot_file, test=False):
        self.video_file = video_file
        self.annot_file = annot_file
        self.videos = None
        self.test = test
        self.read_data()

    def load_json(self, file):
        """ 
        loads json object 
        input param: JSON filename
        output param: JSON object data
        """
        f = open(file)
        data = json.load(f)
        f.close()
        return data

    def generate_scoreboard_mask(self, bbox, img_size):
        """
        generates scoreboard mask
        input: scoreboard bounding box
               image shape
        output: mask
        """
        mask = np.zeros((img_size[0], img_size[1]))
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        mask[y:h,x:w] = 255
        #cv.imshow('mask', mask.astype(np.uint8))
        #cv.waitKey(0)
        return mask

    
    def get_scoreboard(self, annotations, frame):
        """
        cuts scoreboard from frame
        input: scoreboard bounding box
               frame
        output: scoreboard img
        """
        bbox = annotations['bbox']
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        scoreboard = frame[y:h,x:w]
        #cv.imshow('scoreboard', scoreboard.astype(np.uint8))
        #cv.waitKey(0)
        return scoreboard


    def read_data(self):
        """
        read video data with corresponding frame annotations
        input params: STRING video file, JSON annotation file
        output params: dictionary of frame files (& annotations)
        """
        video_frame = 0
        break_idx = 0
        idx = 0
        self.videos = dict()
        
        # load annotations
        data = self.load_json(self.annot_file)
        keys = list(data.keys())
        # start video capture
        video = cv.VideoCapture(self.video_file)
        if not video.isOpened():
            print("ERROR: cannot open video")
        while video.isOpened():
            ret, frame = video.read()
            if ret == True:
                # save annotated frames for train
                if str(video_frame) == keys[idx]:
                    if not self.test and video_frame < TRAIN_LEN:
                        print("==TRAIN: FRAME IDX==", video_frame)
                        values = data[str(video_frame)]
                        # extract scoreboard from annotations
                        scoreboard = self.get_scoreboard(values, frame)
                        # define scoreboard mask
                        mask = self.generate_scoreboard_mask(values['bbox'], np.asarray(frame).shape)
                        key = keys[idx]
                        # save frame, scoreboard, mask & annotations to video dict
                        self.videos[key] = [frame, scoreboard, mask, values]
                    elif self.test and video_frame > TRAIN_LEN:
                        print("==TEST: FRAME IDX==", video_frame)
                        # SAVE TEST DATA
                        values = data[str(video_frame)]
                        # extract scoreboard from annotations
                        scoreboard = self.get_scoreboard(values, frame)
                        # define scoreboard mask
                        mask = self.generate_scoreboard_mask(values['bbox'], np.asarray(frame).shape)
                        key = keys[idx]
                        # save frame, scoreboard, mask & annotations to video dict
                        self.videos[key] = [frame, scoreboard, mask, values]
                    idx += 1
                video_frame += 1
            if video_frame > TEST_LEN:
                break
        video.release()
        cv.destroyAllWindows()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        print("===getitem==")
        frame, sb, mask, gt = self.videos[idx]
        h, w, _ = frame.shape
        frame = normalize(cv.resize(frame, (WIDTH, HEIGHT)))
        mask = normalize(cv.resize(mask, (WIDTH, HEIGHT)))
        sb = cv.resize(sb, (WIDTH, HEIGHT))
        return frame, mask, sb, gt

if __name__ == "__main__":
    
    VIDEO_FILE = 'data/top-100-shots-rallies-2018-atp-season.mp4'
    JSON_FILE = 'data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json'
    VideoDataset(VIDEO_FILE, JSON_FILE)