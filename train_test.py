from cgi import test
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torch
from dataset import VideoDataset
import cv2 as cv
from unet import UNet
from utils import normalize

# constants
VIDEO_FILE = 'data/top-100-shots-rallies-2018-atp-season.mp4'
JSON_FILE = 'data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json'
DATA_FOLDER = 'data'
MODEL_PATH = 'model/sb-unet.pth'
NUM_EPOCHS = 1000

vdata = VideoDataset(VIDEO_FILE, JSON_FILE, test=True)
keys = list(vdata.videos.keys())
print(len(keys))


def test_sbUNET():
    print("=TESTING MODEL=")
    model = UNet()
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    for i in range(len(keys)):       
        # get item
        frame, mask, sb, annot = vdata[keys[i]]   
        frame = normalize(cv.cvtColor((frame * 255).astype(np.uint8), cv.COLOR_BGR2GRAY))
        vid = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
        output = model(vid).permute(0, 2, 3, 1).squeeze().detach().numpy()
        output_mask = output.argmax(2) * 255
        cv.imshow('frame', (frame * 255).astype(np.uint8))
        cv.imshow('mask', (mask*255).astype(np.uint8))
        cv.imshow('out0', (output[:,:,0] * 255).astype(np.uint8))
        cv.imshow('out1', (output[:,:,1] * 255).astype(np.uint8))
        cv.imshow('result mask', (output_mask).astype(np.uint8))
        cv.waitKey(0)

        # calc IOU
        mask = mask * 255
        overlap = np.clip(mask * output_mask, 0, 255) # logical AND
        union =  np.clip(mask + output_mask, 0, 255) # logical OR
        IOU = overlap.sum()/float(union.sum())
        print(f'IOU: {IOU}')


def train_sbUNET():
    loss_b = np.inf
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # setup UNET with 2 classes => FG=1, BG=0
    model = UNet(dimensions=2).to(device)
    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
    # use Adam optimizer & crossentropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    idx = 0
    for epoch in range(NUM_EPOCHS):
        # get item
        frame, mask, sb, annot = vdata[keys[idx]]   
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)        
        optimizer.zero_grad()
        # get model prediction
        output = model(torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor))
        # calc loss
        loss = criterion(output, torch.from_numpy(mask).unsqueeze(0).type(torch.cuda.FloatTensor).long())
        step_loss = loss.item()
        print(f'Epoch: {epoch} \tLoss: {step_loss}')
        if step_loss < loss_b:
            print('Saving model')
            loss_b = step_loss            
            torch.save(model.state_dict(), MODEL_PATH)
        loss.backward()
        optimizer.step()
        idx += 1

        # reset frame idx
        if idx >= len(keys):
            idx = 0
    return

if __name__ == "__main__":
    test_sbUNET()