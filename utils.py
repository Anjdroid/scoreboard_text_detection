import json
import cv2 as cv
import numpy as np


def normalize(data):
    """normalize color range to [0,1]."""        
    image_copy = np.copy(data)
    image_copy = image_copy / 255.0
    return image_copy


