import time
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    crop = False
    count = False

    dir_origin_path = "/data/ToothData/testimages"
    dir_save_path = "/data/ToothData/testpredictimages"

    for filename in os.listdir(dir_origin_path):
        img = os.path.join(dir_origin_path, filename)

        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image, crop=crop, count=count)
            plt.imshow(r_image)
            plt.show()
