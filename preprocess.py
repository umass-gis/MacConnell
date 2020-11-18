import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy
from sklearn.cluster import KMeans
from skimage import data
from skimage.filters import try_all_threshold
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import imutils
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import sys
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer
import pytesseract

imgs_not = []
def get_roi(im, pth):
    try:
        img = Image.open(os.path.join("extract_images/all_imgs/", im))
        width, height = img.size
        # print(width, height)
        img = img.crop((width/4, height/4, 3/4*width, 3/4*height))
        width, height = img.size

        img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

        morph = img.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # take morphological gradient
        gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)


        # split the gradient image into channels
        image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
        channel_height, channel_width, _ = image_channels[0].shape

        # apply Otsu threshold to each channel
        for i in range(0, 3):
            _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

        # merge the channels
        image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
        image_channels = cv2.cvtColor(image_channels,cv2.COLOR_BGR2GRAY)

        # Apply threshold to the Image now
        ret,thresh = cv2.threshold(image_channels,200,255,0)

        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # plt.imshow(thresh, cmap='gray')
        # plt.show()

        custom_config = r'--psm 6'
        details = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng')

        # print(details)

        x_min = float('inf')
        y_min = float('inf')
        y_max = float('-inf')
        x_max = float('-inf')

        total_boxes = len(details['text'])

        for sequence_number in range(total_boxes):
            if int(details['conf'][sequence_number]) >30:
                (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                                details['width'][sequence_number],  details['height'][sequence_number])
                x_min = min(x_min, x, x+w)
                y_min = min(y_min, y, y+h)
                x_max = max(x_max, x, x+w)
                y_max = max(y_max, y, y+h)

        center = np.array(([x_min+((x_max-x_min)/2), y_min + ((y_max-y_min)/2)]), dtype = int)
        radius = int(np.sqrt( (center[0] - x_min)**2 + (center[1] - y_min)**2 ))



        # print(tuple(center), radius)
        res = cv2.circle(img,tuple(center),int(radius) ,(255,255,255),2)

        # plt.imshow(res, cmap = 'gray')
        # plt.show()

        mask = np.ones((height,width), np.uint8)
        circle_img = cv2.circle(mask,tuple(center),int(radius), (255,255,255),thickness=-1)


        # Copy that image using that mask
        masked_data = cv2.bitwise_and(img, img, mask=circle_img)

        # Apply Threshold
        _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

        # Find Contour
        contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        x,y,w,h = cv2.boundingRect(contours[0])

        # Crop masked_data
        crop = masked_data[y:y+h,x:x+w]

        ## Save the cropped mask
        cv2.imwrite(os.path.join(pth , im), crop)
    except:
        imgs_not.append(im)
    # plt.imshow(crop, cmap = 'gray')
    # plt.show()

def get_images(arr, save_path):
    for i in arr:
        print(i)
        get_roi(i, save_path)
        # return


arr = os.listdir("extract_images/all_imgs/")
path_save = "extract_images/extract_nodules/"

get_images(arr, path_save)
print(imgs_not)
