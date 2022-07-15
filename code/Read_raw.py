# this file can also be used for reading the rectangular video
BoundingW = [80,85,80,75,80]
BoundingH = 155
BoundingY = 10
Bonding_location = [70,155,255,355,440]
root = "E:/database/soft phantom recording force videos/"
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:/software_packages/Tesseract/tesseract.exe"

# root = "E:/database/NURD/20th October/"
operatedir = root + "raw/disturb_p6/"
operatedir = root + "raw/phantom_pull_0.1_dis3/"
operatedir = root + "raw/tube1/"
operatedir = root + "raw/tube1slow/"
operatedir = root + "raw/tube2 slow long/"
operatedir = root + "raw/tube3vain/"
operatedir = root + "raw/sqr2/"
operatedir = root + "raw/20220712_182646.mp4"  # 100 seconds
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
import math
import numpy as np

import pandas as pd
import os
import torch
import scipy.signal as signal
from scipy.stats.stats import pearsonr
import random
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
# from Correct_sequence_integral import read_start



base_dir = os.path.basename(os.path.normpath(operatedir))
save_dir = root + "original/" + base_dir + "/"
save_crop_dir = root + "original_cropped/" + base_dir + "/"

# save_dir_cir = root + "resize_circular/" + base_dir + "/"
try:
    os.stat(save_dir)
except:
    os.makedirs(save_dir)
#
try:
    os.stat(save_crop_dir)
except:
    os.makedirs(save_crop_dir)
# read_start1 = 79
# read_start2 = 79
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}
Padding_H = 0

# Padding_H  = 254
# from  path_finding import PATH
Display_STD_flag = False
Padd_zero_top = True
Display_signal_flag = False
Display_Matrix_flag = False
save_matlab_flag = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
video_sizeH = 450
video_sizeW = 900

# videoout = cv2.VideoWriter(save_display_dir+'0output.avi', -1, 20.0, (video_sizeW,video_sizeH))

# show the image results
cap = cv2.VideoCapture(operatedir)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
# fig = figure()
# Read until video is completed
Len_steam = 5
ret, frame = cap.read()
if ret == True:
    H, W, _ = frame.shape
save_sequence_num = 0
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = frame[:,:,1]
        image_norm = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        frame_norm = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(save_dir + str(save_sequence_num) + ".jpg", image_norm)

        crop =  image_norm[900:1041,300:600]
        # crop = cv2.resize(crop, [600,280], interpolation=cv2.INTER_AREA)
        H, W  = crop.shape
        image = frame_norm[900:1041, 300:600]
        # image = cv2.resize(image, [600,280], interpolation=cv2.INTER_AREA)
        # pre-process the image by resizing it, converting it to
        # graycale, blurring it, and computing an edge map

        gray = crop
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 200, 255)
        cv2.imshow('edged',edged)
        cv2.waitKey(10)
        # find contours in the edge map, then sort them by their
        # size in descending order
        if save_sequence_num == 321:
            a =1
            pass
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None
        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            # if the contour has four vertices, then we have found
            # the thermostat display
            if len(approx) == 4:
                displayCnt = approx
                break
        if save_sequence_num == 322:
            pass
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
        warped = cv2.resize(warped, [540, 240], interpolation=cv2.INTER_AREA)
        output = four_point_transform(image, displayCnt.reshape(4, 2))
        output = cv2.resize(output, [540, 240], interpolation=cv2.INTER_AREA)

        cv2.imshow('output', output)
        cv2.waitKey(10)
        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        thresh = cv2.threshold(warped, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        invert = 255 - thresh
        data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
        print(data)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(10)

        # find contours in the thresholded image, then initialize the
        # digit contours lists
        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        #                         cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # digitCnts = []
        # # loop over the digit area candidates
        # for c in cnts:
        #     # compute the bounding box of the contour
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     # if the contour is sufficiently large, it must be a digit
        #     if w >= 15 and (h >= 40 and h <= 80):
        #         digitCnts.append(c)
        # # sort the contours from left-to-right, then initialize the
        # # actual digits themselves
        # digitCnts = contours.sort_contours(digitCnts,
        #                                    method="left-to-right")[0]
        digits = []
        # loop over each of the digits
        # TODO: we know there are 5 digits with fixed location
        # for c in digitCnts :
        for i in np.arange(5):

            # extract the digit ROI
            w= BoundingW[i]
            h= BoundingH
            y=BoundingY
            x = Bonding_location[i]
            # (x, y, w, h) = cv2.boundingRect(c)
            roi = thresh[y:y + h, x:x + w]
            # compute the width and height of each of the 7 segments
            # we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.35), int(roiH * 0.15))
            dHC = int(roiH * 0.075)
            # define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),  # top
                ((0, 0), (dW, h // 2)),  # top-left
                ((w - dW, 0), (w, h // 2)),  # top-right
                ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
                ((0, h // 2), (dW, h)),  # bottom-left
                ((w - dW, h // 2), (w, h)),  # bottom-right
                ((0, h - dH), (w, h))  # bottom
            ]
            on = [0] * len(segments)
        # loop over the segments
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                # if the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area+0.001) > 0.3:
                    on[i] = 1
            # lookup the digit and draw it on the image
            try:

                digit = DIGITS_LOOKUP[tuple(on)]

            except KeyError:

                print("Oops!  That was no valid number.  Try again...")
                digit = 0
            digits.append(digit)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(output, str(digit), (x   , y +h+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        # display the digits
        # print(u"{}{}.{} \u00b0C".format(*digits))
        cv2.imshow("Input", image)
        cv2.imshow("Output", output)
        cv2.waitKey(1)
        cv2.imwrite(save_crop_dir + str(save_sequence_num) + ".jpg", warped)

        save_sequence_num += 1
        print("[%s]   is processed." % (save_sequence_num))


    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
