#coding=utf-8
import numpy as np
import cv2
import dlib
import pandas as pd
import scipy.io as sio
import os
import csv
import datetime

import importlib,sys
importlib.reload(sys)
sys.path.append("../core/")
import data_processing_core as dpc

root = "/data/wyl_data/ETH-Gaze/Image/test"
# sample_root = "/data/MPIIGaze/Evaluation Subset/My sample list for eye image1"
out_root = "/data/wyl_data/ETH-Gaze/Image/traincrop7test"
im_label="/home/user/wyl/DVGaze-main/Label/ETH/cam7.test"

scale = True

shape_detector_path="/home/user/wyl/GazeTR-main1/shape_predictor_68_face_landmarks.dat"
target_size=[768,1366]
left_eye_up_point1=37-1#Outer corner of left eye
left_eye_up_point2=40-1#Inner corner of left eye
right_eye_up_point1=46-1#Outer corner of right eye
right_eye_up_point2=43-1#Inner corner of right eye
lip_point1=49-1#Corner of mouth
lip_point2=55-1#Corner of mouth
# num = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

def datadetector(image):
    dets = detector(image, 1)  # detect face
    if len(dets)==0:
        return 0
    for k, d in enumerate(dets):
        shape = predictor(image, d)  # detect 68 feature points
        # 获取4个特征点
        out = {}
        out["left_left_corner"] = [shape.part(left_eye_up_point1).x,shape.part(left_eye_up_point1).y]
        out["left_right_corner"] = [shape.part(left_eye_up_point2).x,shape.part(left_eye_up_point2).y]
        out["right_left_corner"] = [shape.part(right_eye_up_point1).x,shape.part(right_eye_up_point1).y]
        out["right_right_corner"] = [shape.part(right_eye_up_point2).x,shape.part(right_eye_up_point2).y]

    return out

def read_imgdir():
    path_name = root

    with open(im_label,"r") as file:
        lines=file.readlines()
        lines=lines[1:]

    for line in lines:
        # Read souce information
        line = line.strip().split(" ")
        item=line[0]
        img=cv2.imread(os.path.join(path_name,item))
        i = item.split(".", 2)[0]
        file = i.split("/", 2)[0]
        name=i.split("/", 2)[1]
        if not os.path.exists(os.path.join(out_root, file)):
            os.makedirs(os.path.join(out_root, file))
        im_outpath=os.path.join(out_root, file)
        annotation = datadetector(img)
        print(f"Start Processing")
        if annotation!=0:
            ImageProcessing_Person(im_outpath,annotation, name, img)

def CropEye(im,lcorner, rcorner):
    imsize=(224,224)
    x, y = list(zip(lcorner, rcorner))

    center_x = np.mean(x)
    center_y = np.mean(y)

    width = np.abs(x[0] - x[1]) * 1.8
    times = width / 60
    height = 36 * times

    x1 = [max(center_x - width / 2, 0), max(center_y - height / 2, 0)]
    x2 = [min(x1[0] + width, imsize[0]), min(x1[1] + height, imsize[1])]
    im = im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
    im = cv2.resize(im, (60, 36))
    return im

def ImageProcessing_Person(im_outpath, annotation, i, img):
    print(f"start cropping")

    # Create the handle of label
    if not os.path.exists(os.path.join(im_outpath, "face")):
        os.makedirs(os.path.join(im_outpath, "face"))
    if not os.path.exists(os.path.join(im_outpath, "left")):
        os.makedirs(os.path.join(im_outpath, "left"))
    if not os.path.exists(os.path.join(im_outpath, "right")):
        os.makedirs(os.path.join(im_outpath, "right"))

    im_face = img

    # Crop left eye images
    im_left = CropEye(img,annotation["left_left_corner"],annotation["left_right_corner"])
    # im_left = img[annotation["left_left_corner"][1]:annotation["left_right_corner"][1], annotation["left_left_corner"][0]:annotation["left_right_corner"][0]]
    # im_left = dpc.EqualizeHist(im_left)

     # Crop Right eye images
    im_right = CropEye(img, annotation["right_left_corner"], annotation["right_right_corner"])
    # im_right = img[annotation["right_left_corner"][1]:annotation["right_right_corner"][1], annotation["right_left_corner"][0]:annotation["right_right_corner"][0]]
    # im_right = dpc.EqualizeHist(im_right)
    print(os.path.join(im_outpath, "face", str(i) + ".jpg"))
    # Save the acquired info
    cv2.imwrite(os.path.join(im_outpath, "face", str(i) + ".jpg"), im_face)
    cv2.imwrite(os.path.join(im_outpath, "left", str(i) + ".jpg"), im_left)
    cv2.imwrite(os.path.join(im_outpath, "right", str(i) + ".jpg"), im_right)


if __name__ == "__main__":
    read_imgdir() # image
