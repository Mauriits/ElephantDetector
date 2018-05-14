import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
from pprint import pprint as p
from time import sleep
import os
import cv2
sys.path.append("D:/Maurits/Documenten/Universiteit/Master/Computer Vision/Assignment 4/coco-master/PythonAPI")
p(sys.path)
from pycocotools.coco import COCO

dataDir="."
dataType="val2014"
img_src="D:/Maurits/Documenten/Universiteit/Master/Computer Vision/Assignment 4/"
img_dst="D:/Maurits/Documenten/Universiteit/Master/Computer Vision/Assignment 4/elephants_val/"

annFile="%s/annotations/instances_%s.json"%(dataDir,dataType)

coco = COCO(annFile)

category = "elephant"

image_count = 0 
catId = coco.getCatIds(category)
imgIds = coco.getImgIds(catIds=catId)

counter = 0
for imgId in imgIds:
    annId = coco.getAnnIds(imgIds=imgId, catIds=catId)
    anns = coco.loadAnns(annId)

    bbox = anns[0]['bbox']
    imgname = str(imgId)
    while (len(imgname) < 12):
        imgname = "0" + imgname
        
    imgname = "COCO_" + dataType + "_" + imgname + ".jpg"
    
    image = cv2.imread(img_src + dataType + "/" + imgname)
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    
    x2 = x1 + int(bbox[2])
    y2 = y1 + int(bbox[3])
    
    cropped_img = image[y1:y2, x1:x2]
    
    cv2.imwrite(img_dst + imgname, cropped_img)
    print(counter)
    counter += 1