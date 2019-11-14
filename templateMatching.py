

import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import imutils

template_1_path = './Template/vysttech-symbols/1.png'
template_a_path = './Template/vysttech-symbols/A.png'
template_b_path = './Template/vysttech-symbols/B.png'
imgPath = './drawings/drawing1.png'

# Template Matching for symbols
def templateMatching(img, template):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, h

img  = cv2.imread(imgPath)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_1 = cv2.imread(template_1_path, 0)
found = None
top_left_1, bottom_right_1, h = templateMatching(img, template_1)

threshold = 0.9
x = top_left_1[0]
y = top_left_1[1]
i = 1
while (x + i) < 2500:
    print(img_gray[y, x+i])
    if img_gray[y, x+i] == 255: 
        break 
    i+=1
cv2.rectangle(img, (x,y), (x+i, y+h), 255, 2)

roi = img[y: y + h, x:x + i]
# Template Matching for surface A
template_a = cv2.imread(template_a_path, 0)

top_left_a, bottom_right_a, h_a = templateMatching(roi, template_a)
cv2.rectangle(roi, top_left_a, bottom_right_a, (0, 255, 0), 2)

# Template Matching for surface B
template_b = cv2.imread(template_b_path, 0)

top_left_b, bottom_right_b, h_b = templateMatching(roi, template_b)
cv2.rectangle(roi, top_left_b, bottom_right_b, (0, 0, 255), 2)


cv2.imshow("res", img)
cv2.waitKey(0)
