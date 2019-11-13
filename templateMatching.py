

import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import imutils

templatePath = './GD-T/vysttech-symbols/1.jpg'
imgPath = './drawings/drawing1.png'

img  = cv2.imread(imgPath, 0)
template = cv2.imread(templatePath, 0)
w, h = template.shape[::-1]

found = None 

for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # keep track of the ratio of the resizing 
    resized = imutils.resize(img, width = int(img.shape[1] * scale))
    r = img.shape[1] / float(resized.shape[1])
    
    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    found = (maxVal, maxLoc, r)

(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + w)* r), int((maxLoc[1] + h) * r))

cv2.rectangle(img,(startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("res", img)
#cv2.waitKey(0)

