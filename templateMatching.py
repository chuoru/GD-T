from PIL import Image
import pytesseract
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
import imutils
import os
import xlsxwriter

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('results.xlsx')
worksheet = workbook.add_worksheet()

template_a_path = './Template/vysttech-symbols/A.png'
template_b_path = './Template/vysttech-symbols/B.png'
template_phi_path = './Template/vysttech-symbols/phi.png'
imgPath = './drawings/drawing1.png'

def ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # apply OCR
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    return text

def searchLinear(stt, img):
    template_path = './Template/linear/'+stt+'-linear.png'
    template = cv2.imread(template_path, 0)
    loc, w, h = templateMatchingWithThreshold(img, template, 0.95)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, (pt[0] - 2*w, pt[1]), (pt[0] + 3*w, pt[1] + h), (0, 255, 255), 2)
        roi = img[pt[1]:pt[1] + h, pt[0] - 2*w: pt[0] + 3*w]
        template_phi = cv2.imread(template_phi_path, 0)
        loc_phi, w_phi, h_phi = templateMatchingWithThreshold(roi, template_phi, 0.8)
        for pt_phi in zip(*loc_phi[::-1]):
            cv2.rectangle(roi, pt_phi, (pt_phi[0] + w_phi, pt_phi[1] + h_phi), (255, 0, 255), 2)
        roi_text = img[pt[1] + 8:pt[1] + h + 8, pt[0] - w: pt[0] + 3*w]
        text = ocr(roi_text)
        print(text)
    return img

# Template Matching for symbols
def templateMatchingWithThreshold(img, template, threshold):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    #top_left = max_loc
    #bottom_right = (top_left[0] + w, top_left[1] + h)
    return loc, w, h

# Without threshold
def templateMatching(img, template):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left, bottom_right, h

# Search for GD&T
def searchTemplate(template, img, stt):
    template_path = './Template/vysttech-symbols/'+template+'.png' 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_1 = cv2.imread(template_path, 0)
    found = None
    top_left_1, bottom_right_1, h = templateMatching(img, template_1)
    w, h = template_1.shape[::-1]

    x = top_left_1[0]
    y = top_left_1[1]
    i = 1
    while (x + i) < 2500:
        #print(img_gray[y, x+i])
        if img_gray[y, x+i] == 255: 
            break 
        i+=1
    cv2.rectangle(img, (x,y), (x+i, y+h), 255, 2)
    # Insert image
    worksheet.insert_image(stt, 0, template_path,{'object_position':1})
    worksheet.set_row(stt, 50 )

    roi = img[y: y + h, x:x + i]
    # Template Matching for surface A
    template_a = cv2.imread(template_a_path, 0)

    top_left_a, bottom_right_a, h_a = templateMatching(roi, template_a)
    cv2.rectangle(roi, top_left_a, bottom_right_a, (0, 255, 0), 2)
    x_a = top_left_a[0]
    y_a = top_left_a[1]
    w_a, h_a = template_a.shape[::-1]
    #print(-x_a -w_a +i)
    # Template Matching for surface B
    if (i - x_a - w_a >10):
        template_b = cv2.imread(template_b_path, 0)

        top_left_b, bottom_right_b, h_b = templateMatching(roi, template_b)
        cv2.rectangle(roi, top_left_b, bottom_right_b, (0, 0, 255), 2)

    text_roi = img[y:y+h , x + w:x + x_a]
    text = ocr(text_roi)
    worksheet.write(stt, 1, text)
    worksheet.write(stt, 2, 'A')
    print("Diameter: ")
    print(text)
    return img

# Main
img  = cv2.imread(imgPath)
searchTemplate('1', img, 0)
searchTemplate('12', img, 1)
searchTemplate('3', img, 2)
searchLinear('1', img)
searchLinear('2', img)
searchLinear('4', img)
searchLinear('5', img)
searchLinear('8', img)
searchLinear('9', img)
workbook.close()
cv2.imshow('result', img)
cv2.waitKey(0)
