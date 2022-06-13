import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "MFM/30y/cs10/"

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image , rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

frame = 20
stretch = 1.0

def check_file(path):
    try:
        f = open(path)
        f.close()
        return True
    except FileNotFoundError:
        return False

def str2list(string):
    if "\n" in string: 
      string = string[:-1]
    try:
        string = string[1:-1]
        lst = list(string.split(","))
        if "" in lst: 
            lst = lst.remove("")
        return [int(i) for i in lst]
    except TypeError: 
        return [0]


blur = 67
l =  35

cy = [-5, 20]	 
cx = [0, 10]

static = "SC_000-sc_Phase-fwd.png"
image = cv2.imread(path + static)
image = rotate_image(image,-0.25) 
image = image[50:-50,50:-50]
thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (blur,blur), 0), 255, 1, 1, 231, l)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = cv2.drawContours(np.zeros(thresh.shape, np.uint8), contours, -1, 255, -1)
thresh = cv2.adaptiveThreshold(cv2.bitwise_not(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (blur,blur), 0)), 255, 1, 1, 231, l)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
mask = cv2.drawContours(mask, contours, -1, 255, -1)
x, y, w, h = cv2.boundingRect(mask)
mask = cv2.rectangle(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (x, y+h), (x+w, y), (55, 98, 226), thickness=20)
cv2.imwrite(path + 'rect.png', mask)
#cv2.imshow("mask", mask)
#cv2.waitKey()
#cv2.destroyAllWindows()
with open(path + "simple-reshape/params.txt", "a") as f: 
    f.write(static + "is basis for crop, blur = " + str(blur) + ", l = " + str(l) + "\n")


for i in range(0, 1):
    file = "SC_{:0>3}-sc_Phase-fwd.png".format(i)
    print(file)
    if check_file(path + file):
      image = cv2.imread(path + file)

      image = rotate_image(image, -0.25)
      image = image[50:-50, 50:-50]
  
      image = image[y + cy[0]-frame:y + h + cy[1]+frame,x + cx[0]-frame:x + w + cx[1]+frame]
      image = cv2.resize(image, (int(2000*stretch)-1, 1999)) 

      cv2.imwrite(path + "simple-reshape/" + file, image)
      sub = image
      with open(path + "simple-reshape/params.txt", "a") as f: 
        f.write(file + "\t frame = " + str(frame) + "\t cy = " + str(cy) + "\t cx = " + str(cx) + "\n")
    else: 
      print(file + "does not exist.")
