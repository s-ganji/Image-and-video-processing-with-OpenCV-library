import cv2 as cv;
import sys
import numpy as np

image=cv.imread('test.jpg')
cv.imshow('image',image)
cv.waitKey(0)
b = image.copy()
b[:, :, 1] = 0
b[:, :, 2] = 0
cv.imshow('B-RGB', b)
cv.waitKey(0)
img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('GrayScale', img_gray)
cv.waitKey(0)
blur= cv.GaussianBlur(img_gray,(5,5),0)
cv.imshow('gaussianBlur',blur)
cv.waitKey(0)
out=cv.transpose(image)
out=cv.flip(out,flipCode=1)
cv.imshow('rotated',out)
cv.waitKey(0)
small = cv.resize(image,None, fx=0.5,fy=1,interpolation=cv.INTER_CUBIC)
cv.imshow('resize',small)
cv.waitKey(0)
edges = cv.Canny(image,100,200)
cv.imshow('Edges',edges)
cv.waitKey(0)
ret, thresh = cv.threshold(img_gray, 0, 255,
                            cv.THRESH_BINARY_INV +
                            cv.THRESH_OTSU)
cv.imshow('segmentation', thresh)
cv.waitKey(0)
facecascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = facecascade.detectMultiScale(
    img_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv.CASCADE_SCALE_IMAGE
)
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow('faceDetection',image)
cv.waitKey(0)
cv.destroyAllWindows()
vidcap = cv.VideoCapture('test.avi')
vidcap.set(cv.CAP_PROP_POS_MSEC,10000)
count = 5
while (vidcap.isOpened() and count>0):
    ret, img = vidcap.read()
    cv.imshow('frame',img)
    count=count-1
    if cv.waitKey(500) & 0xFF == ord('q'):
        break
vidcap.release()
cv.destroyAllWindows()
