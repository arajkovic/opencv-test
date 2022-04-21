from math import floor
import numpy as np
import cv2 as cv

im = cv.imread('rectangle-triangle-circle.JPG')
height, width, _ = im.shape
print(f'{width}x{height}')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
averageIntensity = np.average(imgray)

# ret, thresh = cv.threshold(imgray, averageIntensity * 0.75, 255, cv.THRESH_BINARY)
threshold = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 101, 25)
# cv.imshow('threshold', threshold)

kernelSize = floor(min(width, height) * 0.01)
kernel = np.ones((kernelSize, kernelSize), np.uint8)

eroded = cv.erode(threshold, kernel)
dilated = cv.dilate(eroded, kernel)
canny = cv.Canny(dilated, 50, 200, 3)
# cv.imshow('eroded', eroded)
cv.imshow('canny', canny)

lines = cv.HoughLinesP(canny, 1, np.pi / 180, 100, np.array([]), 100, 20)

numOfLines, _, _ = lines.shape
print(f'{numOfLines} lines found.')

for line in lines:
  x1, y1, x2, y2 = line[0]
  cv.line(im, (x1, y1), (x2, y2), (0, 255, 0), 4)

cv.imshow('contours', im)

if cv.waitKey(0) & 0xff == 27:
  cv.destroyAllWindows()