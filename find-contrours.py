from math import floor
import numpy as np
import cv2 as cv

im = cv.imread('rectangle-triangle-circle.JPG')
imageHeight, imageWidth, _ = im.shape

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
averageIntensity = np.average(imgray)

# ret, thresh = cv.threshold(imgray, averageIntensity * 0.75, 255, cv.THRESH_BINARY)
threshold = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 101, 25)
# cv.imshow('threshold', threshold)

kernelSize = floor(min(imageWidth, imageHeight) * 0.01)
kernel = np.ones((kernelSize, kernelSize), np.uint8)

eroded = cv.erode(threshold, kernel)
dilated = cv.dilate(eroded, kernel)
cv.imshow('eroded', eroded)
cv.imshow('dilated', dilated)

contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE )

approxShapes = []

contours = contours[1:]

for contour in contours:
  area = cv.contourArea(contour)
  minimumDifference = 1.0
  shape = ""
  params = []

  x, y, w, h = cv.boundingRect(contour)
  rectangleArea = w * h
  difference = abs((rectangleArea - area)) / area
  print(f'Rect Diff: {difference}')
  if difference < minimumDifference:
    shape = 'rectangle'
    params = [x, y, w, h]
    minimumDifference = difference
    

  center, radius = cv.minEnclosingCircle(contour)
  circleArea = np.pi * radius**2
  difference = abs((circleArea - area)) / area
  print(f'Circ Diff: {difference}')
  if difference < minimumDifference:
    shape = 'circle'
    params = [center, radius]
    minimumDifference = difference

  triangleArea, triangle = cv.minEnclosingTriangle(contour)
  difference = abs((triangleArea - area)) / area

  if difference < minimumDifference:
    shape = 'triangle'
    params = triangle
    minimumDifference = difference

  approxShapes.append([shape, params])

for shape, params in approxShapes:
  if shape == 'rectangle':
    cv.rectangle(im, params, (0, 0, 255), 3)

  if shape == 'circle':
    (x, y), radius = params
    cv.circle(im, (int(x), int(y)), int(radius), (0, 255, 0), 3)

  if shape == 'triangle':
    cv.line(im, params[0][0].astype(int), params[1][0].astype(int), (255, 0, 0), 3)
    cv.line(im, params[1][0].astype(int), params[2][0].astype(int), (255, 0, 0), 3)
    cv.line(im, params[2][0].astype(int), params[0][0].astype(int), (255, 0, 0), 3)

# cv.drawContours(im, contours, -1, (0,255,0), 3)

cv.imshow('contours', im)

if cv.waitKey(0) & 0xff == 27:
  cv.destroyAllWindows()