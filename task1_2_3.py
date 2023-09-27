import cv2 as cv

#Reading image 
image = cv.imread('images.jpeg',1)

# Resizing of original image into 256*256
image_resized = cv.resize(image, (256,256), interpolation=cv.INTER_LINEAR)

# Displaying of original image.
cv.imshow('image', image)
cv.waitKey(0)

# Displaying Resized Image
cv.imshow('image', image_resized)
cv.waitKey(0)


# Conversion of original image into gray scale image.
gray_image = cv.cvtColor(image_resized,cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray_image)
cv.waitKey(0)

# Conversion of gray scale image into binary image
(thresh, binary_image)= cv.threshold(gray_image, 175, 255, cv.THRESH_BINARY)
cv.imshow('Binary Image',binary_image)
cv.waitKey(0)

cv.destroyAllWindows()
