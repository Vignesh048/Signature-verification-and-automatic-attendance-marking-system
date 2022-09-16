import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\vigne\AppData\Local\Programs\Python\Python39\Scripts\Tesseract-OCR\tesseract.exe'

#read your file
file=r'data\attendance_sheets\attendance11.jpg'
img = cv2.imread(file,0)

#thresholding the image to a binary image
thresh,img_bin = cv2.threshold(img,165,255,cv2.THRESH_BINARY)

#inverting the image 
img_bin = 255-img_bin
# cv2.imwrite('/content/newimagest.jpeg',img_bin)
#Plotting the image to see the output
plotting = plt.imshow(img_bin,cmap='gray')

# Length(width) of kernel as 100th of total width
kernel_len = np.array(img).shape[1]//100
# Defining a vertical kernel to detect all vertical lines of image 
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

#Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
#cv2.imwrite("vertical.jpeg",vertical_lines)
#Plot the generated image
plotting = plt.imshow(image_1,cmap='gray')

#Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
#Plot the generated image
plotting = plt.imshow(image_2,cmap='gray')

# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
#Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
bitxor = cv2.bitwise_xor(img,img_vh)
bitnot = cv2.bitwise_not(bitxor)
#Plotting the generated image
plotting = plt.imshow(bitnot,cmap='gray')

# Detect contours for following box detection
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
          i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

# Sort all the contours by top to bottom.
contours, boundingBoxes = sort_contours(contours)

#Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
#Get mean of heights
mean = np.mean(heights)

#Create list box to store all boxes in  
box = []
img = cv2.imread(file,0)
# Get position (x,y), width and height for every contour and show the contour on image
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w>100 and w<1000 and h>10 and h<500):
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,125,10),2)
        box.append([x,y,w,h])
#plotting = plt.imshow(image,cmap="gray")

#Creating two lists to define row and column in which cell is located
row=[]
column=[]
j=0
#Sorting the boxes to their respective row and column
for i in range(len(box)):
    if(i==0):
        column.append(box[i])
        previous=box[i]
    else:
        if(box[i][1]<=previous[1]+mean/2):
            column.append(box[i])
            previous=box[i]
            if(i==len(box)-1):
                row.append(column)
        else:
            row.append(column)
            column=[]
            previous = box[i]
            column.append(box[i])

box = sorted(box, key = lambda x:(x[1],x[0]))
final = []
newl = []
prev = [0,0,0,0]
for i in box:

  if(newl==[] or prev[1]>=i[1]-20 and prev[1]<=i[1]+20 ):
      
      newl.append(i)
      prev = i

  else:
    final.append(newl)
    newl = [i]
    prev = i
final.append(newl)

count = 0
img = cv2.imread(file,0)
for boxes in final:
    
    print(len(boxes))
    if len(boxes) <4:
        continue

    boxes = sorted(boxes)
    roll = boxes[1]
    sign = boxes[3]
    

    texto = img[roll[1]:roll[1]+roll[3],roll[0]:roll[0]+roll[2]]
    cropped_image = img[sign[1]:sign[1]+sign[3],sign[0]:sign[0]+sign[2]]
    text = pytesseract.image_to_string(texto)
    text = re.findall('\d+',text)
    print(text)
    if text == []:
        continue
    text = text[0]
    if(len(text)==9):
        
        cv2.imwrite(r"data/signatures/set1/"+str(text).strip()+".jpeg",cropped_image)
plt.imshow(cropped_image)

