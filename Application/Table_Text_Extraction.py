import cv2
import numpy as np
import matplotlib.pyplot as plt
from commonfunctions import *

class TextExtractionTableMorph:
      contours=None
      gscale_img=None
      def __init__(self, image):
        self.image = image  
        self.gscale_img=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
      def binary_threshold(self):
         self.image = cv2.adaptiveThreshold(self.gscale_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,75,15)
      def morph_op(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        opening = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) 
        self.image=closing 
      def remove_v_h_lines(self):
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        eroded_image_v = cv2.erode(self.image, vertical_kernel, iterations=5)
        eroded_image_h = cv2.erode(self.image, horizontal_kernel, iterations=4)
        subtracted_image = cv2.subtract(self.image, eroded_image_h)
        subtracted_image = cv2.subtract(subtracted_image, eroded_image_v)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        subtracted_image = cv2.morphologyEx(subtracted_image, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        subtracted_image = cv2.morphologyEx(subtracted_image, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        subtracted_image = cv2.morphologyEx(subtracted_image, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        subtracted_image = cv2.morphologyEx(subtracted_image, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        subtracted_image=cv2.dilate(subtracted_image,kernel,iterations=2)
        self.image=subtracted_image
      def get_contours(self):
        self.contours, hierarchy = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      def filter_h_contours(self):
        boxes_h = np.array([cv2.boundingRect(c) for c in self.contours])
        #! These first two columns typically represent the top-left corner coordinates of the bounding boxes (x and y).
        #! The last two columns represent the width and height of the bounding box, respectively.
        #! calculate the center of the bounding boxes
        centers_h = boxes_h[:, :2] + boxes_h[:, 2:] / 2
        #! convert centres to a column vector with either x or y distances then generate a matrix of distances from each centre of #rows = #centres and #cols = #centres
        #! remove contours with no neighbours vertically
        horizontal_distances = np.abs(centers_h[:, 0][:, np.newaxis] - centers_h[:, 0])
        horizontal_distances[horizontal_distances <= 15] = 0
        horizontal_distances[np.arange(horizontal_distances.shape[0])[:, np.newaxis] == np.arange(horizontal_distances.shape[1])] = 1000000
        #! count the number of neighbor contours vertically
        num_zero_indices_h = np.sum(horizontal_distances == 0, axis=1)
        indices_to_remove_h = np.where(num_zero_indices_h < np.ceil(np.mean(num_zero_indices_h)))

        if (len(num_zero_indices_h) > 100):
          indices_to_remove_h = np.where(num_zero_indices_h < 4)

        mask_h = np.ones(len(self.contours), dtype=bool)
        mask_h[indices_to_remove_h[0]] = False
        self.contours = [np.array(contour) for i, contour in enumerate(self.contours) if i not in indices_to_remove_h[0]]
      def filter_v_contours(self):
        #! remove contours with no neighbours horizontally
        boxes_y = np.array([cv2.boundingRect(c) for c in self.contours])
        centers_y = boxes_y[:, :2] + boxes_y[:, 2:] / 2

        vertical_distances = np.abs(centers_y[:, 1][:, np.newaxis] - centers_y[:, 1]) 
        vertical_distances[vertical_distances <= 15] = 0
        vertical_distances[np.arange(vertical_distances.shape[0])[:, np.newaxis] == np.arange(vertical_distances.shape[1])] = 1000000
        #! count the number of neighbor contours horizontally
        num_zero_indices_y = np.sum(vertical_distances == 0, axis=1)
        indices_to_remove_y = np.where(num_zero_indices_y < np.ceil(np.mean(num_zero_indices_y)))
        if (len(num_zero_indices_y) > 100):
            indices_to_remove_y = np.where(num_zero_indices_y < 4)
        mask_v = np.ones(len(self.contours), dtype=bool)
        mask_v[indices_to_remove_y[0]] = False
        self.contours = [np.array(contour) for i, contour in enumerate(self.contours) if i not in indices_to_remove_y[0]]
      def draw_rect(self):
        todraw=self.gscale_img.copy()
        rectangles = np.array([cv2.boundingRect(c) for c in self.contours])
        x_offset=15
        y_offset=2
        for rect in rectangles:
            x, y, w, h = rect
            cv2.rectangle(todraw, (x-x_offset, y-y_offset-3), (x +x_offset +w, y + h+ y_offset -5), (0, 255, 0), 2)
        return np.array(rectangles)
                
                
class TextExtractionTableHough:
    contours=None
    gscale_img=None
    result=None
    Cannyimg=None
    def __init__(self, image):
        self.image = image  
        self.gscale_img=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    def binary_threshold(self):
        self.image = cv2.adaptiveThreshold(self.gscale_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,75,15)
    def Canny_Det(self):
        self.Cannyimg = cv2.Canny(self.gscale_img, 100, 200)
    def Hough_lines(self):
        lines = cv2.HoughLines(self.Cannyimg,1, np.pi / 180, 150, None, 80, 10)
        line_image=np.copy(image)*0
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(line_image, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        line_image_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        _, thresholded_lines = cv2.threshold(line_image_gray, 1, 255, cv2.THRESH_BINARY)
        self.result = cv2.subtract(self.image, thresholded_lines)
        self.contours, hierarchy = cv2.findContours(self.result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def draw_rect(self):
        todraw=self.gscale_img.copy()
        min_rectangle_area = 100
        rectangles = np.array([cv2.boundingRect(c) for c in self.contours])
        rectangles = [rect for rect in rectangles if rect[2] * rect[3] > min_rectangle_area]
        x_offset=18
        y_offset=5
        for rect in rectangles:
            x, y, w, h = rect
            cv2.rectangle(todraw, (x-x_offset, y-y_offset-3), (x +x_offset +w, y + h+ y_offset -5), (0, 255, 0), 2)
        return np.array(rectangles)





        
class TextExtractionTableContours:
    gscale_img=None
    result=None
    def __init__(self, image):
        self.image = image  
        self.gscale_img=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    def binary_threshold(self):
        self.image = cv2.adaptiveThreshold(self.gscale_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,75,15)
    def Contours_detection(self):
        contours, _ = cv2.findContours(self.image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour1 = self.gscale_img.copy()   
        contour_image = np.zeros_like(self.image)
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                cv2.drawContours(contour_image, [contour], -1, (255), thickness=cv2.FILLED)
        self.result = cv2.bitwise_and(self.image, self.image, mask=contour_image)
    def draw_rect(self):
        contours, hierarchy = cv2.findContours(self.result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        todraw=self.gscale_img.copy()
        min_rectangle_area = 50
        rectangles = np.array([cv2.boundingRect(c) for c in contours])
        rectangles = [rect for rect in rectangles if rect[2] * rect[3] > min_rectangle_area]
        x_offset=18
        y_offset=5
        for rect in rectangles:
            x, y, w, h = rect
            cv2.rectangle(todraw, (x-x_offset, y-y_offset-3), (x +x_offset +w, y + h+ y_offset -5), (0, 255, 0), 2)
        return np.array(rectangles)


 
def MorphProcessing(image):
    table=TextExtractionTableMorph(image)
    table.binary_threshold()
    table.morph_op()
    table.remove_v_h_lines()
    table.get_contours()
    table.filter_h_contours()
    table.filter_v_contours()
    return table.draw_rect()

def HoughProcessing(image):
    table=TextExtractionTableHough(image)
    table.binary_threshold()
    table.Canny_Det()
    table.Hough_lines()
    return table.draw_rect()

def ContoursProcessing(image):
    table=TextExtractionTableContours(image)
    table.binary_threshold()
    table.Contours_detection()
    return table.draw_rect()

def extract_first_row(rectangles,n_cells):
    first_row = sorted(rectangles, key=lambda rect: rect[1])
    n_cells = 6  
    first_row = first_row[:n]
    first_row=sorted(first_row, key=lambda rect: rect[0])

def extract_first_col(rectangles,n_cells):
   first_column = sorted(rectangles, key=lambda rect: rect[0], reverse=True)
   n_cells = 33  
   first_column = first_column[:n]
   first_column=sorted(first_column, key=lambda rect: rect[1])
def is_power_of_2(num,n):
    # Check if the number has only one '1' bit using bitwise AND
    return   int(num) > 0 and  int(num) == 2 ** n
def get_most_exact_algo(rect1,rect2,rect3):
    working=None
    rect1 -=1
    rect2 -=1
    rect3 -=1
    for i in range(1,6):
        if (is_power_of_2((rect1-i)/(i+1),i)):
           return "morph"
        if (is_power_of_2((rect2-i)/(i+1),i)):
           return "hough"
        if (is_power_of_2((rect3-i)/(i+1),i)):
           return "contours"
    return None  
    