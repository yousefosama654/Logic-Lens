import cv2
import numpy as np
import matplotlib.pyplot as plt
from commonfunctions import *
import pickle
import skimage.io as io
from skimage.feature import hog
def remove_edges(img):
    image = img.copy()
    h = image.shape[0]
    w = image.shape[1]
    image=image[int(0.05*h):int(0.95*h), int(0.01*w):int(0.99*w)]
    image = cv2.resize(image,(w,h))
    return image
def find_biggest_contours(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 10000:
            peri = cv2.arcLength(i, True)
            corners = cv2.approxPolyDP(i, 0.02 * peri, True)
            # to check that it is a rect and has the max area
            if area > max_area and len(corners) == 4:
                biggest = corners
                max_area = area
    return biggest
def extract_hog_features(img): 
    img=cv2.resize(img,(64,64))
    fd, hog_image = hog(
        img,
        pixels_per_cell=(2, 2),
        cells_per_block=(2, 2),
        visualize=True,        
    )
    return fd,hog_image
def Deskew(image:np.ndarray,show=False): 
    img_original = image.copy()
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # reduce unwanted noise very well while keeping edges fairly sharp
    gray = cv2.bilateralFilter(gray, 50, 15, 15)
    edged = cv2.Canny(gray, 50, 50)
    kernel = np.ones((7, 7))
    dilatedImg = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(dilatedImg, kernel, iterations=1)

    # Contour detection
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contour_Frames = img.copy()
    Contour_Frames = cv2.drawContours(Contour_Frames, contours, -1,  (0, 255, 0), 10)
    if(show):   show_images([Contour_Frames])
    height, width, _ = img.shape
    biggest_contours = find_biggest_contours(contours)
    if(len(biggest_contours)==0): 
        biggest_contours= np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])        
    if(show):    print(biggest_contours)
    new_img = cv2.drawContours(img, [biggest_contours], -1, (0, 255, 0), 3)
    if(show):   show_images([new_img],["after countour"])
    # reorder corner pixels (src)
    points = biggest_contours.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]


    (top_left, top_right, bottom_right, bottom_left) = input_points
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))



    # Output image size
    max_width = max(int(bottom_width), int(top_width))
    max_height = int(max_width * 1.414)  # for A4

    # Desired points values in the output image (dest)
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    # calculate the transformation matrix
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))
    #calculate the final image
    return img_output
def get_letters(img, verbose = False,single_letter = False):
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = img.shape
    retSize = (64,64)    
    tolerance = 0

    #thresholding the image to a binary image
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,75,15)
    thresh=~thresh
    if verbose:
        show_images([thresh],['thresh'])


    # Find the contours
    contours,_ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    test_img=img.copy()
    cv2.drawContours(test_img, contours, -1, (0,255,0), 3)
    if verbose:
        show_images([test_img],['contours before filtering'])
    average_area = 0.0   
    max_area = 0.0   
    max_width  = 0.0  
    max_height  = 0.0 
    average_width  = 0.0  
    average_height  = 0.0 
    
    # only keep the contours that are black  and i needed to discard the small parts
    contours = list(filter(lambda cnt: cv2.contourArea(cnt,True) > 0, contours))
    contours_list = [[box,c] for box,c in [(cv2.boundingRect(c),c) for c in contours]]
    if(single_letter):
        contours_list = sorted(contours_list, key=lambda ctr:cv2.contourArea(ctr[1]))
    else:
        contours_list = sorted(contours_list, key=lambda ctr:ctr[0][0])
    
    
    if len(contours_list) != 0:
        average_area   = sum([ cv2.contourArea(cnt[1]) for cnt in contours_list]) / len(contours_list)
        max_area   = max([cv2.contourArea(cnt[1]) for cnt in contours_list])
        max_width  = max([cnt[0][2] for cnt in contours_list]) 
        max_height = max([cnt[0][3] for cnt in contours_list])
        average_width = sum([cnt[0][2] for cnt in contours_list]) / len(contours_list)
        average_height = sum([cnt[0][3] for cnt in contours_list]) / len(contours_list)
    # only keep the contours that are black  and i needed to discard the small parts
    # x,y,w,h 
    def filter_contours(cnt):
        (x,y,w,h),q = cnt
        # filter contours that are too small
        return  cv2.contourArea(q) >= max_area * 0.95 if single_letter else cv2.contourArea(q) > average_area * 0.2 # and  w > max_width * 0.25 
    contours_list = list(filter(filter_contours, contours_list))
    
    def is_contour_inside_another(box1, box2):
        x1,y1,w1,h1 = box1
        x2,y2,w2,h2 = box2  
        return (x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2 )


    contours_to_remove=[]
    for ind1,(box1,_) in enumerate(contours_list):
        for ind2,(box2,_) in enumerate(contours_list):
            if ind1 == ind2:
                continue  
            if is_contour_inside_another(box1,box2):
                contours_to_remove.append(ind2)
    # Remove the contours that are inside another contour
    contours_list = [contour for idx, contour in enumerate(contours_list) if idx not in contours_to_remove]


    if verbose:
        test_img=img.copy()
        for box,_ in contours_list:
            x,y,w,h = box
            cv2.rectangle(test_img,(x-tolerance,y-tolerance),(x+w+tolerance,y+h+tolerance),(0,255,0),1)
        show_images([test_img],['contours after filtering'])
    masks = []
    for cont in contours_list:
        mask = np.zeros(img.shape, np.float32)
        cv2.drawContours(mask, [cont[1]], 0, (1,1,1),-1)
        masks.append(mask)
        
    # merge list that are too close in x axis
        '''
        union of two BoxRectangles 
        '''    
    def union(a,b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    for ind,(box,_) in enumerate(contours_list):
        x,y,w,h = box
        prev_x = float('-inf') if ind == 0 else contours_list[ind-1][0][0]   
        prev_h = float('-inf') if ind == 0 else contours_list[ind-1][0][3]   
        if (x - prev_x < max_width *0.2 and abs(h - prev_h) < max_height * 0.2):
            # merge contours 
            contours_list[ind] = (union(contours_list[ind][0],contours_list[ind-1][0]),contours_list[ind][1])
            contours_list.pop(ind-1)
            # merge masks of the letters
            masks[ind] = masks[ind] + masks[ind-1]
            masks.pop(ind-1)
            ind -=1
            
        
    # For each contour, find the bounding rectangle and draw it
    ret_images = []
    # x-15 : x+15
    # y- 5 : y+5
    pad_h = 5
    pad_w = 18
    for ind,(box,_) in enumerate(contours_list):
        x,y,w,h = tuple(box)
        new_img  = img[y-tolerance:y+h+tolerance, x-tolerance:x+w+tolerance].astype(np.uint8)
        new_img=cv2.adaptiveThreshold(new_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,75,15)
        new_img = np.pad(new_img, ((pad_h,pad_h),(pad_w,pad_w)), 'constant')
        ret_images.append(new_img)


    if verbose:
        print('contours after merging')
        plt.show()
        
    for i in range(len(ret_images)):
        if(ret_images[i].shape != (0,0)):
            ret_images[i] = cv2.resize(ret_images[i], retSize)        
    if verbose:
        print('letters after resizing')
        show_images(ret_images)
        
    if(len(ret_images) == 0):
        return [cv2.resize(img.max()-img, retSize)]
    

    return ret_images
def expression_preprocessing(image,verbose=False):
    img=image.copy()
    Deskewed_image=Deskew(img,verbose)
    #show_images([Deskewed_image],["after deks"])
    img_output=remove_edges(Deskewed_image)
    letters=get_letters(img_output)
    if verbose:
        show_images(letters)
    return letters
def load_image(path):
    img = io.imread(path)
    return img
def get_distinct_letters(letters):
    distinct_letters=[]
    letters_to_compare=['B','E','Y','H']
    for letter in letters:
        if letter not in distinct_letters and letter in letters_to_compare:
            distinct_letters.append(letter)
    return distinct_letters

def construct_expression(letters):
    expression=""
    for letter in letters:
        if (letter =="^"):
            letter="and"
        elif(letter=="v"):
            letter="or"
        elif(letter=="~"):
            letter="not"
        elif(letter=="*"):
            letter="xor"
        expression+=" "+ letter
    return expression
def load_model(name):
  with open(name, 'rb') as file:
      loaded_array_list = pickle.load(file)
      return loaded_array_list
def get_char_from_digit(digit):
    char = {
        0: 'B',
        1: 'E',
        2: 'H',
        3: 'Y',
        4: 'XOR',
        5: 'AND',
        6: 'NOT',
        7: 'OR',
        8: '(',
        9: ')'
    }
    return char[digit]
def calculate_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:

    return np.mean((feat1 - feat2) ** 2)
def predict(image,features):
    img=image.copy()
    feature_extract,_=extract_hog_features(img)
    #show_images([img])
    distances: float = [
        calculate_distance(feature_extract, src_feat[:-1]) 
        for src_feat in features
        ]
    min_distance_index = np.argmin(distances)
    result=get_char_from_digit(features[min_distance_index][-1])
    return result
