import cv2
import numpy as np
import matplotlib.pyplot as plt
from commonfunctions import *
from Expression_Extraction import *
import ttg

def main():
    img=load_image("trainlet.png")
    letters=expression_preprocessing(img)
    gscale_letters=[cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY) for letter in letters]
    thresh_letters=[cv2.adaptiveThreshold(letter,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,75,15) for letter in gscale_letters]
    # classifier
    # distinct_letters=get_distinct_letters(letters)
    # expression=construct_expression(letters)
    #algorithm
    print(ttg.Truths(['a', 'b'],['a xor b']))

if __name__ == "__main__":
    main()   