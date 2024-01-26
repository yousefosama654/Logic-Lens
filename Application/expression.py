import cv2
import numpy as np
import matplotlib.pyplot as plt
from commonfunctions import *
from Expression_Extraction import *
import ttg

def main_expression(path="./testcases/trainlet.png"):
    img=io.imread(path)
    #show_images([img],["original"])
    letters=expression_preprocessing(img,False)
    # show_images(letters)
    if(len(letters)==0):
        return "Empty Piece of Paper"
    
    letters=letters[2:]

    #show_images(letters)
    # cv2.imwrite("./thresh_letters.png",thresh_letters[0])
    # classifier
    features=load_model("../ML models/letters_model2.pkl")
    classified=[]
    for letter in letters:
        classified.append(predict(letter,features))
    print(classified)
    distinct_letters=get_distinct_letters(classified)
    print(distinct_letters)
    op = ['OR', 'NOT', 'AND', 'XOR']
    string_array_lower = [elem.lower() if elem in op else elem for elem in classified]
    result_string = " ".join(string_array_lower)
    #expression=construct_expression(letters)
    #algorithm
    table=""
    try:
        table= str(ttg.Truths(distinct_letters,[result_string]))
    except Exception as e:
        print("error")
    lines = table.split('\n')
    filtered_lines = [line for line in lines if ('+' not in line) and ('-' not in line)]
    formatted_table = '\n'.join(filtered_lines)
    print(formatted_table)
    return formatted_table
    
     

if __name__ == "__main__":
    main_expression()
    
