import cv2
import numpy as np
import matplotlib.pyplot as plt
from commonfunctions import *
from Table_Text_Extraction import *
from Image_Preprocessing import *
def main():
    img=load_image("test.png")
    table=Table_Preprocessing(img)
    rect1=MorphProcessing(table)
    rect2=HoughProcessing(table)
    rect3=ContourProcessing(table)
    result,exp=get_most_exact_algo(rect1,rect2,rect3)
    candidates={'morph':rect1,'hough':rect2,'contours':rect3}
    final_candidate=None
    if(result==None):
      return "Error, Table Couldn't be detected :("
    else:
        final_candidate=candidates[result]
    model=load_model("../ML models/numbers_model.pkl")
    last_col=extract_last_col(final_candidate,get_n_cells_cols(exp))
    last_col = last_col[1:]
    final_result=[]
    bits=extract_binaries(last_col)
    for bit in bits:
        final_result.append(model.predict(bit))
    
              
    
  
    
    