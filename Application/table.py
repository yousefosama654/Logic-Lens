import cv2
import numpy as np
import matplotlib.pyplot as plt
from commonfunctions import *
from Table_Text_Extraction import *
from Image_Preprocessing import *
from McCluskey import *
def main():
    img=load_image("../img/table_design.jpg")
    show_images([img],["this is the original image"])
    table=Table_Preprocessing(img)
    show_images([table],["this is tablea after Table_Preprocessing"])
    rect1=MorphProcessing(table.copy())
    rect2=HoughProcessing(table.copy())
    rect3=ContoursProcessing(table.copy())
    print(len(rect1),len(rect2),len(rect3) )
    result,exp=get_most_exact_algo(len(rect1),len(rect2),len(rect3))
    print(result)
    candidates={'morph':rect1,'hough':rect2,'contours':rect3}
    final_candidate=None
    if(result==None):
      return "Error, Table Couldn't be detected :("
    else:
        final_candidate=candidates[result]
    model=load_model("../ML models/numbers_model.pkl")
    last_col=extract_last_col(final_candidate,get_n_cells_cols(exp))
    print(len(last_col))
    last_col = last_col[1:]
    final_result=[]
    bits=extract_binaries(last_col,cv2.cvtColor(table, cv2.COLOR_BGR2GRAY))
    #show_images([bits[0]])
    i=0
    for bit in bits:
        final_result.append(model.predict([extract_hog_features(bit)]))
    result_array = np.concatenate(final_result)

    print(result_array)
    numbers =  [int(num) for num in result_array]
    numbers=np.array(numbers)
    numbers = np.where(numbers == 1)[0].tolist()
    print(numbers,exp)
    solveMcCluskey(numbers ,exp)
if __name__ == "__main__":
    main()   