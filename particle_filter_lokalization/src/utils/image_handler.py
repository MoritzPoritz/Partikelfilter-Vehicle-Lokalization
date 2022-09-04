import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
def save_array_as_image(img, name): 
    data_path = os.path.abspath(os.path.join(
        os.path.dirname('data'), 
        os.pardir)
    ) + '\\data\\'

    cv.imwrite(data_path+name+".png", img)
    #plt.plot(array[:,0], array[:,1])
    #plt.imsave(data_path+name+".png", array)

def image_to_array(name): 
    data_path = os.path.abspath(os.path.join(
        os.path.dirname('data'), 
        os.pardir)
    ) + '\\data\\'

    image =  cv.imread(data_path+name+".png",)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image