import numpy as np
import matplotlib.pyplot as plt
import os
import config.config as config

import cv2 as cv
def save_array_as_image(img, path): 
    

    cv.imwrite(path+".png", img)
    #plt.plot(array[:,0], array[:,1])
    #plt.imsave(data_path+path+".png", array)

def image_to_array(path): 
    

    image =  cv.imread(path+".png",)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image