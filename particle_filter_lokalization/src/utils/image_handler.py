import numpy as np
import matplotlib.pyplot as plt
import os
import config.config as config

import cv2 as cv
def save_array_as_image(img, name): 
    name = path = config.paths['data_path']+name

    cv.imwrite(name+".png", img)
    #plt.plot(array[:,0], array[:,1])
    #plt.imsave(data_path+name+".png", array)

def image_to_array(name): 
    data_path = config.paths['data_path']
    print(data_path+name+".png")

    image =  cv.imread(data_path+name+".png",)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray_image