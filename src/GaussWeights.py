#coding:utf-8
import sys
sys.path.append('../Tool/')
import cv2
import numpy as np
import glob
import timeit
import time
import matplotlib.pyplot as plt
from ImgProcessTool import image_process_tool as ipt
from FileInterfaceTool import FileInterfaceTool as fit
import cProfile
import re

def gaussian(x, mu, sig):
    # left = 1./(np.sqrt(2.*np.pi)*sig)
    left = 128
    right = np.exp(-(x - mu) * (x - mu) / (2 * sig * sig))
    return left * right
if __name__ == "__main__":
    print cv2.getBuildInformation()
    for mu, sig in [(127.5, 50)]:
        plt.plot(np.linspace(0, 256, 256), gaussian(np.linspace(0, 256, 256), mu, sig))
    plt.show()
