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
def fun0():
    shape = (1024, 1280)
    img_list = [np.ones(shape, dtype="uint16") for i in xrange(3)]
    for i in range(0, 256, 1):
        img_list[1][:, i*5:(i+1)*5] = i

    img_list[0] = img_list[1] / 2


    img_list[2] = img_list[1] * 2
    img_list[2] = np.clip(img_list[2], 0, 255)

    img_list[0] = img_list[0].astype(np.uint8)
    img_list[1] = img_list[1].astype(np.uint8)
    img_list[2] = img_list[2].astype(np.uint8)

    for i in xrange(3):
        cv2.imshow("window", img_list[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("../image/test_img/test3_img"+str(i)+".png", img_list[i])

def fun1(img_name):
    img = cv2.imread(img_name)
    data = img.reshape(1024, 1280*3)
    np.savetxt('../text/intensity', data, fmt='%i',)

if __name__ == "__main__":
    # fun0()
    fun1("../image/test_img/Gtonemap6.png")
    pass
