#__author__ = 'gimbu'
#coding:utf-8
import sys
sys.path.append('../Tool/')
sys.path.append('../')
import cv2
import numpy as np
import glob
import timeit
import time
import matplotlib.pyplot as plt
from ImgProcessTool import image_process_tool as ipt
from FileInterfaceTool import FileInterfaceTool as fit

LDR_SIZE = 256
def tringleWeights():
    w = np.zeros((LDR_SIZE, ), dtype="float32")
    half = LDR_SIZE / 2
    for i in xrange(LDR_SIZE):
        if i <= half:
            w[i] = i + 1.0
        else:
            w[i] = LDR_SIZE - i
    return w

def GBCalibrate(w, img_name_list, time_list):
    gamma = 50.0
    img_list = []
    times = np.array(time_list, dtype="float32")
    for i in xrange(3):
        tmp = cv2.imread("../image/test_img/"+img_name_list[i])
        tmp2 = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        img_list.append(tmp2)
    A = np.zeros((70*3+LDR_SIZE+1, LDR_SIZE+70), dtype="float32")
    B = np.zeros((A.shape[0]), dtype="float32")
    eq = 0
    for i in xrange(70):
        r = np.random.randint(0, 1024)
        c = np.random.randint(0, 1280)
        for j in xrange(3):
            val = img_list[j][r, c]
            A[eq, val] = w[val]
            A[eq, LDR_SIZE + i] = -w[val]
            B[eq] = w[val] * np.log(times[j])
            eq += 1
    A[eq, LDR_SIZE / 2] = 1
    eq += 1
    for i in xrange(254):
        A[eq, i] = gamma * w[i]
        A[eq, i+1] = -2 * gamma * w[i]
        A[eq, i+2] = gamma * w[i]
        eq += 1
    _, F = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    F = np.exp(F)
    np.savetxt('../text/hdr_Fv.txt', F[:256], fmt='%3.4f',)
    x = xrange(256)
    plt.plot(x, F[x], linewidth=2)
    plt.show()
    # time_list = img_inf_list["times"]
    # w = tringleWeights()
    # F = GBCalibrate(w, img_name_list, time_list)

# todo learn it

def GBFusion(path, filename):
    arg_list = fit.loadYaml(path + filename)
    img_name_list = arg_list["img_names"]
    wcon = arg_list["wcon"]
    wsat = arg_list["wsat"]
    wexp = arg_list["wexp"]
    #read some images in img_list
    img_list = []
    InputPath = arg_list["InputPath"]
    for i in xrange(3):
        tmp = cv2.imread(InputPath+img_name_list[i])
        img_list.append(tmp)
    channels = img_list[0].shape[2]
    rows, cols = img_list[0].shape[0], img_list[0].shape[1]
    shape = (rows, cols)
    #inittial the weights[]
    weights = []
    for i in xrange(3):
        tmp = np.zeros(shape, dtype="float32")
        weights.append(tmp)
    weight_sum = np.zeros(shape, dtype="float32")
    #solve each image weight and solve weight_sum
    for i in xrange(len(img_list)):
        #normalize img make convergence speedup
        img = img_list[i] / 255.0
        img = img.astype("float32")
        #convert to gray
        if channels == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
                gray = img.copyTo()
        #solve contrast, ddepth: cv2.CV_32F
        contrast = cv2.Laplacian(gray, cv2.CV_32F)
        contrast = abs(contrast)
        #
        mean = np.zeros(shape, dtype="float32")
        splitted = cv2.split(img_list[i])
        for img_cn in splitted:
            mean += img_cn
        mean /= channels
        #solve saturation
        saturation = np.zeros(shape, dtype="float32")
        for img_cn in splitted:
            deviation = img_cn - mean
            deviation = cv2.pow(deviation, 2.0)
            saturation += deviation
        saturation = cv2.sqrt(saturation)

        wellexp = np.ones(shape, dtype="float32")
        for img_cn in splitted:
            expo = cv2.subtract(img_cn, 0.5, dtype=cv2.CV_32F)
            expo = cv2.pow(expo, 2.0)
            expo = -expo / 0.08
            #larger '0.08' only make 'cv2.exp(expo)' nearest "1"
            expo = cv2.exp(expo)
            wellexp = cv2.multiply(wellexp, expo)
        # pow respective ratio
        contrast = cv2.pow(contrast, wcon)
        saturation = cv2.pow(saturation, wsat)
        wellexp = cv2.pow(wellexp, wexp)

        weights[i] = contrast
        if channels == 3:
            weights[i] = cv2.multiply(weights[i], saturation)
        weights[i] = cv2.multiply(weights[i], wellexp) + 1e-12
        weight_sum += weights[i]

    maxlevel = int(np.log(min(rows, cols)) / np.log(2))
    #(maxlevel+1) images, following to solve the final pyramid.
    res_pyr = [0] * (maxlevel+1)
    for i in xrange(len(img_list)):
        img_pyr = [0] * (maxlevel+1)
        weight_pyr = [0] * (maxlevel+1)
        img = img_list[i] / 255.0
        img = img.astype("float32")
        img_pyr[0] = img
        weights[i] /= weight_sum
        weight_pyr[0] = weights[i]
        # following: buildPyramid(img, img_pyr, maxlevel)
        # buildPyramid(weights[i], weight_pyr, maxlevel)
        #todo inspection it

        for lvl in xrange(maxlevel):
            img_pyr[lvl+1] = cv2.pyrDown(img_pyr[lvl])
        for lvl in xrange(maxlevel):
            #size = width, height
            size = img_pyr[lvl].shape[:2][::-1]
            up = cv2.pyrUp(img_pyr[lvl+1], dstsize=size)
            img_pyr[lvl] -= up

        for lvl in xrange(maxlevel):
            weight_pyr[lvl+1] = cv2.pyrDown(weight_pyr[lvl])

        for lvl in xrange(maxlevel+1):
            splitted = cv2.split(img_pyr[lvl])
            splitted2 = []
            for img_pyr_cn in splitted:
                tmp = cv2.multiply(img_pyr_cn, weight_pyr[lvl])
                splitted2.append(tmp)
            cv2.merge(splitted2, img_pyr[lvl])
            # first image to assign res_pry[0-maxlevel]
            if i == 0:
                res_pyr[lvl] = img_pyr[lvl]
            # latter image to assign res_pry[0-maxlevel]
            else:
                res_pyr[lvl] += img_pyr[lvl]

    for lvl in range(1, maxlevel+1, 1)[::-1]:
        #size = width, height
        size = dstsize=res_pyr[lvl-1].shape[:2][::-1]
        up = cv2.pyrUp(res_pyr[lvl], dstsize=size)
        res_pyr[lvl - 1] += up

    dst_tmp = res_pyr[0]
    dst_tmp = dst_tmp * 255
    dst = dst_tmp.astype("uint8")
    OutputPath = arg_list["OutputPath"]
    cv2.imwrite(OutputPath, dst)
    cv2.namedWindow("window")
    cv2.imshow("window", dst)
    cv2.waitKey()
    cv2.destroyWindow("window")




if __name__ == '__main__':
    path = "../text/"
    filename = "multi_img_inform.yaml"
    GBFusion(path, filename)













