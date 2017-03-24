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
import cProfile
import re

LDR_SIZE = 256
def TringleWeights():
    w = np.zeros((LDR_SIZE, ), dtype="float32")
    half = LDR_SIZE / 2
    for i in xrange(LDR_SIZE):
        if i < half:
            w[i] = float(i+1)
        elif i >= half and i < 254:
            w[i] = float(LDR_SIZE - i)
        else:
            w[i] = 2
    return w

def GaussianFun(x, mu, sig):
    # left = 1./(np.sqrt(2.*np.pi)*sig)
    left = 128
    right = np.exp(-(x - mu) * (x - mu) / (2 * sig * sig))
    return left * right

def GaussianWeights():
    w = np.zeros((LDR_SIZE, ), dtype="float32")
    for i in xrange(LDR_SIZE):
            w[i] = GaussianFun(i, 127.5, 50)
    return w

def GB_Calibrate(path, filename):
    arg_list = fit.loadYaml(path + filename)
    img_name_list = arg_list["img_names"]
    time_list = arg_list["times"]
    w = GaussianWeights()
    gamma = 50.0
    times = np.array(time_list, dtype="float32")
    img_list = []
    # n_img = len(img_name_list) 优选即
    n_img = 2
    for i in xrange(n_img):
        #RGB
        # img = cv2.imread("../image/test_img/"+img_name_list[i])
        img = cv2.imread("../image/avr_img1x1/"+img_name_list[i])
        img_list.append(img)
    n_chn = img_list[0].shape[2]
    img_channel_list = []
    for i in xrange(n_chn):
        tmp = []
        for j in xrange(n_img):
            img_channel = cv2.split(img_list[j])[i]
            tmp.append(img_channel)
        img_channel_list.append(tmp)

    n_samples = 30
    img_cols = img_list[0][1]
    img_rows = img_list[0][0]
    sample_points_list = []
    if True:
        for i in xrange(n_samples):
            r = np.random.randint(0, 1024)
            c = np.random.randint(0, 1280)
            sample_points_list.append((r, c))

    # if True:
    #     sample_points_list = []
    #     x_points = int(np.sqrt(70 * img_cols / img_rows))
    #     y_points = samples / x_points
    #     step_x = img_cols / x_points
    #     step_y = img_rows / y_points
    #     r = step_x / 2
    #     c = step_y / 2
    #     for j in xrange(y_points):
    #         r = r + j * step_y
    #         for i in xrange(x_points):
    #             c = c + i * step_x
    #             sample_points_list.append((r, c))


    response_list = []
    for z in xrange(n_chn):
        eq = 0
        A = np.zeros((n_samples*n_img+LDR_SIZE+1, LDR_SIZE+n_samples), dtype="float32")
        B = np.zeros((A.shape[0]), dtype="float32")
        for i in xrange(n_samples):
            r = sample_points_list[i][0]
            c = sample_points_list[i][1]
            for j in xrange(n_img):
                val = img_channel_list[z][j][r, c]
                A[eq, val] = w[val]
                A[eq, LDR_SIZE + i] = -w[val]
                B[eq] = w[val] * np.log(times[j])
                eq += 1
        #F(128)曝光量对数设0, 也就是曝光量为单位1, 不关事
        A[eq, LDR_SIZE / 2] = 1
        eq += 1
        for i in range(0, 254):
            A[eq, i] = gamma * w[i]
            A[eq, i+1] = -2 * gamma * w[i]
            A[eq, i+2] = gamma * w[i]
            eq += 1
        _, response = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        # just a exposure lum.
        response = cv2.exp(response)
        response = response[:256]
        response_list.append(response)
    response_array = cv2.merge(response_list)
    response_array = response_array.reshape(256, 3)
    showSaveData(response_array, "hdr_response_gamma10.txt")
    #need return 256x3 nparray.
    return response_array

def showSaveData(_response_array, txt_name):
    np.savetxt('../text/'+txt_name, _response_array, fmt='%.2f',)
    _response_array = np.transpose(_response_array)
    x = np.array(xrange(256))
    plt.figure(1)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    plt.sca(ax1)
    plt.plot(x, _response_array[0], linewidth=2, color="b")
    plt.sca(ax2)
    plt.plot(x, _response_array[1], linewidth=2, color="g")
    plt.sca(ax3)
    plt.plot(x, _response_array[2], linewidth=2, color="r")
    plt.show()



def GB_mergeHDR(path, filename, response_256x3):
    arg_list = fit.loadYaml(path + filename)
    img_list = []
    img_name_list = arg_list["img_names"]
    n_img = len(img_name_list)
    for i in xrange(n_img):
        tmp = cv2.imread("../image/avr_img1x1/"+img_name_list[i])
        # tmp = cv2.imread("../image/test_img/"+img_name_list[i])
        img_list.append(tmp)

    time_list = arg_list["times"]
    times = np.array(time_list, dtype="float32")
    weights = GaussianWeights()
    # weights = np.ones((256,), dtype="float32")

    response_256x1x3 = response_256x3.reshape((256, 1, 3))
    log_response = np.log(response_256x1x3)
    log_time = np.log(times)
    channels = img_list[0].shape[2]

    shape = img_list[0].shape
    result = np.zeros(shape, dtype="float32")
    #list result_split
    result_split_list = cv2.split(result)
    weight_sum = np.zeros(shape[:2], dtype="float32")
    for i in xrange(n_img):
        splitted = cv2.split(img_list[i])
        w = np.zeros(shape[:2], dtype="float32")
        for splitted_cn in splitted:
            splitted_cn = cv2.LUT(splitted_cn, weights)
            w += splitted_cn
        #第n张图3个通道的平均权值图像
        w /= channels
        response_img = cv2.LUT(img_list[i], log_response)
        splitted = cv2.split(response_img)
        for cn in xrange(channels):
            #w:图片的平均权值 splitted[cn]:通道的log_response log_time[i]:图片的log_time.
            result_split_list[cn] += cv2.multiply(w, splitted[cn] - log_time[i])
            #全部图的平均权值的和
        weight_sum += w
    #全部图的平均权值的和的倒数
    weight_sum = 1.0 / weight_sum
    for cn in xrange(channels):
        result_split_list[cn] = cv2.multiply(result_split_list[cn], weight_sum)
    result = cv2.merge(result_split_list)
    #lux
    hdr_img = cv2.exp(result)
    return hdr_img

def mapLuminance(src, lum, new_lum, saturation):
    channels_list = cv2.split(src)
    #following just best
    for cn in xrange(len(channels_list)):
        channels_list[cn] = cv2.multiply(channels_list[cn], 1.0/lum)
        channels_list[cn] = cv2.pow(channels_list[cn], saturation)
        channels_list[cn] = cv2.multiply(channels_list[cn], new_lum)
    dst = cv2.merge(channels_list)
    return dst

def GB_ToneMapping(path, filename, hdr_img):
    arg_list = fit.loadYaml(path + filename)
    gamma = arg_list["gamma"]
    contrast = arg_list["contrast"]
    saturation = arg_list["saturation"]
    sigma_space = arg_list["sigma_space"]
    sigma_color = arg_list["sigma_color"]
    hdr_img_2d = hdr_img.reshape(1024, 1280*3)
    minval, maxvalue, _, _ = cv2.minMaxLoc(hdr_img_2d)
    img = (hdr_img - minval) / (maxvalue - minval)
    img = img.clip(1.0e-4)
    img = cv2.pow(img, 1.0 / gamma)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    log_img = np.log(gray_img)
    #todo list effect, verbose bilateral: validate 0 -> small -0.03 -> small map_img
    map_img = cv2.bilateralFilter(log_img, -1, sigma_color, sigma_space)
    minval, maxval, _, _ = cv2.minMaxLoc(map_img)
    scale = contrast / (maxval - minval)
    map_img = cv2.exp(map_img * (scale - 1.0) + log_img)
    img = mapLuminance(img, gray_img, map_img, saturation)
    img = cv2.pow(img, 1.0 / gamma)
    #no problem!!
    img = img.clip(None, 1.0)
    img = img * 255
    img = img.astype("uint8")
    OutputPath = '../image/test_img/Gtonemap7.png'
    cv2.imwrite(OutputPath, img)
    cv2.imshow("window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#todo: optimize the algorithm
def GB_Fusion(path, filename):
    arg_list = fit.loadYaml(path + filename)
    img_name_list = arg_list["img_names"]
    wcon = arg_list["wcon"]
    wsat = arg_list["wsat"]
    wexp = arg_list["wexp"]
    #read some images in img_list
    img_list = []
    InputPath = arg_list["InputPath"]
    n_img = len(img_name_list)


    for i in xrange(n_img):
        tmp = cv2.imread(InputPath+img_name_list[i])
        # tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
        img_list.append(tmp)
    #camera images both are [b, g, r] 3channels
    n_chn = img_list[0].shape[2]
    # n_chn = 1
    rows, cols = img_list[0].shape[0], img_list[0].shape[1]
    shape = (rows, cols)
    #inittial the weights[]
    weights = []
    for i in xrange(n_img):
        tmp = np.zeros(shape, dtype="float32")
        weights.append(tmp)
    weight_sum = np.zeros(shape, dtype="float32")
    #solve each image weight and solve weight_sum
    for i in xrange(len(img_list)):
        #normalize img make convergence speedup
        img = img_list[i] / 255.0
        img = img.astype("float32")
        #convert to gray
        if n_chn == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
                gray = img
        #solve contrast, ddepth: cv2.CV_32F
        contrast = cv2.Laplacian(gray, cv2.CV_32F)
        contrast = abs(contrast)
        #
        mean = np.zeros(shape, dtype="float32")
        splitted = cv2.split(img_list[i])
        for img_cn in splitted:
            mean += img_cn
        mean /= n_chn
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
        if n_chn == 3:
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
    pr = cProfile.Profile()
    pr.enable()
    path = "../text/"
    filename = "multi_img_inform.yaml"

    response_256x3 = GB_Calibrate(path, filename)
    # 3.2.0 spend time 367ms
    hdr_img = GB_mergeHDR(path, filename, response_256x3)
    GB_ToneMapping(path, filename, hdr_img)
    #spend time 491ms
    # GB_Fusion(path, filename)

    cProfile.run('re.compile("GB_ToneMapping")', 'stats')
    pr.disable()
    pr.print_stats(sort='time')
    pass















