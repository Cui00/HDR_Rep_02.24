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
def GaussianWeights(mu=127.5, sig=50):
    """
    :Description: to generate a gaussian weights
    :param mu:  mu
    :param sig: sig
    :return w_256x: weights array
    """
    w_256x = np.zeros((LDR_SIZE, ), dtype="float32")
    for i in xrange(LDR_SIZE):
            # left = 1./(np.sqrt(2.*np.pi)*sig)
            left = 128
            right = np.exp(-(i - mu) * (i - mu) / (2 * sig * sig))
            w_256x[i] = left * right
    return w_256x

def GB_Calibrate(images, times, samples=70, random=False):
    """
    :Description: to calibrate CRF curve
    :param images: image list
    :param times: time list
    :param samples: samples point count
    :param random: whether samples random
    :return: response_256x1x3, CRF array
    """
    w = GaussianWeights()
    gamma = 10.0
    images = np.array(images, dtype="uint8")
    times = np.array(times, dtype="float32")
    n_img = len(images)
    n_chn = images[0].shape[2]
    img_channel_list = []
    for i in xrange(n_chn):
        tmp = []
        for j in xrange(n_img):
            img_channel = cv2.split(images[j])[i]
            tmp.append(img_channel)
        img_channel_list.append(tmp)
    img_shape = img_list[0].shape
    img_cols = img_shape[1]
    img_rows = img_shape[0]
    sample_points_list = []

    #set random situation.
    if random == True:
        for i in xrange(samples):
            r = np.random.randint(0, img_rows)
            c = np.random.randint(0, img_cols)
            sample_points_list.append((r, c))
    if random == False:
        x_points = int(np.sqrt(samples * (img_cols) / img_rows))
        y_points = samples / x_points
        n_samples = x_points * y_points
        step_x = img_cols / x_points
        step_y = img_rows / y_points
        r = step_x / 2
        for j in xrange(y_points):
            rr = r + j * step_y
            c = step_y / 2
            for i in xrange(x_points):
                cc = c + i * step_x
                sample_points_list.append((rr, cc))

    #svd solve response curve.
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

        # just from ln(lum) convert to lum.
        response = cv2.exp(response)
        response_256x1 = response[:256]
        response_list.append(response_256x1)
    response_256x1x3 = cv2.merge(response_list)
    response_256x3 = response_256x1x3.reshape(256, 3)
    showSaveData(response_256x3, "hdr_response_gamma10.txt")
    #need return 256x3 nparray.
    return response_256x1x3

def showSaveData(_response_256x3, filename):
    """
    :Description: showSaveData, show and save CRF curve data
    :param _response_256x3: CRF array
    :param filename: filename want to save
    :return None:
    """
    np.savetxt('../text/'+filename, _response_256x3, fmt='%.2f',)
    _response_array = np.transpose(_response_256x3)
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

def GB_mergeHDR(images, times, _response_256x1x3):
    """
    :Description: use images, times, and CRF to merge HDRI
    :param images: image list
    :param times: times list
    :param _response_256x1x3: CRF array
    :return hdr_img: HDRI(lux_img)
    """
    weights = GaussianWeights()
    images = np.array(images, dtype="uint8")
    times = np.array(times, dtype="float32")
    n_img = len(images)
    n_chn = images[0].shape[2]
    # response_256x1x3 = _response_256x3.reshape((256, 1, 3))
    log_response = np.log(_response_256x1x3)
    log_time = np.log(times)
    # log_hdr_img channel list
    hdr_chn_list = [0, 0, 0]
    img_avr_w_sum = np.zeros(images[0].shape[:2], dtype="float32")
    for i in xrange(n_img):
        src_chn_list = cv2.split(images[i])
        img_avr_w = np.zeros(images[0].shape[:2], dtype="float32")
        for cn in xrange(n_chn):
            img_cn_w = cv2.LUT(src_chn_list[cn], weights)
            img_avr_w += img_cn_w
        #第n张图3个通道的平均权值图像
        img_avr_w /= n_chn
        #一张图的log_response(log(lum))
        response_img = cv2.LUT(images[i], log_response)
        response_chn_list = cv2.split(response_img)
        for chn in xrange(n_chn):
            #w:图片的平均权值 response_chn_list[chn]:通道的log_response log_time[i]:图片的log_time.
            hdr_chn_list[chn] += cv2.multiply(img_avr_w, response_chn_list[chn] - log_time[i])
            #全部图的平均权值的和
        img_avr_w_sum += img_avr_w
    #全部图的平均权值的和的倒数
    img_avr_w_sum = 1.0 / img_avr_w_sum
    for cn in xrange(n_chn):
        hdr_chn_list[cn] = cv2.multiply(hdr_chn_list[cn], img_avr_w_sum)
    log_hdr_img = cv2.merge(hdr_chn_list)
    #this is lux, 为什么和官方的数值有数量级的差别。
    hdr_img = cv2.exp(log_hdr_img)
    return hdr_img

def mapLuminance(src, lum, new_lum, saturation):
    """
    :param src: BGR img
    :param lum: GRAY img
    :param new_lum: map img
    :param saturation: saturation
    :return new_img: new img
    """
    chn_list = cv2.split(src)
    #following just best
    for cn in xrange(len(chn_list)):
        chn_list[cn] = cv2.multiply(chn_list[cn], 1.0/lum)
        chn_list[cn] = cv2.pow(chn_list[cn], saturation)
        chn_list[cn] = cv2.multiply(chn_list[cn], new_lum)
    new_img = cv2.merge(chn_list)
    return new_img

def GB_ToneMapping(hdr_img):
    """
    :param hdr_img: HDRI(lux img)
    :return ldr_img: LDRI
    """
    path = "../text/"
    filename = "multi_img_inform.yaml"
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
    map_img = cv2.bilateralFilter(log_img, -1, sigma_color, sigma_space)
    minval, maxval, _, _ = cv2.minMaxLoc(map_img)
    scale = contrast / (maxval - minval)
    map_img = cv2.exp(map_img * (scale - 1.0) + log_img)
    img = mapLuminance(img, gray_img, map_img, saturation)
    img = cv2.pow(img, 1.0 / gamma)
    #no problem!!
    img = img.clip(None, 1.0)
    img = img * 255
    ldr_img = img.astype("uint8")
    return ldr_img

#todo: optimize the algorithm
def GB_Fusion(images):
    """
    :param images: img list
    :return fusion_img: fusion_img
    """
    path = "../text/"
    filename = "multi_img_inform.yaml"
    arg_list = fit.loadYaml(path + filename)
    wcon = arg_list["wcon"]
    wsat = arg_list["wsat"]
    wexp = arg_list["wexp"]
    n_img = len(images)
    #camera images both are [b, g, r] 3channels
    n_chn = images[0].shape[2]
    # n_chn = 1
    rows, cols = images[0].shape[0], images[0].shape[1]
    shape = (rows, cols)
    #inittial the weights[]
    weights = []
    for i in xrange(n_img):
        tmp = np.zeros(shape, dtype="float32")
        weights.append(tmp)
    weight_sum = np.zeros(shape, dtype="float32")
    #solve each image weight and solve weight_sum
    for i in xrange(len(images)):
        #normalize img make convergence speedup
        img = images[i] / 255.0
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
        splitted = cv2.split(images[i])
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
    for i in xrange(len(images)):
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
    fusion_img = dst_tmp.astype("uint8")
    return fusion_img




if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    path = "../text/"
    filename = "multi_img_inform.yaml"
    arg_list = fit.loadYaml(path + filename)
    img_input_path = arg_list["InputPath"]
    img_output_path = arg_list["OutputPath"]
    img_name_list = arg_list["img_names"]
    time_list = arg_list["times"]
    img_list = []
    for i in xrange(len(img_name_list)):
        img = cv2.imread(img_input_path+img_name_list[i])
        img_list.append(img)

    response_256x1x3 = GB_Calibrate(img_list, time_list)
    # 3.2.0 spend time 367ms
    hdr_img = GB_mergeHDR(img_list, time_list, response_256x1x3)
    ldr_img = GB_ToneMapping(hdr_img)
    cv2.imwrite(img_output_path+"Gtonemap.png", ldr_img)
    cv2.imshow("window", ldr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #spend time 491ms
    fusion_img = GB_Fusion(img_list)
    cv2.imwrite(img_output_path+"Gexpfusion.png", fusion_img)
    cv2.namedWindow("window")
    cv2.imshow("window", fusion_img)
    cv2.waitKey(0)
    cv2.destroyWindow("window")

    cProfile.run('re.compile("GB_Algorithm")', 'stats')
    pr.disable()
    pr.print_stats(sort='time')
    pass
















