__author__ = 'gimbu'
#coding:utf-8
import sys
sys.path.append('../Tool/')
sys.path.append('../')
import cv2
import numpy as np
import glob
import cProfile
import re
import matplotlib.pyplot as plt
from ImgProcessTool import image_process_tool as ipt
from FileInterfaceTool import FileInterfaceTool as fit


Debevec = 0000
Robertson = 0001
Mertens = 1000

def HDR(_imgs_nx1, _times_nx1, method=Debevec):
    assert _imgs_nx1.dtype == np.uint8 and _times_nx1.dtype == np.float32, "Type Error"
    assert len(_imgs_nx1) == len(_times_nx1) and len(_times_nx1) > 0, "Len Error"
    if method == Debevec:
        CalibrateDebevec = cv2.createCalibrateDebevec(samples=70, random=True)
        crf = CalibrateDebevec.process(src=_imgs_nx1, times=_times_nx1)
        merge_debvec = cv2.createMergeDebevec()
        hdr_img = merge_debvec.process(src=_imgs_nx1,  times=_times_nx1, response=crf)
        tonemap = cv2.createTonemapDurand(gamma=1.4)
        res_img = tonemap.process(hdr_img.copy())
        return crf, hdr_img, res_img
    if method == Robertson:
        CalibrateRobertson = cv2.createCalibrateRobertson()
        crf = CalibrateRobertson.process(src=_imgs_nx1, times=_times_nx1)
        merge_robertson = cv2.createMergeRobertson()
        hdr_img = merge_robertson.process(src=_imgs_nx1,  times=_times_nx1, response=crf)
        #local tonermap
        tonemap = cv2.createTonemapDurand(gamma=1.4)
        res_img = tonemap.process(hdr_img.copy())
        return crf, hdr_img, res_img
    if method == Mertens:
        merge_mertens = cv2.createMergeMertens()
        res_img = merge_mertens.process(_imgs_nx1)
        # cv2.imshow("ss", res_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # res_mertens_8bit = np.clip(res_img*255, 0, 255).astype('uint8')
        # cv2.imwrite("PyFusion.png", res_mertens_8bit)
        return res_img

def showData(crf, res_img):
    #draw response curve
    trans_crf = np.transpose(crf)
    plt.figure(1) # 创建图表1
    ax1 = plt.subplot(311) # 在图表1中创建子图1
    ax2 = plt.subplot(312) # 在图表1中创建子图2
    ax3 = plt.subplot(313) # 在图表1中创建子图3
    #x, f(x) both np.array
    x = np.array(xrange(256))
    plt.sca(ax1) # 选择图表1的子图1,trans_crf[0][0] is the same 1-d array as x.
    plt.plot(x, trans_crf[0][0], color="b")
    plt.sca(ax2)  # 选择图表1的子图2
    plt.plot(x, trans_crf[1][0], color="g")
    plt.sca(ax3)  # 选择图表1的子图3
    plt.plot(x, trans_crf[2][0], color="r")
    plt.show()
    plt.close()

    res_debvec_8bit = np.clip(res_img*255, 0, 255).astype('uint8')
    cv2.imshow("res", res_debvec_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #ready
    yaml_path = "../other/"
    filename = "HdrConfig.yaml"
    arg_list = fit.loadYaml(yaml_path + filename)
    img_path = arg_list["InputPath"]
    img_name_list = arg_list["img_names"]
    img_list = [cv2.imread(img_path+img_name)
                for img_name in img_name_list]
    imgs_nx1 = np.array(img_list)
    time_list = arg_list["times"]
    times_nx1 = np.array(time_list, dtype=np.float32)

    pr = cProfile.Profile()
    pr.enable()

    # core, process spend time 225ms, calibrate spend time 2s.
    crf, hdr_img, res_img = HDR(imgs_nx1, times_nx1, method=Debevec)
    showData(crf, res_img)

    cProfile.run('re.compile("GB_HDR")', 'stats')
    pr.disable()
    pr.print_stats(sort='time')
    pass


