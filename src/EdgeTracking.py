#__author__ = 'gimbu'
#coding:utf-8
import cv2
import numpy as np
import sys
sys.path.append('../Tool/')

from MVCamLib.MVCam_py import MVCam

def On_trackbar():
    new0 = cv2.getTrackbarPos('Down_Threshold', ouput_wd_name)
    new2 = cv2.getTrackbarPos('Up_Threshold', ouput_wd_name)

    edge_img = cv2.Canny(org_img, new0, new2)
    return edge_img

def On_trackbar2(pos):
    global hCam
    Cam0.SetExposureTime(hCam, pos)

def mergePct(src1, src2, src3):
    merge_img = np.zeros(src1.shape, dtype="uint8")
    merge_img = cv2.add(merge_img, src1, dtype=cv2.CV_8UC1)
    merge_img = cv2.add(merge_img, src2, dtype=cv2.CV_8UC1)
    merge_img = cv2.add(merge_img, src3, dtype=cv2.CV_8UC1)

    # merge_img = merge_img / 3
    return merge_img

if __name__ == '__main__':
    Cam0 = MVCam()
    CamName_list = ['cam4']
    if not Cam0.EnumerateDevice():
        print 'Cannot find any cameras!'
        exit(-1)
    hCam = Cam0.Init(CamName_list[0])
    Cam0.SetAeState(hCam, False)
    Cam0.SetImageResolution(hCam, 1280, 960)
    Cam0.Play(hCam)
    original_wd_name = "Cam4 Window"
    ouput_wd_name = "Edge Window"
    cv2.namedWindow(original_wd_name)
    cv2.namedWindow(ouput_wd_name)
    cv2.createTrackbar('Down_Threshold', ouput_wd_name, 50, 255, On_trackbar)
    cv2.createTrackbar('Up_Threshold', ouput_wd_name, 150, 255, On_trackbar)
    cv2.createTrackbar('Exposure', ouput_wd_name, 30000, 100000, On_trackbar2)
    count = 0
    count2 = 0
    src = []
    edge = []
    while True:
        org_img = Cam0.getImage(hCam, 1000)
        if org_img!=None:
            cv2.imshow(original_wd_name, org_img)
        org_img = cv2.GaussianBlur(org_img, (3, 3), 0)
        edge_img = On_trackbar()
        cv2.imshow(ouput_wd_name, edge_img)
        key = chr(cv2.waitKey(5) & 255)
        if key in ['c', 'C']:
            # cv2.imwrite("../res/WRZ/org_"+str(count)+"_.png", org_img)
            cv2.imwrite("../res/edge_img/org_"+str(count)+"_.png", org_img)
            cv2.imwrite("../res/edge_img/edge_"+str(count)+"_.png", edge_img)
            count += 1
        if key in ['m', 'M']:
            for i in range(3):
                src.append(cv2.imread("../res/edge_img/org_"+str(3*count2+i)+"_.png"))
                edge.append(cv2.imread("../res/edge_img/edge_"+str(3*count2+i)+"_.png"))
            merge_edge_img = mergePct(edge[3*count2+0], edge[3*count2+1], edge[3*count2+2])
            cv2.imwrite("../res/edge_img/merge_edge_"+str(count2)+"_.png", merge_edge_img)
            count2 += 1
            cv2.namedWindow("merge edge")
            cv2.imshow("merge edge", merge_edge_img)
            cv2.waitKey()
            cv2.destroyWindow("merge edge")
        if key in ['q', 'Q']:
            print '\nUnInit', Cam0.UnInit(hCam)
            break

