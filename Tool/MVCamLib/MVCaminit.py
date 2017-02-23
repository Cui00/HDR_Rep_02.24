from MVCam_py import MVCam
import cv2
import numpy as np
import time

if __name__ == '__main__':
    num = 0
    while True:
        print num
        hCam = 0
        num += 1
        MV = MVCam()
        # print 'aaaaa'
        MV.EnumerateDevice()
        print MV.getMVNum()
        hCam = MV.Init('Cam5')
        hCam1 = MV.Init('cam2')
        print 'Init', hCam
        print 'Init', hCam1
        # print 'Stop', MV.Stop(hCam)
        print 'UnInit', MV.UnInit(hCam)
        print 'UnInit1', MV.UnInit(hCam1)
        time.sleep(1)
        del MV