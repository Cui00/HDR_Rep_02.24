#from XiMeaCam_py import test
from MVCam_py import MVCam
import cv2
import numpy as np
import time


def expose(pos):
    global cam_list
    Newexpose = cv2.getTrackbarPos('expose', cam_list[0])
    # print pos
    print 'exposeset:', MVCam1.SetExposure(cam_list[0], Newexpose)

def colorTemp(pos):
    global cam_list
    Newred = cv2.getTrackbarPos('red', cam_list[0])
    Newgreen = cv2.getTrackbarPos('green', cam_list[0])
    Newblue = cv2.getTrackbarPos('blue', cam_list[0])
    Newsaturation = cv2.getTrackbarPos('saturation', cam_list[0])


    print 'colorTempset:', MVCam1.SetColorTemp(cam_list[0],Newred,Newgreen,Newblue,Newsaturation)

def gammacontrast(pos):
    global cam_list
    Newgamma = cv2.getTrackbarPos('gamma', cam_list[0])
    Newcontrast = cv2.getTrackbarPos('contrast', cam_list[0])
    # print pos
    print 'gammacontrast:', MVCam1.SetContrast(cam_list[0], Newgamma,Newcontrast)

FastPhotoModel = ''
MVCam1 = MVCam(FastPhotoModel,3)
# cv2.namedWindow("MVCam1", cv2.WINDOW_NORMAL)
cam_list = ['Cam5','Cam7']
i = 0
cv2.namedWindow(cam_list[0], cv2.WINDOW_NORMAL)
MVCam1.releaseImageBuffer(cam_list[0])
MVCam1.releaseImageBuffer(cam_list[1])
flag = MVCam1.openCam(cam_list[0])
flag = MVCam1.openCam(cam_list[1])

print 'start'
cv2.createTrackbar('expose', cam_list[0], 30000, 100000, expose)
cv2.createTrackbar('red', cam_list[0], 100, 399, colorTemp)
cv2.createTrackbar('green', cam_list[0], 100, 399, colorTemp)
cv2.createTrackbar('blue', cam_list[0], 100, 399, colorTemp)
cv2.createTrackbar('saturation', cam_list[0], 100, 200, colorTemp)
cv2.createTrackbar('gamma', cam_list[0], 100, 100, gammacontrast)
cv2.createTrackbar('contrast', cam_list[0], 100, 200, gammacontrast)
NowExpose = 0
PreExpose = 0
while True:
    time1 = time.time()
    img0 = MVCam1.takePhoto(cam_list[0])
    img1 = MVCam1.takePhoto(cam_list[1])
    # print "time:", time.time() - time1
    # time.sleep(0.05)
    # if img!=None:
    #     for i in range(20):
    #         img = MVCam1.takePhoto()
    #         if img!=None:
    #             # //cv2.imwrite('./images/'+str(i)+'.bmp', img)
    #             print './images/'+str(i)+'.bmp'
    # img = np.zeros((100,100),dtype=np.uint8)
    # print 'shape of img : ' , img.shape #one channel!!

    #print img
    # NowExpose = cv2.getTrackbarPos('expose', 'MVCam1')
    # if PreExpose != NowExpose:
    #     print 'NowExpose:', NowExpose
    #     print 'set:', MVCam1.SetExposure('Cam1',NowExpose)
    #     PreExpose = NowExpose
    if img0!=None:
        cv2.imshow("MVCam0", img0)
    else:
        print "Cam0 wrong"
    if img1!=None:
        cv2.imshow("MVCam1", img1)
    else:
        print "Cam1 wrong"
        # cv2.imwrite("img.png", img)qq
        # time.sleep(0.5)
    key = chr(cv2.waitKey(30) & 255)
    # del MVCam1
    if key in ['q', 'Q']:
        break
    elif key in ['p', 'P']:
        flag = MVCam1.stopCam(cam_list[0])
        print 'stop:', flag
    #     i = 0 if i == 1 else 1
    elif key in ['s', 'S']:
        flag = MVCam1.openCam(cam_list[0])
        print 'open:', flag
    elif key in ['a', 'A']:
        print 'save0:', cv2.imwrite('0.png', img0)
        print 'save1:', cv2.imwrite('1.png', img1)