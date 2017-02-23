from MVCam_py import MVCam
import cv2
import time
import numpy as np

if __name__ == "__main__":
    # MVCamE = {}
    # for i in (0,1):
    #     print 'i',i
    #     MVCamE[i] = MVCam()
    #     print 'EnumerateDevice', MVCamE[i].EnumerateDevice()
    # MVCam1 = MVCamE[0]
    # MVCam0 = MVCamE[1]

    MVCam1 = MVCam()
    print 'EnumerateDevice', MVCam1.EnumerateDevice()

    print 'Cam Init'
    Cam_list = ['Cam5','Cam1']
    cv2.namedWindow("MVCam0")
    # print 'open:',MVCam1.isOpen(Cam_list[0])
    # print MVCam1.getchar()
    hCam = MVCam1.Init(Cam_list[0])
    # hCam0 = MVCam0.Init(Cam_list[1])
    # print 'ConnectTest', MVCam1.ConnectTest(hCam)
    print 'hCam',hCam
    # print 'hCam0',hCam0
    print 'getMVNum',MVCam1.getMVNum()
    print 'SetAeState', MVCam1.SetAeState(hCam,False)
    MVCam1.SetExposureTime(hCam,60000)
    print 'SetAnalogGain', MVCam1.SetAnalogGain(hCam,1)
    print 'GetAnalogGain', MVCam1.GetAnalogGain(hCam)
    # MVCam1.SetOnceWB(hCam)
    print 'SetImageResolution', MVCam1.SetImageResolution(hCam,1280,720)
    print 'GetImageResolutionHeight', MVCam1.GetImageResolutionHeight(hCam)
    # print 'SetSharpness:', MVCam1.SetSharpness(hCam,50)
    # print 'GetSharpness:', MVCam1.GetSharpness(hCam)
    print 'SetLutMode:', MVCam1.SetLutMode(hCam,0)
    print 'GetLutMode:', MVCam1.GetLutMode(hCam)
    # print 'SetWbMode:', MVCam1.SetWbMode(hCam,True)
    # print 'GetWbMode:', MVCam1.GetWbMode(hCam)
    print 'GetClrTempMode:', MVCam1.GetClrTempMode(hCam)
    print 'SetClrTempMode:', MVCam1.SetClrTempMode(hCam,2)
    print 'SetUserClrTempGain:', MVCam1.SetUserClrTempGain(hCam,100,100,100)
    # print 'GetClrTempMode:', MVCam1.GetClrTempMode(hCam)
    # print 'SetPresetClrTemp:', MVCam1.SetPresetClrTemp(hCam,2)
    # print 'GetPresetClrTemp:', MVCam1.GetPresetClrTemp(hCam)
    print 'GetUserClrTempGain:', MVCam1.GetUserClrTempGain(hCam,2)
    # print 'SetAeTarget:', MVCam1.SetAeTarget(hCam,50)
    # print 'GetAeTarget:', MVCam1.GetAeTarget(hCam)
    # print 'SetPresetClrTemp:', MVCam1.SetPresetClrTemp(hCam,1)
    # print 'GetPresetClrTemp:', MVCam1.GetPresetClrTemp(hCam)
    # print 'SetGain', MVCam1.SetGain(hCam,200,100,256)
    # print 'GetGain', MVCam1.GetGain(hCam,1)
    print 'SetOnceWB', MVCam1.SetOnceWB(hCam)
    print 'SetGamma', MVCam1.SetGamma(hCam,50)
    print 'GetGamma', MVCam1.GetGamma(hCam)
    print 'SetContrast', MVCam1.SetContrast(hCam,150)
    print 'GetContrast', MVCam1.GetContrast(hCam)
    print 'SetSaturation', MVCam1.SetSaturation(hCam,100)
    print 'GetSaturation', MVCam1.GetSaturation(hCam)
    # print 'SetMonochrome', MVCam1.SetMonochrome(hCam,True)
    # print 'GetMonochrome', MVCam1.GetMonochrome(hCam)
    # print 'SetInverse', MVCam1.SetInverse(hCam,True)
    # print 'GetInverse', MVCam1.GetInverse(hCam)
    # print 'GetNoiseFilterState', MVCam1.GetNoiseFilterState(hCam)
    print 'SetNoiseFilter', MVCam1.SetNoiseFilter(hCam,False)
    print 'SetSharpness', MVCam1.SetSharpness(hCam,10)
    print 'GetSharpness', MVCam1.GetSharpness(hCam)
    print 'GetSharpness', MVCam1.GetSharpness(hCam)
    print 'GetNoiseFilterState', MVCam1.GetNoiseFilterState(hCam)
    print 'GetFriendlyName', MVCam1.GetFriendlyName(hCam)
    # print 'SetFriendlyName', MVCam1.SetFriendlyName(hCam,'Cam2')
    # print 'GetFriendlyName', MVCam1.GetFriendlyName(hCam)
    # print 'CameraReConnect', MVCam1.ReConnect(hCam)
    print 'ConnectTest', MVCam1.ConnectTest(hCam)

    print 'ReleaseImageBuffer', MVCam1.ReleaseImageBuffer(hCam)
    MVCam1.Play(hCam)
    # print 'getMVunuseNum',MVCam1.getMVunuseNum()
    # print 'SelectLutPreset:', MVCam1.SelectLutPreset(hCam,0)
    # print 'GetLutPresetSel:', MVCam1.GetLutPresetSel(hCam)

    # print 'open:',MVCam1.isOpen(Cam_list[0])
    # MVCam1.SetOnceWB(hCam)
    while True:
        st = time.time()
        while True:
            img0 = MVCam1.getImage(hCam,1000)
            if img0 != None:
                break
        ed = time.time()
        print ed - st
        if img0!=None:
            # print img0
            cv2.imshow("MVCam0", img0)
        key = chr(cv2.waitKey(30) & 255)
        if key in ['q', 'Q']:
            break
        elif key in ['p', 'P']:
            print 'Play', MVCam1.Play(hCam)
        elif key in ['z', 'Z']:
            print 'Pause', MVCam1.Pause(hCam)
            # print 'SetImageResolution', MVCam1.SetImageResolution(hCam,1280,960)
            # print 'GetImageResolutionHeight', MVCam1.GetImageResolutionHeight(hCam)
            print MVCam1.SetExposureTime(hCam,60000)
            print 'GetExposureTime', MVCam1.GetExposureTime(hCam)
            
        elif key in ['s', 'S']:
            print 'Stop', MVCam1.Stop(hCam)
            print MVCam1.SetExposureTime(hCam,10000)
            print 'GetExposureTime', MVCam1.GetExposureTime(hCam)
            MVCam1.SetOnceWB(hCam)
        elif key in ['a', 'A']:
            cv2.imwrite('./img1.png',img0)
        # elif key in ['x', 'X']:
            # print 'CameraReConnect', MVCam1.ReConnect(hCam)

