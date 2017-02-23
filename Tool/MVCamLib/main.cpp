#include "Camera.h" //相机SDK的API头文件

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;

//unsigned char           * g_pRgbBuffer;     //处理后数据缓存区

int main()
{
    std::cout<<"______________-"<<std::endl;

    Camera Camera1;
    //std::string name = "/home/pi/Desktop/MVImages/Image_0.bmp";
    std::string name = "/home/ljb/Desktop/camera_mindvison/CameraMV/images/1.BMP";
    Mat Iimag;
    int iDisplayFrames = 10000;

    //循环显示1000帧图像
    int i=0;
    cv::namedWindow("OpenCV Demo",cv::WINDOW_NORMAL);
    //Camera1.SetExposure(100000);
    while(iDisplayFrames--)
    {
        //Iimag = Camera1.CapImage();
        Iimag = cv::imread(name);
        std::cout<<"______________-"<<std::endl;

        //Iimag = cv::imread(name);
        cv::imshow("OpenCV Demo",Iimag);
        if ((cv::waitKey(30) & 255) == 'q'){break;}
    }

    return 0;
}

