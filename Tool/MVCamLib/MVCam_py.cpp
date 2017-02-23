#include <iostream>
#include <string.h>
#include "Camera.h"
#include "conversion.h"
#include <boost/python.hpp>
#include  "opencv2/opencv.hpp"

using namespace boost::python ;

class MVCam
{
    public:
    Camera * cam;
    MVCam()
    {
        std::cout<<"Pi "<<std::endl;
        this->cam = new Camera();
        //this->cam->TrigMode(0);
        // std::cout<<"end to initialize  !!"<<std::endl;
    }

    ~MVCam()
    {
        //this->cam->CloseCam();
        delete this->cam;
        std::cout<<"end to close Camera  !!"<<std::endl;
    }
    
    bool EnumerateDevice()
    {
        return this->cam->EnumerateDevice();
    }    
    std::string Getname()
    {
        return this->cam->Getname();
    }

    bool SdkInit(int iLanguageSel)
    {
        return this->cam->SdkInit(iLanguageSel);
    }
    int Init(char* Cam)
    {
        return this->cam->Init(Cam);
    }
    int getMVNum()
    {
        return this->cam->getMVNum();
    }
    int getMVunuseNum()
    {
        return this->cam->getMVunuseNum();
    }

    bool isOpen(char* Cam)
    {
        return this->cam->isOpen(Cam);
    }
    bool UnInit(int hCam)
    {
        return this->cam->UnInit(hCam);
    }
    PyObject* getImage(int hCam, UINT wTimes_ms)
    {
        cv::Mat frame;
        NDArrayConverter cvt_;
        frame = this->cam->getImage(hCam, wTimes_ms);
        PyObject * imgNd;
        imgNd = cvt_.toNDArray(frame);
        return  imgNd;
    }
    bool ReleaseImageBuffer(int hCam)
    {
        return this->cam->ReleaseImageBuffer(hCam);
    }
    bool Play(int hCam)
    {
        return this->cam->Play(hCam);
    }
    bool Pause(int hCam)
    {
        return this->cam->Pause(hCam);
    }
    bool Stop(int hCam)
    {
        return this->cam->Stop(hCam);
    }
    int GetImageResolutionWidth(int hCam)
    {
        return this->cam->GetImageResolutionWidth(hCam);
    }
    int GetImageResolutionHeight(int hCam)
    {
        return this->cam->GetImageResolutionHeight(hCam);
    }
    bool SetImageResolution(int hCam,int Width,int Height)
    {
        return this->cam->SetImageResolution(hCam,Width,Height);
    }
    bool SetOffsetFOV(int hCam,int HOffsetFOV,int VOffsetFOV)
    {
        return this->cam->SetOffsetFOV(hCam,HOffsetFOV,VOffsetFOV);
    }
    bool SetAeState(int hCam, bool State)
    {
        return this->cam->SetAeState(hCam, State);
    }
    int GetAeState(int hCam)
    {
        return this->cam->GetAeState(hCam);
    }
    bool SetSharpness(int hCam,int iSharpness)
    {
        return this->cam->SetSharpness(hCam,iSharpness);
    }
    int GetSharpness(int hCam)
    {
        return this->cam->GetSharpness(hCam);
    }
    bool SetLutMode(int hCam,int emLutMode)
    {
        return this->cam->SetLutMode(hCam,emLutMode);
    }
    int GetLutMode(int hCam)
    {
        return this->cam->GetLutMode(hCam);
    }
    bool SelectLutPreset(int hCam,int iSel)
    {
        return this->cam->SelectLutPreset(hCam,iSel);
    }
    int GetLutPresetSel(int hCam)
    {
        return this->cam->GetLutPresetSel(hCam);
    }
    bool SetWbMode(int hCam, bool bAuto)
    {
        return this->cam->SetWbMode(hCam, bAuto);
    }
    int GetWbMode(int hCam)
    {
        return this->cam->GetWbMode(hCam);
    }
    bool SetPresetClrTemp(int hCam,int iSel)
    {
        return this->cam->SetPresetClrTemp(hCam,iSel);
    }
    int GetPresetClrTemp(int hCam)
    {
        return this->cam->GetPresetClrTemp(hCam);
    }
    bool SetUserClrTempGain(int hCam,int iRgain,int iGgain,int iBgain)
    {
        return this->cam->SetUserClrTempGain(hCam,iRgain,iGgain,iBgain);
    }
    int GetUserClrTempGain(int hCam,int Colorsel)
    {
        return this->cam->GetUserClrTempGain(hCam,Colorsel);
    }
    bool SetClrTempMode(int hCam,int iMode)
    {
        return this->cam->SetClrTempMode(hCam,iMode);
    }
    int GetClrTempMode(int hCam)
    {
        return this->cam->GetClrTempMode(hCam);
    }
    bool SetOnceWB(int hCam)
    {
        return this->cam->SetOnceWB(hCam);
    }
    bool SetAeTarget(int hCam,int iAeTarget)
    {
        return this->cam->SetAeTarget(hCam,iAeTarget);
    }
    int GetAeTarget(int hCam)
    {
        return this->cam->GetAeTarget(hCam);
    }
    bool SetExposureTime(int hCam,int fExposureTime)
    {
        return this->cam->SetExposureTime(hCam,fExposureTime);
    }
    int GetExposureTime(int hCam)
    {
        return this->cam->GetExposureTime(hCam);
    }
    bool SetAnalogGain(int hCam,int iAnalogGain)
    {
        return this->cam->SetAnalogGain(hCam,iAnalogGain);
    }
    int GetAnalogGain(int hCam)
    {
        return this->cam->GetAnalogGain(hCam);
    }
    bool SetGain(int hCam,int iRgain,int iGgain,int iBgain)
    {
        return this->cam->SetGain(hCam,iRgain,iGgain,iBgain);
    }
    int GetGain(int hCam,int Colorsel)
    {
        return this->cam->GetGain(hCam,Colorsel);
    }
    bool SetGamma(int hCam,int iGamma)
    {
        return this->cam->SetGamma(hCam,iGamma);
    }
    int GetGamma(int hCam)
    {
        return this->cam->GetGamma(hCam);
    }
    bool SetContrast(int hCam,int iContrast)
    {
        return this->cam->SetContrast(hCam,iContrast);
    }
    int GetContrast(int hCam)
    {
        return this->cam->GetContrast(hCam);
    }
    bool SetSaturation(int hCam,int iSaturation)
    {
        return this->cam->SetSaturation(hCam,iSaturation);
    }
    int GetSaturation(int hCam)
    {
        return this->cam->GetSaturation(hCam);
    }
    // std::string getchar()
    // {
    //     return this->cam->getchar();
    // }
    bool SetMonochrome(int hCam,bool pbEnable)
    {
        return this->cam->SetMonochrome( hCam, pbEnable);
    }
    int GetMonochrome(int hCam)
    {
        return this->cam->GetMonochrome( hCam);
    }
    bool SetInverse(int hCam,bool pbEnable)
    {
        return this->cam->SetInverse( hCam, pbEnable);
    }
    int GetInverse(int hCam)
    {
        return this->cam->GetInverse( hCam);
    }
    bool SetAntiFlick(int hCam,bool pbEnable)
    {
        return this->cam->SetAntiFlick( hCam, pbEnable);
    }
    int GetAntiFlick(int hCam)
    {
        return this->cam->GetAntiFlick( hCam);
    }
    bool SetLightFrequency(int hCam,int iFrequencySel)
    {
        return this->cam->SetLightFrequency( hCam, iFrequencySel);
    }
    int GetLightFrequency(int hCam)
    {
        return this->cam->GetLightFrequency( hCam);
    }
    bool SetFrameSpeed(int hCam,int piFrameSpeed)
    {
        return this->cam->SetFrameSpeed( hCam, piFrameSpeed);
    }
    int GetFrameSpeed(int hCam)
    {
        return this->cam->GetFrameSpeed( hCam);
    }
    // int GetFrameSpeed(int hCam)
    // {
    //     return this->cam->GetFrameSpeed( hCam);
    // }
    bool SaveParameterToFile(int hCam,char* sFileName)
    {
        return this->cam->SaveParameterToFile( hCam, sFileName);
    }
    bool ReadParameterFromFile(int hCam,char* sFileName)
    {
        return this->cam->ReadParameterFromFile( hCam,sFileName);
    }
    bool WriteSN(int hCam,BYTE* pbySN, int iLevel)
    {
        return this->cam->WriteSN( hCam, pbySN,  iLevel);
    }
    BYTE ReadSN(int hCam,int iLevel)
    {
        return this->cam->ReadSN( hCam, iLevel);
    }
    bool SetTriggerDelayTime(int hCam,UINT uDelayTimeUs)
    {
        return this->cam->SetTriggerDelayTime( hCam, uDelayTimeUs);
    }
    UINT GetTriggerDelayTime(int hCam)
    {
        return this->cam->GetTriggerDelayTime( hCam);
    }
    bool SetTriggerCount(int hCam,INT iCount)
    {
        return this->cam->SetTriggerCount( hCam, iCount);
    }
    int GetTriggerCount(int hCam)
    {
        return this->cam->GetTriggerCount( hCam);
    }
    int SoftTrigger(int hCam)
    {
        return this->cam->SoftTrigger( hCam);
    }

    bool SetTriggerMode(int hCam,INT iModeSel)
    {
        return this->cam->SetTriggerMode( hCam, iModeSel);
    }
    int GetTriggerMode(int hCam)
    {
        return this->cam->GetTriggerMode( hCam);
    }
    bool SetStrobeMode(int hCam,INT iMode)
    {
        return this->cam->SetStrobeMode( hCam,iMode);
    }
    int GetStrobeMode(int hCam)
    {
        return this->cam->GetStrobeMode( hCam);
    }
    bool SetStrobeDelayTime(int hCam,UINT uDelayTimeUs)
    {
        return this->cam->SetStrobeDelayTime( hCam,uDelayTimeUs);
    }
    int GetStrobeDelayTime(int hCam)
    {
        return this->cam->GetStrobeDelayTime( hCam);
    }
    bool SetStrobePulseWidth(int hCam,UINT uDelayTimeUs)
    {
        return this->cam->SetStrobePulseWidth( hCam, uDelayTimeUs);
    }
    int GetStrobePulseWidth(int hCam)
    {
        return this->cam->GetStrobePulseWidth( hCam);
    }
    bool SetStrobePolarity(int hCam,INT uPolarity)
    {
        return this->cam->SetStrobePolarity( hCam, uPolarity);
    }
    int GetStrobePolarity(int hCam)
    {
        return this->cam->GetStrobePolarity( hCam);
    }
    bool SetExtTrigSignalType(int hCam,INT iType)
    {
        return this->cam->SetExtTrigSignalType( hCam, iType);
    }
    int GetExtTrigSignalType(int hCam)
    {
        return this->cam->GetExtTrigSignalType( hCam);
    }
    bool SetExtTrigShutterType(int hCam,INT iType)
    {
        return this->cam->SetExtTrigShutterType( hCam, iType);
    }
    int GetExtTrigShutterType(int hCam)
    {
        return this->cam->GetExtTrigShutterType( hCam);
    }
    bool SetExtTrigDelayTime(int hCam,UINT uDelayTimeUs)
    {
        return this->cam->SetExtTrigDelayTime( hCam, uDelayTimeUs);
    }
    int GetExtTrigDelayTime(int hCam)
    {
        return this->cam->GetExtTrigDelayTime( hCam);
    }
    bool SetExtTrigJitterTime(int hCam,UINT uDelayTimeUs)
    {
        return this->cam->SetExtTrigJitterTime( hCam, uDelayTimeUs);
    }
    UINT GetExtTrigJitterTime(int hCam)
    {
        return this->cam->GetExtTrigJitterTime( hCam);
    }
    UINT GetExtTrigCapability(int hCam)
    {
        return this->cam->GetExtTrigCapability( hCam);
    }
    // bool SetResolutionForSnap(int hCam,tSdkImageResolution* pImageResolution)
    // {
    //     return this->cam->SetResolutionForSnap( hCam, pImageResolution);
    // }
    bool SetNoiseFilter(int hCam,BOOL bEnable)
    {
        return this->cam->SetNoiseFilter( hCam, bEnable);
    }
    int GetNoiseFilterState(int hCam)
    {
        return this->cam->GetNoiseFilterState( hCam);
    }
    bool SetFriendlyName(int hCam,char* pName)
    {
        return this->cam->SetFriendlyName( hCam, pName);
    }
    std::string GetFriendlyName(int hCam)
    {
        return this->cam->GetFriendlyName( hCam);
    }
    bool ReConnect(int hCam)
    {
        return this->cam->ReConnect( hCam);
    }
    bool ConnectTest(int hCam)
    {
        return this->cam->ConnectTest( hCam);
    }
};



BOOST_PYTHON_MODULE(MVCam_py)
{
    class_< MVCam >("MVCam", init<>())
    .def("EnumerateDevice",  &MVCam::EnumerateDevice)
    .def("Getname",  &MVCam::Getname)
    .def("SdkInit",  &MVCam::SdkInit)
    .def("isOpen",  &MVCam::isOpen)
    .def("Init",  &MVCam::Init)
    .def("getMVNum",  &MVCam::getMVNum)
    .def("getMVunuseNum",  &MVCam::getMVunuseNum)
    .def("UnInit",  &MVCam::UnInit)
    .def("getImage",  &MVCam::getImage)
    .def("ReleaseImageBuffer",  &MVCam::ReleaseImageBuffer)
    .def("Play",  &MVCam::Play)
    .def("Pause",  &MVCam::Pause)
    .def("Stop",  &MVCam::Stop)
    .def("GetImageResolutionWidth",  &MVCam::GetImageResolutionWidth)
    .def("GetImageResolutionHeight",  &MVCam::GetImageResolutionHeight)
    .def("SetImageResolution",  &MVCam::SetImageResolution)
    .def("SetOffsetFOV",  &MVCam::SetOffsetFOV)
    .def("SetAeState" , &MVCam::SetAeState)
    .def("GetAeState" , &MVCam::GetAeState)
    .def("SetSharpness" , &MVCam::SetSharpness)
    .def("GetSharpness" , &MVCam::GetSharpness)
    .def("SetLutMode" , &MVCam::SetLutMode)
    .def("GetLutMode" , &MVCam::GetLutMode)
    .def("SelectLutPreset" , &MVCam::SelectLutPreset)
    .def("GetLutPresetSel" , &MVCam::GetLutPresetSel)
    .def("SetWbMode" , &MVCam::SetWbMode)
    .def("GetWbMode" , &MVCam::GetWbMode)
    .def("SetPresetClrTemp" , &MVCam::SetPresetClrTemp)
    .def("GetPresetClrTemp" , &MVCam::GetPresetClrTemp)
    .def("SetUserClrTempGain" , &MVCam::SetUserClrTempGain)
    .def("GetUserClrTempGain" , &MVCam::GetUserClrTempGain)
    .def("SetClrTempMode" , &MVCam::SetClrTempMode)
    .def("GetClrTempMode" , &MVCam::GetClrTempMode)
    .def("SetOnceWB" , &MVCam::SetOnceWB)
    .def("SetAeTarget" , &MVCam::SetAeTarget)
    .def("GetAeTarget" , &MVCam::GetAeTarget)
    .def("SetExposureTime" , &MVCam::SetExposureTime)
    .def("GetExposureTime" , &MVCam::GetExposureTime)
    .def("SetAnalogGain" , &MVCam::SetAnalogGain)
    .def("GetAnalogGain" , &MVCam::GetAnalogGain)
    .def("SetGain" , &MVCam::SetGain)
    .def("GetGain" , &MVCam::GetGain)
    .def("SetGamma" , &MVCam::SetGamma)
    .def("GetGamma" , &MVCam::GetGamma)
    .def("SetContrast" , &MVCam::SetContrast)
    .def("GetContrast" , &MVCam::GetContrast)
    .def("SetSaturation" , &MVCam::SetSaturation)
    .def("GetSaturation" , &MVCam::GetSaturation)
    // .def("getchar" , &MVCam::getchar);
    .def("SetMonochrome" , &MVCam::SetMonochrome)
    .def("GetMonochrome" , &MVCam::GetMonochrome)
    .def("SetInverse" , &MVCam::SetInverse)
    .def("GetInverse" , &MVCam::GetInverse)
    .def("SetAntiFlick" , &MVCam::SetAntiFlick)
    .def("GetAntiFlick" , &MVCam::GetAntiFlick)
    .def("SetLightFrequency" , &MVCam::SetLightFrequency)
    .def("GetLightFrequency" , &MVCam::GetLightFrequency)
    .def("SetFrameSpeed" , &MVCam::SetFrameSpeed)
    .def("GetFrameSpeed" , &MVCam::GetFrameSpeed)
    .def("SaveParameterToFile" , &MVCam::SaveParameterToFile)
    .def("ReadParameterFromFile" , &MVCam::ReadParameterFromFile)
    .def("WriteSN" , &MVCam::WriteSN)
    .def("ReadSN" , &MVCam::ReadSN)
    .def("SetTriggerDelayTime" , &MVCam::SetTriggerDelayTime)
    .def("GetTriggerDelayTime" , &MVCam::GetTriggerDelayTime)
    .def("SetTriggerCount" , &MVCam::SetTriggerCount)
    .def("GetTriggerCount" , &MVCam::GetTriggerCount)
    .def("SoftTrigger" , &MVCam::SoftTrigger)
    .def("SetTriggerMode" , &MVCam::SetTriggerMode)
    .def("GetTriggerMode" , &MVCam::GetTriggerMode)
    .def("SetStrobeMode" , &MVCam::SetStrobeMode)
    .def("GetStrobeMode" , &MVCam::GetStrobeMode)
    .def("SetStrobeDelayTime" , &MVCam::SetStrobeDelayTime)
    .def("GetStrobeDelayTime" , &MVCam::GetStrobeDelayTime)
    .def("SetStrobePulseWidth" , &MVCam::SetStrobePulseWidth)
    .def("GetStrobePulseWidth" , &MVCam::GetStrobePulseWidth)
    .def("SetStrobePolarity" , &MVCam::SetStrobePolarity)
    .def("GetStrobePolarity" , &MVCam::GetStrobePolarity)
    .def("SetExtTrigSignalType" , &MVCam::SetExtTrigSignalType)
    .def("GetExtTrigSignalType" , &MVCam::GetExtTrigSignalType)
    .def("SetExtTrigShutterType" , &MVCam::SetExtTrigShutterType)
    .def("GetExtTrigShutterType" , &MVCam::GetExtTrigShutterType)
    .def("SetExtTrigDelayTime" , &MVCam::SetExtTrigDelayTime)
    .def("GetExtTrigDelayTime" , &MVCam::GetExtTrigDelayTime)
    .def("SetExtTrigJitterTime" , &MVCam::SetExtTrigJitterTime)
    .def("GetExtTrigJitterTime" , &MVCam::GetExtTrigJitterTime)
    .def("GetExtTrigCapability" , &MVCam::GetExtTrigCapability)
    // .def("SetResolutionForSnap" , &MVCam::SetResolutionForSnap)
    .def("SetNoiseFilter" , &MVCam::SetNoiseFilter)
    .def("GetNoiseFilterState" , &MVCam::GetNoiseFilterState)
    .def("SetFriendlyName" , &MVCam::SetFriendlyName)
    .def("GetFriendlyName" , &MVCam::GetFriendlyName)
    .def("ReConnect" , &MVCam::ReConnect)
    .def("ConnectTest" , &MVCam::ConnectTest);
    // .def("GetExtTrigJitterTime" , &MVCam::GetExtTrigJitterTime)
    // .def("GetExtTrigJitterTime" , &MVCam::GetExtTrigJitterTime)
    // .def("GetExtTrigJitterTime" , &MVCam::GetExtTrigJitterTime)
}