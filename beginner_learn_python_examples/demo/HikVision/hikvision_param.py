# -*- coding: utf-8 -*-
"""
 Created on 2021/6/18 22:27
 Filename   : hikvision_param.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
from ctypes import *


# 登录参数
class NET_DVR_USER_LOGIN_INFO(Structure):
    _fields_ = [
        ("sDeviceAddress", c_byte * 129),  # 设备 IP
        ("byUseTransport", c_byte),
        ("wPort", c_uint16),               # 设备端口号
        ("sUserName", c_byte * 64),        # 登录用户名
        ("sPassword", c_byte * 64),        # 登录密码
        # ("fLoginResultCallBack",)
        ("pUser", c_void_p),
        ("bUseAsynLogin", c_bool),
        ("byProxyType", c_byte),
        ("byUseUTCTime", c_byte),
        ("byLoginMode", c_byte),
        ("byHttps", c_byte),
        ("iProxyID", c_long),
        ("byVerifyMode", c_byte),
        ("byRes3", c_byte * 119),
    ]


# 设备参数
class NET_DVR_DEVICEINFO_V30(Structure):
    _fields_ = [
        ("sSerialNumber", c_byte * 48),    # 序列号
        ("byAlarmInPortNum", c_byte),
        ("byAlarmOutPortNum", c_byte),
        ("byDiskNum", c_byte),
        ("byDVRType", c_byte),             # 设备类型
        ("byChanNum", c_byte),
        ("byStartChan", c_byte),
        ("byAudioChanNum", c_byte),
        ("byIPChanNum", c_byte),
        ("byZeroChanNum", c_byte),
        ("byMainProto", c_byte),
        ("bySubProto", c_byte),
        ("bySupport", c_byte),
        ("bySupport1", c_byte),
        ("bySupport2", c_byte),
        ("wDevType", c_uint16),            # 设备型号
        ("bySupport3", c_byte),
        ("byMultiStreamProto", c_byte),
        ("byStartDChan", c_byte),
        ("byStartDTalkChan", c_byte),
        ("byHighDChanNum", c_byte),
        ("bySupport4", c_byte),
        ("byLanguageType", c_byte),
        ("byVoiceInChanNum", c_byte),
        ("byStartVoiceInChanNo", c_byte),
        ("bySupport5", c_byte),
        ("bySupport6", c_byte),
        ("byMirrorChanNum", c_byte),
        ("wStartMirrorChanNo", c_uint16),
        ("bySupport7", c_byte),
        ("byRes2", c_byte)]


# 设备参数
class NET_DVR_DEVICEINFO_V40(Structure):
    _fields_ = [
        ("struDeviceV30", NET_DVR_DEVICEINFO_V30),
        ("bySupportLock", c_byte),
        ("byRetryLoginTime", c_byte),
        ("byPasswordLevel", c_byte),
        ("byProxyType", c_byte),
        ("dwSurplusLockTime", c_ulong),
        ("byCharEncodeType", c_byte),
        ("bySupportDev5", c_byte),
        ("bySupport", c_byte),
        ("byLoginMode", c_byte),
        ("dwOEMCode", c_ulong),
        ("iResidualValidity", c_int),
        ("byResidualValidity", c_byte),
        ("bySingleStartDTalkChan", c_byte),
        ("bySingleDTalkChanNums", c_byte),
        ("byPassWordResetLevel", c_byte),
        ("bySupportStreamEncrypt", c_byte),
        ("byMarketType", c_byte),
        ("byRes2", c_byte * 238),
    ]


# 布防
class NET_DVR_SETUPALARM_PARAM(Structure):
    _fields_ = [
        ("dwSize", c_ulong),
        ("beLevel", c_byte),
        ("byAlarmInfoType", c_byte),
        ("byRetAlarmTypeV40", c_byte),
        ("byRetDevInfoVersion", c_byte),
        ("byRetVQDAlarmType", c_byte),
        ("byFaceAlarmDetection", c_byte),
        ("bySupport", c_byte),
        ("byBrokenNetHttp", c_byte),
        ("wTaskNo", c_uint16),
        ("byDeployType", c_byte),
        ("bySubScription", c_byte),
        ("byRes1", c_byte * 2),
        ("byAlarmTypeURL", c_byte),
        ("byCustomCtrl", c_byte)
    ]


# SDK 功能信息
class NET_DVR_SDKABL(Structure):
    _fields_ = [
        ("dwMaxLoginNum", c_ulong),
        ("dwMaxRealPlayNum", c_ulong),
        ("dwMaxPlayBackNum", c_ulong),
        ("dwMaxAlarmChanNum", c_ulong),
        ("dwMaxFormatNum", c_ulong),
        ("dwMaxFileSearchNum", c_ulong),
        ("dwMaxLogSearchNum", c_ulong),
        ("dwMaxSerialNum", c_ulong),
        ("dwMaxUpgradeNum", c_ulong),
        ("dwMaxVoiceComNum", c_ulong),
        ("dwMaxBroadCastNum", c_ulong),
        ("dwRes", c_ulong * 10),
    ]


# 车牌位置
class NET_VCA_RECT(Structure):
    _fields_ = [
        ("fX", c_float),       # 边界框左上角点的X轴坐标, 0.000~1
        ("fY", c_float),       # 边界框左上角点的Y轴坐标, 0.000~1
        ("fWidth", c_float),   # 边界框的宽度, 0.000~1
        ("fHeight", c_float),  # 边界框的高度, 0.000~1
    ]


# 车牌识别结果子结构
class NET_DVR_PLATE_INFO(Structure):
    _fields_ = [
        ("byPlateType", c_byte),              # 车牌类型
        ("byColor", c_byte),                  # 车牌颜色
        ("byBright", c_byte),
        ("byLicenseLen", c_byte),
        ("byEntireBelieve", c_byte),
        ("byRegion", c_byte),
        ("byCountry", c_byte),
        ("byArea", c_byte),
        ("byPlateSize", c_byte),
        ("byAddInfoFlag", c_byte),
        ("wCRIndex", c_uint16),
        ("byRes", c_byte * 4),
        ("pAddInfoBuffer", POINTER(c_byte)),
        ("byRes2", c_byte * 4),
        ("byRes2", c_char * 8),
        ("dwXmlLen", c_ulong),
        ("pXmlBuf", POINTER(c_char)),
        ("struPlateRect", NET_VCA_RECT),      # 车牌位置
        ("sLicense", c_char * 8),             # 车牌号码
        ("byBelieve", c_byte * 8),
    ]


# 车辆信息
class NET_DVR_VEHICLE_INFO(Structure):
    _fields_ = [
        ("dwIndex", c_ulong),
        ("byVehicleType", c_byte),              # 车辆类型
        ("byColorDepth", c_byte),
        ("byColor", c_byte),                    # 车身颜色
        ("byRadarState", c_byte),
        ("wSpeed", c_uint16),
        ("wLength", c_uint16),
        ("byIllegalType", c_byte),
        ("byVehicleLogoRecog", c_byte),
        ("byVehicleSubLogoRecog", c_byte),
        ("byVehicleModel", c_byte),
        ("byCustomInfo", c_byte * 16),
        ("wVehicleLogoRecog", c_uint16),
        ("byIsParking", c_byte),
        ("byRes", c_byte),
        ("dwParkingTime", c_ulong),
        ("byBelieve", c_byte),
        ("byCurrentWorkerNumber", c_byte),
        ("byCurrentGoodsLoadingRate", c_byte),
        ("byDoorsStatus", c_byte),
        ("byRes3", c_byte * 4),
    ]


# 车牌检测结果
class NET_DVR_PLATE_RESULT(Structure):
    _fields_ = [
        ("dwSize", c_ulong),
        ("byResultType", c_byte),                     # 0-视频识别结果，1图像识别结果 2 大于10M时走下载路线
        ("byChanIndex", c_byte),
        ("wAlarmRecordID", c_uint16),
        ("dwRelativeTime", c_ulong),
        ("byAbsTime", c_byte * 32),                   # 绝对时间点
        ("dwPicLen", c_ulong),                        # 图片长度(近景图)
        ("dwPicPlateLen", c_ulong),                   # 车牌小图片长度
        ("dwVideoLen", c_ulong),
        ("byTrafficLight", c_byte),
        ("byPicNum", c_byte),
        ("byDriveChan", c_byte),
        ("byVehicleType", c_byte),                    # 车辆类型
        ("dwBinPicLen", c_ulong),
        ("dwCarPicLen", c_ulong),
        ("dwFarCarPicLen", c_ulong),
        ("pBuffer3", POINTER(c_byte)),
        ("pBuffer4", POINTER(c_byte)),
        ("pBuffer5", POINTER(c_byte)),
        ("byRelaLaneDirectionType", c_byte),
        ("byCarDirectionType", c_byte),
        ("byRes3", c_byte * 6),
        ("struPlateInfo", NET_DVR_PLATE_INFO),        # 车牌信息
        ("struVehicleInfo", NET_DVR_VEHICLE_INFO),    # 车辆信息
        ("pBuffer1", POINTER(c_byte)),                # 近景图
        ("pBuffer2", POINTER(c_byte)),                # 车牌图
    ]


# 图片信息
class NET_ITS_PICTURE_INFO(Structure):
    _fields_ = [
        ("dwDataLen", c_ulong),               # 媒体数据长度
        ("byType", c_byte),
        ("byDataType", c_byte),
        ("byCloseUpType", c_byte),
        ("byPicRecogMode", c_byte),
        ("dwRedLightTime", c_ulong),
        ("byAbsTime", c_byte * 32),           # 绝对时间点
        ("struPlateRect", NET_VCA_RECT),      # 车牌位置
        ("struPlateRecgRect", NET_VCA_RECT),
        ("pBuffer", POINTER(c_byte)),         # 数据指针
        ("dwUTCTime", c_ulong),
        ("byCompatibleAblity", c_byte),
        ("byTimeDiffFlag", c_byte),
        ("cTimeDifferenceH", c_char),
        ("cTimeDifferenceM", c_char),
        ("byRes2", c_byte * 4),
    ]


# 车牌检测结果
class NET_ITS_PLATE_RESULT(Structure):
    _fields_ = [
        ("dwSize", c_ulong),
        ("dwMatchNo", c_ulong),
        ("byGroupNum", c_byte),
        ("byPicNo", c_byte),
        ("bySecondCam", c_byte),
        ("byFeaturePicNo", c_byte),
        ("byDriveChan", c_byte),
        ("byVehicleType", c_byte),                    # 车辆类型
        ("byDetSceneID", c_byte),
        ("byVehicleAttribute", c_byte),
        ("wIllegalType", c_uint16),
        ("byIllegalSubType", c_byte * 8),
        ("byPostPicNo", c_byte),
        ("byChanIndex", c_byte),
        ("wSpeedLimit", c_uint16),
        ("byChanIndexEx", c_byte),
        ("byVehiclePositionControl", c_byte),
        ("struPlateInfo", NET_DVR_PLATE_INFO),        # 车牌信息
        ("struVehicleInfo", NET_DVR_VEHICLE_INFO),    # 车辆信息
        ("byMonitoringSiteID", c_byte * 48),
        ("byDeviceID", c_byte * 48),                  # 设备编号
        ("byDir", c_byte),
        ("byDetectType", c_byte),
        ("byRelaLaneDirectionType", c_byte),
        ("byCarDirectionType", c_byte),
        ("dwCustomIllegalType", c_ulong),
        ("pIllegalInfoBuf", POINTER(c_byte)),
        ("byIllegalFromatType", c_byte),
        ("byPendant", c_byte),
        ("byDataAnalysis", c_byte),
        ("byYellowLabelCar", c_byte),
        ("byDangerousVehicles", c_byte),
        ("byPilotSafebelt", c_byte),
        ("byCopilotSafebelt", c_byte),
        ("byPilotSunVisor", c_byte),
        ("byCopilotSunVisor", c_byte),
        ("byPilotCall", c_byte),
        ("byBarrierGateCtrlType", c_byte),
        ("byAlarmDataType", c_byte),
        # ("struSnapFirstPicTime", NET_DVR_TIME_V30),
        ("dwIllegalTime", c_ulong),
        ("dwPicNum", c_ulong),                         # 图片数量
        ("struPicInfo", NET_ITS_PICTURE_INFO * 6),     # 图片信息
    ]


# 报警设备信息
class NET_DVR_ALARMER(Structure):
    _fields_ = [
        ("byUserIDValid", c_byte),
        ("bySerialValid", c_byte),
        ("byVersionValid", c_byte),
        ("byDeviceNameValid", c_byte),
        ("byMacAddrValid", c_byte),
        ("byLinkPortValid", c_byte),
        ("byDeviceIPValid", c_byte),
        ("bySocketIPValid", c_byte),
        ("lUserID", c_long),
        ("sSerialNumber", c_byte * 48),
        ("dwDeviceVersion", c_ulong),
        ("sDeviceName", c_char * 32),
        ("byMacAddr", c_byte * 6),
        ("wLinkPort", c_uint16),
        ("sDeviceIP", c_char * 128),
        ("sSocketIP", c_char * 128),
        ("byIpProtocol", c_byte),
        ("byRes1", c_byte * 2),
        ("bJSONBroken", c_byte),
        ("bySocketIPValid", c_uint16),
        ("byRes2", c_byte * 6),
    ]
