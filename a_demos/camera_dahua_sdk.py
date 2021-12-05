# -*- coding: utf-8 -*-
"""
 Created on 2021/6/18 22:08
 Filename   : camera_dahua_sdk.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Struct import *
from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE, EM_EVENT_IVS_TYPE
from NetSDK.SDK_Callback import CB_FUNCTYPE
import time
from queue import Queue
from datetime import datetime


class TrafficCallBackAlarmInfo:
    def __init__(self):
        self.time_str = ""
        self.plate_number_str = ""
        self.plate_color_str = ""
        self.object_subType_str = ""
        self.vehicle_color_str = ""

    def get_alarm_info(self, alarm_info):
        self.time_str = '{}-{}-{} {}:{}:{}'.format(alarm_info.UTC.dwYear, alarm_info.UTC.dwMonth, alarm_info.UTC.dwDay,
                                                   alarm_info.UTC.dwHour, alarm_info.UTC.dwMinute, alarm_info.UTC.dwSecond)
        self.plate_number_str = str(alarm_info.stTrafficCar.szPlateNumber.decode('gb2312'))
        self.plate_color_str = str(alarm_info.stTrafficCar.szPlateColor, 'utf-8')
        self.object_subType_str = str(alarm_info.stuVehicle.szObjectSubType, 'utf-8')
        self.vehicle_color_str = str(alarm_info.stTrafficCar.szVehicleColor, 'utf-8')


class DaHuaCarPlateApi:
    def __init__(self):
        self.loginID = C_LLONG()
        self.attachID = C_LLONG()
        # 获取NetSDK对象并初始化
        self.sdk = NetClient()
        self.sdk.InitEx()

    def login(self, ip='44.195.14.202', port=37777, username="admin", password="Zhzt2026"):
        in_param = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        in_param.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
        # 设定IP, 端口, 用户名, 密码
        in_param.szIP = ip.encode()
        in_param.nPort = port
        in_param.szUserName = username.encode()
        in_param.szPassword = password.encode()
        in_param.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        in_param.pCapParam = None
        out_param = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        out_param.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)
        self.loginID, device_info, error_message = self.sdk.LoginWithHighLevelSecurity(in_param, out_param)
        return device_info, error_message

    @CB_FUNCTYPE(None, C_LLONG, C_DWORD, c_void_p, POINTER(c_ubyte), C_DWORD, C_LDWORD, c_int, c_void_p)
    def AnalyzerDataCallBack(lAnalyzerHandle, dwAlarmType, pAlarmInfo, pBuffer, dwBufSize, dwUser, nSequence, reserved):
        """
        大华的获取图片以及数据的回调函数

        后续需要修改保存图片的方式（从保存到本地转为保存到文件服务器）
        并返回图片ID保存在最后的data字典中
        """
        print('Enter AnalyzerDataCallBack')
        # 当报警类型是交通卡口事件时在界面上进行显示相关信息
        if dwAlarmType == EM_EVENT_IVS_TYPE.TRAFFICJUNCTION:
            global callback_num
            local_path = os.path.abspath('demo')
            is_global = False
            is_small = False
            show_info = TrafficCallBackAlarmInfo()
            callback_num += 1
            alarm_info = cast(pAlarmInfo, POINTER(DEV_EVENT_TRAFFICJUNCTION_INFO)).contents
            show_info.get_alarm_info(alarm_info)

            car_plate = show_info.plate_number_str
            time_now = datetime.now().strftime('%Y%m%d%H%M%S%f')

            GlobalScene_buf = None
            small_buf = None
            if alarm_info.stuObject.bPicEnble:
                is_global = True
                GlobalScene_buf = cast(pBuffer, POINTER(c_ubyte * alarm_info.stuObject.stPicInfo.dwOffSet)).contents
                if not os.path.isdir(os.path.join(local_path, 'Global')):
                    os.mkdir(os.path.join(local_path, 'Global'))
                with open('./Global/' + time_now + '_' + car_plate + '.jpg', 'wb+') as global_pic:
                    global_pic.write(bytes(GlobalScene_buf))
                if alarm_info.stuObject.stPicInfo.dwFileLenth > 0:
                    is_small = True
                    small_buf = pBuffer[
                                alarm_info.stuObject.stPicInfo.dwOffSet:alarm_info.stuObject.stPicInfo.dwOffSet +
                                                                        alarm_info.stuObject.stPicInfo.dwFileLenth]
                    if not os.path.isdir(os.path.join(local_path, 'Small')):
                        os.mkdir(os.path.join(local_path, 'Small'))
                    with open('./Small/' + time_now + '_' + car_plate + '.jpg', 'wb+') as small_pic:
                        small_pic.write(bytes(small_buf))
            elif dwBufSize > 0:
                is_global = True
                GlobalScene_buf = cast(pBuffer, POINTER(c_ubyte * dwBufSize)).contents
                if not os.path.isdir(os.path.join(local_path, 'Global')):
                    os.mkdir(os.path.join(local_path, 'Global'))
                with open('./Global/Global_Img' + str(callback_num) + car_plate + '.jpg', 'wb+') as global_pic:
                    global_pic.write(bytes(GlobalScene_buf))
                print('./Global/Global_Img' + str(callback_num) + car_plate + '.jpg')
            data = {
                "dwAlarmType": dwAlarmType,
                "show_info": show_info,
                "callback_num": callback_num,
                "is_global": is_global,
                "is_small": is_small,
                "GlobalScene_buf": GlobalScene_buf,
                "small_buf": small_buf
            }
            data_queue.put(data)
            return

    def subscribe(self, channel=0):
        if self.loginID:
            bNeedPicFile = 1
            dwUser = 0
            self.attachID = self.sdk.RealLoadPictureEx(self.loginID, channel, EM_EVENT_IVS_TYPE.TRAFFICJUNCTION,
                                                       bNeedPicFile, self.AnalyzerDataCallBack, dwUser, None)
            if not self.attachID:
                raise Exception("error " + str(self.sdk.GetLastError()))
            else:
                print("订阅成功(Subscribe success)")

    def get(self):
        # 获取数据
        data = data_queue.get()
        # alarm_info = data["show_info"]
        print(data['show_info'].plate_number_str)
        return data

    def leave(self):
        if self.attachID == 0:
            return
        self.sdk.StopLoadPic(self.attachID)
        self.attachID = 0


if __name__ == '__main__':
    ip_list = ['44.195.14.202', '44.195.14.201', '44.195.14.200', '44.195.14.199', '44.195.14.165', '44.195.14.164']
    data_queue = Queue()
    dahuaAPI = DaHuaCarPlateApi()
    for ip in ip_list:
        device_info, error_message = dahuaAPI.login(ip)
        dahuaAPI.subscribe(channel=0)
    while True:
        for ip in ip_list:
            data = dahuaAPI.get()
        time.sleep(0.1)
    dahuaAPI.leave()
