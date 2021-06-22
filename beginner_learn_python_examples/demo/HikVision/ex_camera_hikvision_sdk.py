# -*- coding: utf-8 -*-
"""
 Created on 2021/6/18 22:26
 Filename   : ex_camera_hikvision_sdk.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
import os
import time
from queue import Queue
from hikvision_param import *


# c_byte * number  --> str
def c_byte_array_to_str(c_byte_array):
    result = ''
    for o in c_byte_array:
        result += str(o)
    return result


class HikVision:
    """ HikVision 车辆卡口接口 """
    queue = Queue(maxsize=1000)
    serial_number = ''

    def __init__(self, ip='44.199.84.181', port=8000, user='admin', passwd='ztjjdd123'):
        self.ip = ip
        self.port = port
        self.user = user
        self.passwd = passwd
        self.lib = cdll.LoadLibrary('lib/HCNetSDK.dll')
        self.user_id = 0
        self.l_handle = -1
        """ 初始化，一个程序进程初始化一次即可 """
        self.init_sdk()
        self.set_connet()

    # 报警回调函数  * --> POINTER
    @staticmethod
    @CFUNCTYPE(c_bool, c_long, POINTER(NET_DVR_ALARMER), POINTER(c_char), c_ulong, c_void_p)
    def message_callback(lCommand: c_long, pAlarmer: POINTER(NET_DVR_ALARMER), pAlarmInfo: POINTER(c_char), dwBufLen: c_ulong, pUser: c_void_p):
        """ 报警回调函数 """
        """ 老报警消息(COMM_UPLOAD_PLATE_RESULT) 新报警信息(COMM_ITS_PLATE_RESULT) """
        if lCommand == 0x2800 or lCommand == 0x3050:
            print(f'报警消息：{lCommand}')
            if lCommand == 0x2800:
                plate_result = NET_DVR_PLATE_RESULT()
            else:
                plate_result = NET_ITS_PLATE_RESULT()
            """ strcpy --> memmove   & --> pointer 指针左值，可变可取地址 """
            memmove(pointer(plate_result), pAlarmInfo, sizeof(plate_result))
            """ 车牌 """
            plate_number = c_byte_array_to_str(plate_result.struPlateInfo.sLicense)
            """ 时间 """
            time_capture = ''
            if lCommand == 0x2800:
                time_capture = c_byte_array_to_str(plate_result.byAbsTime)
            else:
                if plate_result.dwPicNum:
                    time_capture = c_byte_array_to_str(plate_result.struPicInfo[0].byAbsTime)
            """ 车辆类型 """
            vehicle_type = plate_result.byVehicleType
            color_list = ['blue', 'yellow', 'white', 'black', 'green']
            """ 车牌颜色 """
            plate_color = 'Other'
            if plate_result.struPlateInfo.byColor in range(0, 5):
                plate_color = color_list[plate_result.struPlateInfo.byColor]
            """ 车辆颜色 """
            vehicle_color = 'Other'
            if plate_result.struVehicleInfo.byColor in range(0, 5):
                vehicle_color = color_list[plate_result.struVehicleInfo.byColor]

            """ 结构化信息 """
            info = {'camera_type': 'hk',
                    'time': time_capture,
                    'serial_number': HikVision.serial_number,
                    'plate_number': plate_number,
                    'plate_color': plate_color,
                    'object_subType': vehicle_type,
                    'vehicle_color': vehicle_color}
            print(info)

            """ 场景图和车牌图  感觉有问题 需要测试"""
            if lCommand == 0x2800:
                """ 场景图 """
                if (not plate_result.dwPicLen) and (plate_result.byResultType == 1):
                    if not os.path.isdir(os.path.abspath('resources/car_plate')):
                        os.makedirs(os.path.abspath('resources/car_plate'))
                    with open(os.path.abspath(f'resources/car_plate/{time_capture}_{plate_number}_global_.jpg'), 'wb') as global_pic:
                        global_buf = plate_result.pBuffer1[0:plate_result.dwPicLen]
                        global_pic.write(bytes(global_buf))
                """ 车牌图 """
                if (not plate_result.dwPicPlateLen) and (plate_result.byResultType == 1):
                    if not os.path.isdir(os.path.abspath('resources/car_plate')):
                        os.makedirs(os.path.abspath('resources/car_plate'))
                    with open(os.path.abspath(f'resources/car_plate/{time_capture}_{plate_number}.jpg'), 'wb') as small_pic:
                        small_buf = plate_result.pBuffer1[0:plate_result.dwPicLen]
                        small_pic.write(bytes(small_buf))
            else:
                for i in range(0, plate_result.dwPicNum):
                    """ 场景图 """
                    if ((not plate_result.struPicInfo[i].dwDataLen) and plate_result.struPicInfo[i].byType == 1) or (plate_result.struPicInfo[i].byType == 2):
                        if not os.path.isdir(os.path.abspath('resources/car_plate')):
                            os.makedirs(os.path.abspath('resources/car_plate'))
                        with open(os.path.abspath(f'resources/car_plate/{time_capture}_{plate_number}_global_{i}.jpg'), 'wb') as global_pic:
                            global_buf = plate_result.struPicInfo[i].pBuffer[0:plate_result.struPicInfo[i].dwDataLen]
                            global_pic.write(bytes(global_buf))
                    """ 车牌图 """
                    if (not plate_result.struPicInfo[i].dwDataLen) and (plate_result.struPicInfo[i].byType == 0):
                        if not os.path.isdir(os.path.abspath('resources/car_plate')):
                            os.makedirs(os.path.abspath('resources/car_plate'))
                        with open(os.path.abspath(f'resources/car_plate/{time_capture}_{plate_number}_{i}.jpg'), 'wb') as small_pic:
                            small_buf = plate_result.struPicInfo[i].pBuffer[0:plate_result.dwPicLen]
                            small_pic.write(bytes(small_buf))

            """ 传递数据 """
            HikVision.queue.put(info)

        return True

    # 打印错误代码
    def print_error_code(self, msg=""):
        error_info = self.lib.NET_DVR_GetLastError()
        print(msg + str(error_info))

    # 初始化 sdk
    def init_sdk(self):
        init_res = self.lib.NET_DVR_Init()
        print(f"初始化 SDK ... {init_res}")
        if not init_res:
            self.print_error_code("NET_DVR_GetLastError 初始化 SDK 失败: the error code is ")
        return init_res

    # 释放 sdk
    def clean_sdk(self):
        result = self.lib.NET_DVR_Cleanup()
        print(f"释放资源 ... {result}")
        if not result:
            self.print_error_code("NET_DVR_Cleanup 释放 SDK 失败: the error code is ")
        return result

    # 设置连接时间、重连
    def set_connet(self):
        # 设置连接时间
        set_overtime = self.lib.NET_DVR_SetConnectTime(2000, 5)
        print(f"NET_DVR_SetConnectTime 设置连接时间 ... {set_overtime}")
        if not set_overtime:
            self.print_error_code("NET_DVR_SetConnectTime 设置超时错误信息失败：the error code is ")
            return False
        # 设置重连
        self.lib.NET_DVR_SetReconnect(10000, True)

    # 登陆设备
    def login(self):
        login_info = NET_DVR_USER_LOGIN_INFO()
        """ 同步登录 """
        login_info.byUseTransport = 0

        """ C++ 传递的是 byte 型数据，需要转换 """
        """ ip """
        i = 0
        for o in bytes(self.ip, "ascii"):
            login_info.sDeviceAddress[i] = o
            i += 1
        """ 端口 """
        login_info.wPort = self.port
        """ 登录名 """
        i = 0
        for o in bytes(self.user, "ascii"):
            login_info.sUserName[i] = o
            i += 1
        """ 密码 """
        i = 0
        for o in bytes(self.passwd, "ascii"):
            login_info.sPassword[i] = o
            i += 1

        """ 设备信息，输出参数 """
        device_info = NET_DVR_DEVICEINFO_V40()
        """ 打印一些设备信息 """
        self.serial_number = c_byte_array_to_str(device_info.struDeviceV30.sSerialNumber)
        print(f'设备序列号：{self.serial_number}')
        # print(f'设备类型：{device_info.struDeviceV30.byDVRType}')
        # print(f'设备型号：{device_info.struDeviceV30.wDevType}')
        """ & --> byref 指针右值，本身没有分配空间 """
        loginInfo1 = byref(login_info)
        loginInfo2 = byref(device_info)
        self.user_id = self.lib.NET_DVR_Login_V40(loginInfo1, loginInfo2)
        print(f'设备登陆 ... {self.user_id}')
        """ -1 表示失败，其他值表示返回的用户 ID 值 """
        if self.user_id == -1:
            self.print_error_code("NET_DVR_Login_V40 用户登录失败: the error code is ")
            self.clean_sdk()
            return False
        return True

    # 登出设备
    def logout(self):
        result = self.lib.NET_DVR_Logout(self.user_id)
        print(f"设备登出 ... {result}")
        if not result:
            self.print_error_code("NET_DVR_Logout 登出设备失败: the error code is ")
        return result

    # 设置报警回调函数
    def set_message_callback_v31(self):
        result = self.lib.NET_DVR_SetDVRMessageCallBack_V31(HikVision.message_callback, self.user_id)
        print(f'设置报警回调函数 ... {result}')
        if result == -1:
            self.print_error_code("NET_DVR_SetDVRMessageCallBack_V31 设置报警回调函数失败: the error code is ")
        return result

    # 启用报警布防
    def setup_alarm_chan_v41(self):
        alarm_param = NET_DVR_SETUPALARM_PARAM()
        alarm_param.dwSize = sizeof(alarm_param)
        alarm_param_ref = byref(alarm_param)
        self.l_handle = self.lib.NET_DVR_SetupAlarmChan_V41(self.user_id, alarm_param_ref)
        print(f'启用报警布防 ... {self.l_handle}')
        if self.l_handle == -1:
            self.print_error_code("NET_DVR_SetupAlarmChan_V41 报警布防失败: the error code is ")
            self.logout()
            self.clean_sdk()
        return self.l_handle

    # 报警撤防
    def close_alarm(self):
        result = self.lib.NET_DVR_CloseAlarmChan_V30(self.l_handle)
        print(f'报警撤防 ... {result}')
        if result == -1:
            self.print_error_code("NET_DVR_CloseAlarmChan_V30 报警布防失败: the error code is ")
            self.logout()
            self.clean_sdk()
        return result

    # ---------------------------------------------------
    # ------------------ 对外三个接口函数 ------------------
    # ---------------------------------------------------
    def subscribe(self):
        """
        请求数据
        """
        self.login()
        self.set_message_callback_v31()
        self.setup_alarm_chan_v41()

    def get(self):
        """
        获取数据
        """
        info = self.queue.get()
        print(info)
        return info

    def quite(self):
        """
        退出
        """
        self.close_alarm()
        self.logout()
        self.clean_sdk()


if __name__ == '__main__':
    hikvision = HikVision()
    hikvision.subscribe()
    if hikvision.user_id == -1:
        exit(-1)

    while True:
        hikvision.get()
        time.sleep(0.1)

    hikvision.quite()
