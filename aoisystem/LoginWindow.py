# This Python file uses the following encoding: utf-8
import os
import sys
import json 
import time
import logging
#import pymysql
import SPiiPlusPython as sp
from PyQt5 import uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PLC.PLCmodbus import PLCConnection
from AoiSystemWindow_test.AoiSystemMain import MyWindow
from InitializeWindow.InitializeWindow import InitializeWindow

class LoginWidget(QWidget):
    def __init__(self):
        super(LoginWidget, self).__init__()
        self.plc_client = None
        self.acs_connection = None
        self.ui = uic.loadUiType("LoginWindow.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon('./icons/xjz.png'))
        self.ui.pushButton_login.clicked.connect(self.login)
        self.load_remembered_credentials()

    
    def Initialization(self):
        try:
            self.plc_client = PLCConnection('20.0.0.200')
            # platform_status = self.plc_client.read_bits(258,1)[0]
            # if platform_status == False:
            #     self.plc_client.write_bit(258,1)
                # time.sleep(20)
            ready_status = self.plc_client.read_bits(100, 1)[0]
            # print(ready_status)+
            # # ready_status =True

            if ready_status:
            # platform_status = self.plc_client.read_bits(258,1)[0]
            # if platform_status:self.plc_client
                self.acs_connection = sp.OpenCommEthernetTCP("10.0.0.100", 701)
                if self.acs_connection == -1:
                    return False
                else:
                    return True
            else:
                return False
            # else:self.plc_client
            #     return False
        except Exception as e:
            print (f"Error:{e}")

    def login(self):
        username = self.ui.username_input.text()
        password = self.ui.password_input.text()
        initialization_flag = self.Initialization()
        if initialization_flag==True:
            self.close()
            self.main_window = MyWindow(app, self.plc_client, self.acs_connection)
            # self.main_window = InitializeWindow(app, self.plc_client, self.acs_connection)
            self.main_window.show()
            # self.main_window.homing_thread()

            
        else:
            QMessageBox.information(self, '警告', '请检查PLC或ACS通讯', QMessageBox.Yes)

#        # 连接数据库验证用户
#        try:
#            with pymysql.connect(
#                host="localhost",
#                user="root",
#                password="123456",
#                database="aoi_image"
#            ) as connection:
#                with connection.cursor() as cursor:
#                    query = "SELECT * FROM users WHERE username = %s AND password = %s"
#                    cursor.execute(query, (username, password))
#                    result = cursor.fetchone()
#                    if result:
#                        # 登录成功，打开主界面
#                        self.main_window = MyWindow()+
#                        self.main_window.show()
#                        self.close()
#                        # 复选框勾选，保存用户名和密码
#                        if self.ui.checkBox_remember_password.isChecked():
#                            self.save_remembered_credentials(username, password)
#                    else:
#                        QMessageBox.warning(self, "错误", "用户名或密码错误")
#        except pymysql.Error as error:
#            QMessageBox.critical(self, "数据库错误", str(error))

    def load_remembered_credentials(self):
        try:
            with open('remembered_credentials.json', 'r') as file:
                data = json.load(file)
                if data.get('remember'):
                    self.ui.username_input.setText(data.get('username'))
                    self.ui.password_input.setText(data.get('password'))
                    self.ui.checkBox_remember_password.setChecked(True)
                    self.ui.checkBox_remember_password.setChecked(True)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_remembered_credentials(self, username, password):
        data = {
            'username': username,
            'password': password,
            'remember': self.ui.checkBox_remember_password.isChecked()
        }
        with open('remembered_credentials.json', 'w') as file:
            json.dump(data, file)


if __name__ == "__main__":
    app = QApplication([])
    widget = LoginWidget()
    widget.show()
    sys.exit(app.exec_())
