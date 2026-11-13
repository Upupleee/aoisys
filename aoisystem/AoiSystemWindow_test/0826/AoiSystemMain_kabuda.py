# 标准库导入
import time
import re
import os
import gc
import logging
import math
import yaml
import tifffile
import threading
import configparser
import numpy as np
import json
from pathlib import Path
from queue import Queue

# 第三方库导入
import SPiiPlusPython as sp
from PIL import Image
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import (QTimer, Qt, QDateTime, QThread, pyqtSignal,  QObject,
                          QSettings)
from PyQt5.QtGui import (QPixmap, QIcon, QPixmapCache)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QProgressBar, 
                            QComboBox, QMenu, QInputDialog, QMessageBox, 
                            QFileDialog, QDialog, QAction, QLineEdit)

# 自定义模块导入
from AoiSystemWindow_new.ImageView import Viewer
from AoiSystemWindow_new.ViewerWindow import ViewerDialog
from .AcsThread import AcsThread
from AoiSystemWindow_new.CameraControl import ProcessController
from .WorkerThread import DetectionController
from Logger.LoggerDialog import LoggerDialog
from ConfigWindow.ConfigDialog import ConfigDialog
from LogoutWindow.LogoutDialog import LogoutDialog
from SystemSettingWindow.SystemSettingWindow import SystemSettingDialog


# 全局配置
Image.MAX_IMAGE_PIXELS = None  # 禁用图片像素限制
MATERIAL_ROOT = "material"
os.makedirs(MATERIAL_ROOT, exist_ok=True)  # 确保料号根目录存在


class MyWindow(QMainWindow):
    def __init__(self, app, plc_client, acs_comm):
        super(MyWindow, self).__init__()
        self.app = app
        self.plc_client = plc_client
        self.acs_comm = acs_comm
        
        # 初始化UI和基本设置
        self._init_ui()
        self._init_variables()
        self._init_logger()
        self._init_timers()
        self._init_acs()
        
        # 设置UI组件
        self._setup_toolbars()
        self._setup_graphics_views()
        self._setup_tables()
        
        # 连接信号和槽
        self._connect_menu_signals()
        self._connect_button_signals()
        self._connect_checkbox_signals()
        self._connect_combo_signals()
        
        # 加载保存的设置和数据
        self.load_settings()
        self.load_material_data()


    def _init_ui(self):
        """初始化UI界面"""
        self.ui = uic.loadUiType("./AoiSystemWindow_new/AoiSystemtest.ui")[0]()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("./icons/xjz.png"))


    def _init_variables(self):
        """初始化成员变量"""
        # 文件路径相关
        self.folder_path = ''
        self.inipath = '/home/aoi/pretest/AOI_V1/SystemSettingWindow/motion_settings.ini'
        self.temp_folder_path = "/home/aoi/pretest/images"
        self.acs_config_path = "./Acs/config_1104.yaml"
        self.ccf_path = "/home/aoi/Documents/best_0429.ccf"
        self.settings_path = os.path.join(os.path.expanduser("~"), ".aoi_software_settings.ini")

        self.current_board_width = None
        self.current_board_height = None
        self.current_board_thickness = None
        
        # 图像和检测相关
        self.image_paths = []
        self.slice_paths = []
        self.temp_slice_widths = None
        self.temp_height = None
        self.selected_image = None
        self.selected_temp_image = None
        self.selector_image = None
        self.gerber_image = False
        self.gerber_image_path = None
        self.current_index = 0
        self.sample_data_display = False
        self.defect_data_display = False
        self.sample_data_path = None
        self.template_fileName = None
        
        # 线程和控制器
        self.quick_start_running = False
        self.combine_th = None
        self.matching_thread = None
        self.detection_project = None
        self.controller = ProcessController()
        self.detection_controller = DetectionController(self)
        self.detection_controller.detection_round_finished.connect(self.on_detection_round_complete)
        self.start_sample_detection_running = False
        
        
        # 数据存储
        self.material_data = {}  # 料号数据（键为不含material根目录的料号名称）
        self.current_material = None  # 当前选中的料号（不含material根目录）
        self.detection_mode = "A"  # 默认检测模式
        self.inference_mode = "整图批量推理"
        self.ab_cycle_count = 0  # AB面模式计数器
        self.scan_records = {}  # 扫描记录（键为不含material根目录的料号名称）
        self.current_scan_id = None  # 当前扫描的ID
        self.current_ab_cycle = 0  # 当前AB面循环次数
        self.current_ab_folder = None  # 当前AB面结果文件夹（包含material根目录）
        self.auto_confirm = False  # 自动确认标志
        
        # 真空和风机控制
        self.vacuum_on = False
        self.blower_on = False
        self.blower_timer = None
        self.last_click_time = 0
        self.time_interval_threshold = 300  # 时间间隔阈值，单位：秒
        
        # 设置
        self.settings = QSettings(self.settings_path, QSettings.IniFormat)


    def _init_logger(self):
        """初始化日志记录器"""
        logging.basicConfig(
            filename='./Logger/aoi_system.log', 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'
        )


    def _init_timers(self):
        """初始化定时器"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_readings)
        self.timer.start(100)


    def _init_acs(self):
        """初始化ACS通信线程"""
        self.acs_thread_obj = AcsThread(self.acs_comm)


    def _setup_toolbars(self):
        """设置工具栏"""
        # 样本图片工具栏
        self._setup_raw_toolbar()
        
        # 模板图片工具栏
        self._setup_template_toolbar()


    def _setup_raw_toolbar(self):
        """设置样本图片工具栏"""
        toolbar = QtWidgets.QToolBar(self.ui.Atoolbar_widget)
        self.ui.Atoolbar_verticalLayout.addWidget(toolbar)
        
        actions = [
            QAction('放大', self.ui.Atoolbar_widget),
            QAction('缩小', self.ui.Atoolbar_widget),
            QAction('重置大小', self.ui.Atoolbar_widget),
            QAction('适应窗口', self.ui.Atoolbar_widget)
        ]
        
        for action in actions:
            toolbar.addAction(action)
            
        actions[0].triggered.connect(self.sample_zoom_in)
        actions[1].triggered.connect(self.sample_zoom_out)
        actions[2].triggered.connect(self.sample_original_size)
        actions[3].triggered.connect(self.sample_fit_to_window)


    def _setup_template_toolbar(self):
        """设置模板图片工具栏"""
        toolbar = QtWidgets.QToolBar(self.ui.Btoolbar_widget)
        self.ui.Btoolbar_verticalLayout.addWidget(toolbar)
        
        actions = [
            QAction('放大', self.ui.Btoolbar_widget),
            QAction('缩小', self.ui.Btoolbar_widget),
            QAction('重置大小', self.ui.Btoolbar_widget),
            QAction('适应窗口', self.ui.Btoolbar_widget)
        ]
        
        for action in actions:
            toolbar.addAction(action)
            
        actions[0].triggered.connect(self.template_zoom_in)
        actions[1].triggered.connect(self.template_zoom_out)
        actions[2].triggered.connect(self.template_original_size)
        actions[3].triggered.connect(self.template_fit_to_window)


    def _setup_graphics_views(self):
        """设置图形视图"""
        # A面视图
        self._setup_graphics_view(
            parent_layout=self.ui.A_verticalLayout,
            old_widget=self.ui.graphicsView_left,
            new_widget_name='graphicsView_A',
            frame_title='A窗口'
        )
        
        # B面视图
        self._setup_graphics_view(
            parent_layout=self.ui.B_verticalLayout,
            old_widget=self.ui.graphicsView_right,
            new_widget_name='graphicsView_B',
            frame_title='B窗口'
        )


    def _setup_graphics_view(self, parent_layout, old_widget, new_widget_name, frame_title):
        """通用的图形视图设置方法"""
        # 创建新的Viewer
        new_viewer = Viewer()
        new_viewer.setFrameShape(QtWidgets.QFrame.Box)
        new_viewer.setFrameShadow(QtWidgets.QFrame.Sunken)
        new_viewer.setLineWidth(2)
        new_viewer.clear()
            
        # 替换旧的widget
        parent_layout.replaceWidget(old_widget, new_viewer)
        old_widget.deleteLater()
        # 设置新widget的属性
        setattr(self.ui, new_widget_name, new_viewer)


    def _setup_tables(self):
        """设置表格属性"""
        # 资料料号表格
        self.ui.materialTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.materialTable.setSelectionBehavior(QTableWidget.SelectRows)
        self.ui.materialTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.materialTable.customContextMenuRequested.connect(self.show_context_menu)
        self.ui.materialTable.cellClicked.connect(self.table_cell_clicked)
        
        # 扫描记录表格
        self.ui.boardTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.boardTable.setSelectionBehavior(QTableWidget.SelectRows)
        self.ui.boardTable.cellClicked.connect(self.board_table_clicked)


    def _connect_menu_signals(self):
        """连接菜单信号"""
        # viewer_menu
        self.ui.Viewer_action.triggered.connect(self.onViewer)
        
        # setting_menu
        self.ui.ConfigSet_action.triggered.connect(self.onConfigSet)
        self.ui.System_action.triggered.connect(self.onSystemSet)
        
        # help_menu
        self.ui.About_action.triggered.connect(self.onHelpAbout)
        self.ui.Log_action.triggered.connect(self.onHelpLog)
        
        # exit_menu
        self.ui.Close_action.triggered.connect(self.onExit)


    def _connect_button_signals(self):
        """连接按钮信号"""
        # 真空控制按钮
        self._setup_button(
            button=self.ui.SorptionButton,
            color="gray",
            slot=self.toggle_vacuum
        )
        
        # 开始样本检测按钮
        self._setup_button(
            button=self.ui.StartSampleButton,
            color="gray",
            slot=self.start_sample_detection
        )
        
        # 开始缺陷检测按钮
        # self._setup_button(
        #     button=self.ui.StartDetectionButton,
        #     color="gray",
        #     slot=self.start_match_detection
        # )
        
        # 重置按钮
        self._setup_button(
            button=self.ui.ResetButton,
            color="gray",
            slot=self.reset
        )
        
        # 设置光源按钮
        self._setup_button(
            button=self.ui.pushButton,
            color="gray",
            slot=self.setLight
        )
        
        # 报警复位按钮
        self._setup_button(
            button=self.ui.AlarmRestButton,
            color="gray",
            slot=self.ResetAlarm
        )


    def _setup_button(self, button, color, slot):
        """通用的按钮设置方法"""
        button.setStyleSheet(f"background-color: {color}")
        button.clicked.connect(slot)


    def _connect_checkbox_signals(self):
        """连接复选框信号"""
        self.ui.lock_ab_checkbox.toggled.connect(self.toggle_lock_ab_mode)
        self.ui.auto_confirm_checkbox.toggled.connect(self.toggle_auto_confirm)


    def _connect_combo_signals(self):
        """连接下拉框信号"""
        self.ui.mode_combo.currentTextChanged.connect(self.change_detection_mode)
        self.ui.inference_comboBox.currentTextChanged.connect(self.change_inference_mode)


    def update_readings(self):
        light_curtain_status = self.plc_client.read_bits(200, 1)[0]
        if light_curtain_status:
            self.plc_client.write_bit(252, 1)
            self.plc_client.write_bit(253, 1)
        start_status = self.plc_client.read_bits(201, 1)[0]
        if start_status:
            if not hasattr(self, 'vacuum_status') or self.vacuum_status is False:
                self.toggle_vacuum()
            self.start_sample_detection()

        wetness = self.plc_client.read_float(204)
        temperature = self.plc_client.read_float(202)
        pressure = self.plc_client.read_float(200)
        self.ui.label_temperature.setText(f"温度: {temperature:.2f} °C")
        self.ui.label_pressure.setText(f"负压: {pressure:.2f} Pa")
        self.ui.label_wetness.setText(f"湿度: {wetness:.2f} %")


    def setLight(self):
        self.ui.pushButton.setStyleSheet("background-color: green")
        value = self.ui.spinBox.value()
        self.plc_client.write_register(150, value)
        self.ui.pushButton.setStyleSheet("background-color: gray")


    def reset(self):
        self.ui.ResetButton.setStyleSheet("background-color: green")
        self.plc_client.write_bit(150, 0)
        self.plc_client.write_bit(151, 1)
        self.plc_client.write_bit(200, 0)
        self.ACS_disconnect()


    def toggle_vacuum(self):
        current_time = time.time()
        time_interval = current_time - self.last_click_time
        self.last_click_time = current_time

        try:
            if not self.vacuum_on:
                self.start_vacuum(time_interval)
            else:
                self.stop_vacuum(time_interval)
            self.vacuum_on = not self.vacuum_on
        except Exception as e:
            print(f"操作出错: {e}")


    def start_vacuum(self, time_interval):
        self.ui.SorptionButton.setStyleSheet("background-color: green")
        try:
            if not self.blower_on or time_interval > self.time_interval_threshold:
                self.turn_on_blower()
            self.plc_client.write_bit(256, 0)
            self.plc_client.write_bit(255, 1)  # 吸真空打开
            self.vacuum_status = True
        except Exception as e:
            print(f"吸真空操作出错: {e}")
            self.ui.SorptionButton.setStyleSheet("background-color: gray")
            self.vacuum_on = False


    def stop_vacuum(self, time_interval):
        self.ui.SorptionButton.setStyleSheet("background-color: gray")
        try:
            self.plc_client.write_bit(256, 1)  # 破真空打开
            self.plc_client.write_bit(255, 0)  # 吸真空关闭
            self.vacuum_status = False
            if time_interval > self.time_interval_threshold:
                if self.blower_timer:
                    self.blower_timer.cancel()
                self.blower_timer = threading.Timer(10, self.turn_off_blower)
                self.blower_timer.start()
        except Exception as e:
            print(f"破真空操作出错: {e}")
            self.ui.SorptionButton.setStyleSheet("background-color: green")
            self.vacuum_on = True


    def turn_on_blower(self):
        self.plc_client.write_bit(254, 1)
        self.blower_on = True


    def turn_off_blower(self):
        self.plc_client.write_bit(254, 0)
        self.blower_on = False


    def on_settings_update(self, row, col):
        self.row = row
        self.col = col
        self.ui.tableWidget.setItem(row, col, QTableWidgetItem("123"))


    def onViewer(self, *args):
        ViewerWidget = ViewerDialog()
        ViewerWidget.exec_()


    # 帮助菜单
    def onHelpAbout(self):
        logging.info('打开帮助窗口')
        QtWidgets.QMessageBox.information(self, 'VersionAbout', '机器视觉前端演示版')


    def onConfigSet(self, *args):
        logging.info('打开参数设置窗口')
        ConfigWidget = ConfigDialog()
        ConfigWidget.exec_()


    def onSystemSet(self, *args):
        logging.info('打开系统设置窗口')
        SystemSetWidget = SystemSettingDialog()
        SystemSetWidget.exec_()


    def onHelpLog(self, *args):
        logging.info('打开帮助日志窗口')
        LoggerWidget = LoggerDialog()
        LoggerWidget.exec_()


    def onExit(self):
        logging.info('打开关闭登出窗口')
        LogoutWidget = LogoutDialog()
        LogoutWidget.exec_()


    # ACS线程
    def acs_thread(self):
        self.acs_thread_obj.add_operation('moving')
        value = math.ceil(int(self.current_board_width )/ 127)
        self.detection_controller.reset(value)
        self.detection_controller.prepare_detection(self.detection_mode)
        print(value)
        if value is not None:
            self.file_queue = Queue()
            self.acs_thread_obj.add_iteration(value)
            self.acs_thread_obj.add_queue(self.file_queue)
            self.acs_thread_obj.signal_received_path.connect(self.on_new_file)
            self.acs_thread_obj.signal_received_finish.connect(self.on_new_files_found)
            self.acs_thread_obj.start()

    def on_new_file(self, _):
        print(_)
        self.detection_controller.enqueue_file(_)


    def on_new_files_found(self):
        QMessageBox.information(self, '提示', '采图结束', QMessageBox.Yes)
        self.plc_client.write_bit(150, 0)
        self.controller.send_exit_command()
        if self.acs_thread_obj is not None:
            try:
                self.acs_thread_obj.signal_received_finish.disconnect()
                self.acs_thread_obj.signal_received_path.disconnect()
            except:
                pass

    def on_detection_round_complete(self):
        if self.detection_mode =="AB":
            self.ab_cycle_count +=1
        self.ui.StartSampleButton.setStyleSheet("background-color: gray")
        self.quick_start_running = False
        self.start_sample_detection_running = False


    # 缩放图片
    def sample_zoom_in(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.zoomIn()
        elif self.defect_data_display:
            self.ui.graphicsView_A.zoomIn()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def sample_zoom_out(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.zoomOut()
        elif self.defect_data_display:
            self.ui.graphicsView_A.zoomOut()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def sample_original_size(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.resetZoom()
        elif self.defect_data_display:
            self.ui.graphicsView_A.resetZoom()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def sample_fit_to_window(self):
        if self.sample_data_display:
            self.ui.graphicsView_A.zoomFit()
        elif self.defect_data_display:
            self.ui.graphicsView_A.zoomFit()
        else:
            QMessageBox.warning(self, "警告", "请先加载图像！")


    def template_zoom_in(self):
        if self.gerber_image:
            self.ui.graphicsView_B.zoomIn()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_zoom_out(self):
        if self.gerber_image:
            self.ui.graphicsView_B.zoomOut()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_original_size(self):
        if self.gerber_image:
            self.ui.graphicsView_B.resetZoom()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_fit_to_window(self):
        if self.gerber_image:
            self.ui.graphicsView_B.zoomFit()
        else:
            QMessageBox.warning(self, "警告", "请先导入template文件！")


    def template_crop(self):
        if self.CropActionTemplate.isChecked():
            self.ui.graphicsView_B.set_crop_flag(True)
        else:
            self.ui.graphicsView_B.set_crop_flag(False)


    def sample_crop(self):
        if self.CropActionSample.isChecked():
            self.ui.graphicsView_A.set_crop_flag(True)
        else:
            self.ui.graphicsView_A.set_crop_flag(False)


    def ACS_disconnect(self):
        self.acs_thread_obj.add_operation('homing')
        self.acs_thread_obj.signal_received_reset.connect(self.disconnect_reset)
        self.acs_thread_obj.start()


    def disconnect_reset(self):
        self.ui.ResetButton.setStyleSheet("background-color: gray")


    def ResetAlarm(self):
        self.plc_client.write_bit(150, 0)
        self.plc_client.write_bit(151, 1)
        self.ui.AlarmRestButton.setStyleSheet("background-color: green")
        if self.plc_client.read_bits(250, 1)[0]:
            self.plc_client.write_bit(250, 0)
        if self.plc_client.read_bits(251, 1)[0]:
            self.plc_client.write_bit(251, 0)
        if self.plc_client.read_bits(252, 1)[0]:
            self.plc_client.write_bit(252, 0)
        if self.plc_client.read_bits(253, 1)[0]:
            self.plc_client.write_bit(253, 0)
        self.ui.AlarmRestButton.setStyleSheet("background-color: gray")


    def start_sample_detection(self):
        if not self.current_material:
            self.show_message("警告", "请先选择一个料号!")
            return
        if self.start_sample_detection_running is True:
            QMessageBox.warning(self, "错误", "正在运行，请等待完成！")
            return
        else:
            self.start_sample_detection_running = True
        self.ui.StartSampleButton.setStyleSheet("background-color: green")
        if self.controller.is_running():
            self.controller.send_start_command()
        else:
            # 防止多次连接重复发送
            try:
                self.controller.started_process.disconnect()
            except:
                pass
            self.controller.started_process.connect(self.controller.send_start_command)
            self.controller.start_process()
            
        QMessageBox.information(self, "提示", "开始采图!")
        time.sleep(5)
        self.plc_client.write_bit(150, 1)
        self.acs_thread()
        return

    def read_yaml_width(self):
        try:
            with open(self.acs_config_path, 'r') as file:
                data = yaml.safe_load(file)
                board_width = data.get('Board', {}).get('width')
                return math.ceil(board_width / 127)
        except FileNotFoundError:
            print("错误: 文件未找到!")
        except yaml.YAMLError as e:
            print(f"错误: YAML 文件解析出错: {e}")
        except Exception as e:
            print(f"错误: 发生了一个未知错误: {e}")
        return None


    def read_ini_param(self, file_path, param_name, expected_type):
        config = configparser.ConfigParser()
        config.read(file_path)

        if 'General' in config:
            section = config['General']
            if param_name in section:
                value = section.get(param_name)
                try:
                    if expected_type == int:
                        value = int(value)
                    elif expected_type == float:
                        value = float(value)
                    elif expected_type == bool:
                        value = config.getboolean('General', param_name)
                except (ValueError, configparser.NoOptionError):
                    pass
                return value
            else:
                print(f"在 'General' 部分中未找到参数 '{param_name}'。")
        else:
            print(f"在.ini文件中未找到 'General' 部分。")

        return None

    def get_slice_image_paths(self, side):
            """获取当前检测面的所有切分后图片路径（包含material根目录）"""
            if not self.current_material or self.current_material not in self.material_data:
                return []
                
            # 从料号数据中获取切分后的路径
            material_data = self.material_data[self.current_material]
            if f"{side}面切分后" in material_data:
                return material_data[f"{side}面切分后"]
                
            # 如果没有直接保存的切分路径，尝试从料号目录中查找
            full_material_id = os.path.join(MATERIAL_ROOT, self.current_material)
            side_folder = os.path.join(full_material_id, side)
            
            if not os.path.exists(side_folder):
                return []
                
            # 获取该目录下的所有图片文件（切分后的）
            slice_paths = []
            for file in os.listdir(side_folder):
                if file.lower().endswith(('.bmp', '.jpg', '.png', '.tif', '.tiff')):
                    slice_path = os.path.join(side_folder, file)
                    slice_paths.append(slice_path)
                    
            # 按文件名排序
            slice_paths.sort()
            # 更新料号数据中的切分后路径
            material_data[f"{side}面切分后"] = slice_paths
            self.save_material_data()
            
            return slice_paths

    def toggle_lock_ab_mode(self, checked):
        """切换锁定AB模式状态"""
        if checked:
            self.ui.mode_combo.setCurrentText("AB")
            self.ui.mode_combo.setEnabled(False)
            self.detection_mode = "AB"
        else:
            self.ui.mode_combo.setEnabled(True)
        self.save_settings()


    def load_settings(self):
        """加载设置"""
        self.auto_confirm = self.settings.value("auto_confirm", False, type=bool)
        self.ui.auto_confirm_checkbox.setChecked(self.auto_confirm)
        
        lock_ab = self.settings.value("lock_ab_mode", False, type=bool)
        self.ui.lock_ab_checkbox.setChecked(lock_ab)
        
        if lock_ab:
            self.ui.mode_combo.setCurrentText("AB")
            self.ui.mode_combo.setEnabled(False)
            self.detection_mode = "AB"
        else:
            last_mode = self.settings.value("last_detection_mode", "A")
            self.ui.mode_combo.setCurrentText(last_mode)
            self.detection_mode = last_mode


    def save_settings(self):
        """保存设置"""
        self.settings.setValue("auto_confirm", self.auto_confirm)
        self.settings.setValue("lock_ab_mode", self.ui.lock_ab_checkbox.isChecked())
        if not self.ui.lock_ab_checkbox.isChecked():
            self.settings.setValue("last_detection_mode", self.detection_mode)
        self.settings.sync()


    def load_material_data(self):
        """从文件加载料号数据"""
        try:
            if os.path.exists("material_data.json"):
                with open("material_data.json", "r", encoding="utf-8") as f:
                    self.material_data = json.load(f)
                self.update_material_table()
        except Exception as e:
            self.show_message("警告", f"加载料号数据失败: {str(e)}")


    def load_scan_records(self):
        """从文件加载扫描记录"""
        try:
            if os.path.exists("scan_records.json"):
                with open("scan_records.json", "r", encoding="utf-8") as f:
                    self.scan_records = json.load(f)
                self.update_board_table()
        except Exception as e:
            self.show_message("警告", f"加载扫描记录失败: {str(e)}")


    def save_material_data(self):
        """保存料号数据到文件"""
        try:
            with open("material_data.json", "w", encoding="utf-8") as f:
                json.dump(self.material_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.show_message("警告", f"保存料号数据失败: {str(e)}")


    def save_scan_records(self):
        """保存扫描记录到文件"""
        try:
            with open("scan_records.json", "w", encoding="utf-8") as f:
                json.dump(self.scan_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.show_message("警告", f"保存扫描记录失败: {str(e)}")


    def update_material_table(self):
        """更新料号表格"""
        self.ui.materialTable.setRowCount(0)
        for row, (material_id, data) in enumerate(self.material_data.items()):
            self.ui.materialTable.insertRow(row)
            
            # 表格中显示的是不含material根目录的料号名称
            id_item = QTableWidgetItem(material_id)
            self.ui.materialTable.setItem(row, 0, id_item)
            
            a_path = data.get("A面", "")
            a_text = os.path.basename(a_path) if a_path else "未设置"
            a_item = QTableWidgetItem(a_text)
            self.ui.materialTable.setItem(row, 1, a_item)
            
            b_path = data.get("B面", "")
            b_text = os.path.basename(b_path) if b_path else "未设置"
            b_item = QTableWidgetItem(b_text)
            self.ui.materialTable.setItem(row, 2, b_item)


    def update_board_table(self):
        """更新板材表格"""
        self.ui.boardTable.setRowCount(0)
        if not self.current_material or self.current_material not in self.scan_records:
            return
            
        records = self.scan_records[self.current_material]
        row_index = 0
        
        for record in records:
            scan_id = record["scan_id"]
            
            # 添加A面记录
            a_defects = record.get("A面", {}).get("defects", [])
            a_image = record.get("A面", {}).get("image_path", "")
            if a_defects or a_image:
                self.ui.boardTable.insertRow(row_index)
                self.ui.boardTable.setItem(row_index, 0, QTableWidgetItem(scan_id))
                self.ui.boardTable.setItem(row_index, 1, QTableWidgetItem("A"))
                self.ui.boardTable.setItem(row_index, 2, QTableWidgetItem(record["scan_time"]))
                
                view_btn = QPushButton("查看详情")
                view_btn.clicked.connect(lambda checked, sid=scan_id, s="A": self.view_scan_details(sid, s))
                self.ui.boardTable.setCellWidget(row_index, 3, view_btn)
                row_index += 1
            
            # 添加B面记录
            b_defects = record.get("B面", {}).get("defects", [])
            b_image = record.get("B面", {}).get("image_path", "")
            if b_defects or b_image:
                self.ui.boardTable.insertRow(row_index)
                self.ui.boardTable.setItem(row_index, 0, QTableWidgetItem(scan_id))
                self.ui.boardTable.setItem(row_index, 1, QTableWidgetItem("B"))
                self.ui.boardTable.setItem(row_index, 2, QTableWidgetItem(record["scan_time"]))
                
                view_btn = QPushButton("查看详情")
                view_btn.clicked.connect(lambda checked, sid=scan_id, s="B": self.view_scan_details(sid, s))
                self.ui.boardTable.setCellWidget(row_index, 3, view_btn)
                row_index += 1


    def show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu()
        new_action = menu.addAction("新建料号")
        load_action = menu.addAction("载入当前料号")
        delete_action = menu.addAction("删除料号")
        
        action = menu.exec_(self.ui.materialTable.mapToGlobal(position))
        if action == new_action:
            self.open_material_import_dialog()
        elif action == load_action:
            self.load_current_material()
        elif action == delete_action:
            self.delete_material()


    def open_material_import_dialog(self):
        """打开料号导入对话框"""
        dialog = MaterialImportDialog(self)
        dialog.auto_confirm = self.auto_confirm
        dialog.material_imported.connect(self.import_material)
        dialog.exec_()


    def import_material(self, material_id, data):
        """导入料号数据（material_id不含material根目录）"""
        if material_id in self.material_data and not self.auto_confirm:
            reply = QMessageBox.question(self, "确认覆盖", f"料号 '{material_id}' 已存在，是否覆盖?", 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        
        self.material_data[material_id] = data
        self.save_material_data()
        self.update_material_table()
        
        if not self.auto_confirm:
            # 显示的路径包含material根目录
            full_material_path = os.path.join(MATERIAL_ROOT, material_id)
            QMessageBox.information(self, "成功", f"料号 '{material_id}' 导入成功，保存路径: {full_material_path}")


    def load_current_material(self):
        """载入当前选中的料号"""
        selected_rows = set(index.row() for index in self.ui.materialTable.selectedIndexes())
        if not selected_rows:
            self.show_message("警告", "请先选择一个料号!")
            return
        
        row = selected_rows.pop()
        material_id = self.ui.materialTable.item(row, 0).text()  # 不含material根目录
        self.current_material = material_id
        self.load_material_images()
        self.update_board_table()

         # 加载并显示板材参数
        material_data = self.material_data.get(self.current_material, {})
        self.current_board_width = material_data.get("板宽(mm)")
        self.current_board_height = material_data.get("板高(mm)")
        self.current_board_thickness = material_data.get("板厚(mm)")
        self.temp_slice_widths = material_data.get("条带宽")
        self.temp_height = material_data.get("资料高")
        self.save_ccf_settings("crop_height")

        
        self.ab_cycle_count = 0
        self.current_ab_cycle = self.get_next_ab_cycle()
        
        if not self.auto_confirm:
            # 显示的路径包含material根目录
            full_material_path = os.path.join(MATERIAL_ROOT, material_id)
            self.show_message("成功", f"已加载料号: {material_id}，路径: {full_material_path}")

    def save_ccf_settings(self, key):
        if key == "crop_height":
            new_value = (int(self.current_board_height)+45+12.883)*129.706
            print(self.current_board_height)
        lines = []
        with open(self.ccf_path, 'r') as f:
            lines = f.readlines()
        with open(self.ccf_path, "w") as f:
            for line in lines:
                line = line.strip()  # 去除行首尾的空白字符
                if line.startswith('Crop Height' + '='):
                    f.write(f"Crop Height={new_value}\n")
                else:
                    f.write(line + '\n')

    def get_next_ab_cycle(self):
        """获取下一个AB循环编号"""
        if not self.current_material:
            return 0
            
        max_cycle = 0
        for record in self.scan_records.get(self.current_material, []):
            cycle = int(record["scan_id"].split("_")[-1]) if "_" in record["scan_id"] else 0
            if cycle > max_cycle:
                max_cycle = cycle
        return max_cycle + 1


    def delete_material(self):
        """删除选中的料号（同时删除实际文件夹）"""
        selected_rows = set(index.row() for index in self.ui.materialTable.selectedIndexes())
        if not selected_rows:
            self.show_message("警告", "请先选择一个料号!")
            return
        
        row = selected_rows.pop()
        material_id = self.ui.materialTable.item(row, 0).text()  # 不含material根目录
        
        # 包含material根目录的完整路径
        full_material_path = os.path.join(MATERIAL_ROOT, material_id)
        
        if not self.auto_confirm:
            reply = QMessageBox.question(self, "确认删除", 
                                        f"确定要删除料号 '{material_id}' 及其所有文件吗?\n路径: {full_material_path}\n此操作不可撤销!", 
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        
        # 从数据中删除
        if material_id in self.material_data:
            del self.material_data[material_id]
        
        if self.current_material == material_id:
            self.current_material = None
            self.clear_material_images()
        
        if material_id in self.scan_records:
            del self.scan_records[material_id]
        
        # 删除实际文件夹
        try:
            import shutil
            if os.path.exists(full_material_path):
                shutil.rmtree(full_material_path)
                print(f"已删除文件夹: {full_material_path}")
        except Exception as e:
            print(f"删除文件夹失败: {e}")
            self.show_message("警告", f"料号数据已删除，但文件夹删除失败: {e}")
        
        self.save_material_data()
        self.save_scan_records()
        self.update_material_table()
        self.update_board_table()


    def load_material_images(self):
        """加载当前料号的图片（优先显示缩略图）"""
        if not self.current_material or self.current_material not in self.material_data:
            return
            
        material_data = self.material_data[self.current_material]
        
        # 加载A面图片（优先缩略图，路径已包含material根目录）
        a_image_path = material_data.get("A面", "")
        a_thumbnail = material_data.get("A面缩略图", "")
        
        if a_thumbnail and os.path.exists(a_thumbnail):
            self.ui.graphicsView_A.set_image(a_thumbnail)
            self.ui.A_label.setText(f"A面图片: {os.path.basename(a_image_path)}")
        elif a_image_path and os.path.exists(a_image_path):
            self.ui.graphicsView_A.set_image(a_image_path)
            self.ui.A_label.setText(f"A面图片: {os.path.basename(a_image_path)}")
        else:
            self.ui.A_label.setText("A面检测结果: 未检测")
        
        # 加载B面图片（优先缩略图，路径已包含material根目录）
        b_image_path = material_data.get("B面", "")
        b_thumbnail = material_data.get("B面缩略图", "")
        
        if b_thumbnail and os.path.exists(b_thumbnail):
            self.ui.graphicsView_B.set_image(b_thumbnail)
            self.ui.B_label.setText(f"B面图片: {os.path.basename(b_image_path)}")
        elif b_image_path and os.path.exists(b_image_path):
            self.ui.graphicsView_B.set_image(b_image_path)
            self.ui.B_label.setText(f"B面图片: {os.path.basename(b_image_path)}")
        else:
            self.ui.B_label.setText("B面检测结果: 未检测")


    def clear_material_images(self):
        """清除当前料号的图片显示"""
        self.ui.A_label.setText("A面检测结果: 未检测")
        self.ui.B_label.setText("B面检测结果: 未检测")
        self.ui.AprogressBar.setValue(0)
        self.ui.BprogressBar.setValue(0)


    def change_inference_mode(self, mode):
        self.inference_mode = mode
        self.save_settings()

    def change_detection_mode(self, mode):
        """更改检测模式"""
        self.detection_mode = mode
        self.save_settings()


    def toggle_auto_confirm(self, checked):
        """切换自动确认模式"""
        self.auto_confirm = checked
        self.save_settings()


    def table_cell_clicked(self, row, col):
        """表格单元格点击事件"""
        if col in [1, 2] and row >= 0:
            material_id = self.ui.materialTable.item(row, 0).text()  # 不含material根目录
            side = "A" if col == 1 else "B"
            
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"选择{side}面图片", "", "图片文件 (*.bmp *.jpg *.png *.tif)"
            )
            
            if file_path:
                if material_id not in self.material_data:
                    self.material_data[material_id] = {}
                
                # 保存原始路径，后续处理会自动保存到material目录下
                self.material_data[material_id][f"{side}面"] = file_path
                self.material_data[material_id][f"{side}面全部"] = [file_path]
                
                # 生成新的缩略图（路径包含material根目录）
                full_material_id = os.path.join(MATERIAL_ROOT, material_id)
                thumbnails_dir = os.path.join(full_material_id, "thumbnails")
                os.makedirs(thumbnails_dir, exist_ok=True)
                
                try:
                    with Image.open(file_path) as img:
                        thumbnail_path = self.generate_thumbnail(img, os.path.basename(file_path), thumbnails_dir)
                        if thumbnail_path:
                            self.material_data[material_id][f"{side}面缩略图"] = thumbnail_path
                except Exception as e:
                    print(f"生成缩略图失败: {str(e)}")
                
                self.update_material_table()
                self.save_material_data()
                
                if self.current_material == material_id:
                    self.load_material_images()


    def generate_thumbnail(self, img, original_filename, thumbnails_dir):
        """生成图片缩略图（路径包含material根目录）"""
        try:
            width, height = img.size
            new_width = max(1, width // 8)
            new_height = max(1, height // 8)
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            base_name, ext = os.path.splitext(original_filename)
            thumbnail_filename = f"{base_name}_thumb.jpg"
            thumbnail_path = os.path.join(thumbnails_dir, thumbnail_filename)
            
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            resized_img.save(thumbnail_path, "JPEG", quality=80)
            
            return thumbnail_path
        except Exception as e:
            print(f"生成缩略图失败: {str(e)}")
            return None


    def get_current_image_path(self, side):
        """获取当前检测面的图片路径（包含material根目录）"""
        if not self.current_material or self.current_material not in self.material_data:
            return None
        return self.material_data[self.current_material].get(f"{side}面", "")


    def view_scan_details(self, scan_id, side):
        """查看扫描详情"""
        if self.current_material not in self.scan_records:
            self.show_message("错误", "找不到此料号的扫描记录")
            return
            
        record = next((r for r in self.scan_records[self.current_material] if r["scan_id"] == scan_id), None)
        
        if not record or side not in record:
            self.show_message("错误", "找不到此扫描记录")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle(f"扫描详情 - {scan_id} - {side}面")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(600, 400)
        image_label.setStyleSheet("border: 1px solid #ccc;")
        layout.addWidget(image_label)
        
        defects_label = QLabel("缺陷信息")
        layout.addWidget(defects_label)
        
        defects_table = QTableWidget()
        defects_table.setColumnCount(3)
        defects_table.setHorizontalHeaderLabels(["缺陷类型", "位置", "大小"])
        defects_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(defects_table)
        
        image_path = record[side]["image_path"]  # 包含material根目录
        defects = record[side]["defects"]
        
        self.display_image_with_defects(image_label, image_path, defects)
        
        defects_table.setRowCount(len(defects))
        for row, defect in enumerate(defects):
            defects_table.setItem(row, 0, QTableWidgetItem(defect["type"]))
            defects_table.setItem(row, 1, QTableWidgetItem(f"({defect['position'][0]}, {defect['position'][1]})"))
            defects_table.setItem(row, 2, QTableWidgetItem(str(defect['size'])))
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()


    def board_table_clicked(self, row, col):
        """板材表格点击事件"""
        if col == 3:
            return


    def show_message(self, title, message):
        """显示消息框"""
        if self.auto_confirm:
            return
        QMessageBox.information(self, title, message)


    def closeEvent(self, event):
        # self.disconnect()
        self.plc_client.write_bit(254, 0)  # CLOSE
        self.plc_client.write_bit(255, 0)
        self.plc_client.write_bit(256, 0)
        self.plc_client.write_register(150, 0)
        sp.DisableAll(self.acs_comm)
        self.plc_client.disconnect()

        event.accept()
        super().closeEvent(event)



class MaterialImportDialog(QDialog):
    """料号导入对话框（优化线程管理）"""
    material_imported = pyqtSignal(str, dict)
    processing_started = pyqtSignal()
    processing_updated = pyqtSignal(int, str)
    processing_completed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("料号导入")
        self.setMinimumSize(800, 600)
        
        self.image_data = []  # 存储 {path, name, side, thumbnail}
        self.material_id = None  # 仅存储料号名称（不含material根目录）
        self.processing_thread =  QThread()   # 处理线程
        self.processing_worker = None  # 处理工作对象
        self.auto_confirm = False
        
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout()
        # 料号和板材参数布局
        params_layout = QVBoxLayout()
        
        # 料号输入
        id_layout = QHBoxLayout()
        id_label = QLabel("料号:")
        self.id_input = QLabel("未设置")
        id_btn = QPushButton("设置料号")
        id_btn.clicked.connect(self.set_material_id)
        
        id_layout.addWidget(id_label)
        id_layout.addWidget(self.id_input)
        id_layout.addWidget(id_btn)
        id_layout.addStretch()
        
        # 板材参数输入
        board_params_layout = QHBoxLayout()
        
        width_layout = QVBoxLayout()
        width_label = QLabel("板宽 (mm):")
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("请输入板宽")
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_input)
        
        height_layout = QVBoxLayout()
        height_label = QLabel("板高 (mm):")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("请输入板高")
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_input)
        
        thickness_layout = QVBoxLayout()
        thickness_label = QLabel("板厚 (mm):")
        self.thickness_input = QLineEdit()
        self.thickness_input.setPlaceholderText("请输入板厚")
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_input)
        
        board_params_layout.addLayout(width_layout)
        board_params_layout.addLayout(height_layout)
        board_params_layout.addLayout(thickness_layout)
        board_params_layout.addStretch()
        
        params_layout.addLayout(id_layout)
        params_layout.addLayout(board_params_layout)
        
        main_layout.addLayout(params_layout)

        
        # 文件夹选择区域
        select_layout = QHBoxLayout()
        self.folder_label = QLabel("未选择文件夹")
        self.select_folder_btn = QPushButton("选择图片文件夹")
        self.select_folder_btn.clicked.connect(self.select_folder)
        
        select_layout.addWidget(self.folder_label)
        select_layout.addWidget(self.select_folder_btn)
        
        main_layout.addLayout(select_layout)
        
        # 表格区域
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["图片名称", "A/B面设置", "缩略图"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setColumnWidth(2, 150)  # 缩略图列宽度
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        
        main_layout.addWidget(self.table)
        
        # 进度显示
        self.processing_progress = QProgressBar()
        self.processing_progress.setTextVisible(True)
        self.processing_progress.setValue(0)
        self.processing_status = QLabel("就绪")
        main_layout.addWidget(self.processing_status)
        main_layout.addWidget(self.processing_progress)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        self.process_all_btn = QPushButton("全部解析")
        self.process_all_btn.clicked.connect(self.process_all_images)
        self.process_all_btn.setEnabled(False)
        
        self.confirm_btn = QPushButton("确定")
        self.confirm_btn.clicked.connect(self.confirm_import)
        self.confirm_btn.setEnabled(False)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.process_all_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(btn_layout)
        
        self.setLayout(main_layout)
        
        # 连接进度信号
        self.processing_updated.connect(self.update_processing_progress)
        self.processing_completed.connect(self.on_processing_completed)
    def set_material_id(self):
        """设置料号"""
        material_id, ok = QInputDialog.getText(self, "设置料号", "请输入料号:", text=self.material_id)
        if ok and material_id:
            self.material_id = material_id
            self.id_input.setText(material_id)
    
    def select_folder(self):
        """选择图片文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder_path:
            self.folder_label.setText(f"当前文件夹: {folder_path}")
            
            # 自动根据文件夹名称和时间设置料号
            folder_name = os.path.basename(folder_path)
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.material_id = f"{folder_name}_{current_time}"
            self.id_input.setText(self.material_id)
            
            self.load_images(folder_path)
    
    def load_images(self, folder_path):
        """加载图片到表格"""
        # 清空表格和数据
        self.table.setRowCount(0)
        self.image_data = []
        
        # 获取所有图片文件
        img_files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.bmp', '.jpg', '.png', '.tif'))]
        
        if not img_files:
            self.show_message("警告", "所选文件夹中没有图片文件")
            return
        
        # 创建缩略图目录（路径包含material根目录）
        full_material_id = os.path.join(MATERIAL_ROOT, self.material_id)
        thumbnails_dir = os.path.join(full_material_id, "thumbnails")
        os.makedirs(thumbnails_dir, exist_ok=True)
        os.chmod(thumbnails_dir, 0o777)
        
        # 填充表格
        self.table.setRowCount(len(img_files))
        
        for row, filename in enumerate(img_files):
            file_path = os.path.join(folder_path, filename)
            
            # 图片名称
            name_item = QTableWidgetItem(filename)
            self.table.setItem(row, 0, name_item)
            
            # A/B面设置下拉框
            combo_box = QComboBox()
            combo_box.addItems(["未设置", "A", "B"])
            self.table.setCellWidget(row, 1, combo_box)
            
            # 缩略图预览
            thumbnail_label = QLabel("生成中...")
            thumbnail_label.setAlignment(Qt.AlignCenter)
            thumbnail_label.setStyleSheet("border: 1px solid #ccc;")
            self.table.setCellWidget(row, 2, thumbnail_label)
            
            # 解析按钮
            process_btn = QPushButton("解析此图片")
            process_btn.clicked.connect(lambda checked, r=row: self.process_single_image(r))
            self.table.setCellWidget(row, 3, process_btn)
            
            # 保存图片数据
            self.image_data.append({
                "path": file_path,
                "name": filename,
                "side": None,
                "thumbnail": None,
                "slice_widths":None,
                "temp_height":None
            })
            
            # 异步生成缩略图
            self.generate_thumbnail_async(file_path, row, thumbnail_label)
        
        self.process_all_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
    
    def generate_thumbnail_async(self, image_path, row, thumbnail_label):
        """异步生成缩略图（优化QPixmap引用）"""
        try:
            full_material_id = os.path.join(MATERIAL_ROOT, self.material_id)
            thumbnails_dir = os.path.join(full_material_id, "thumbnails")
            os.makedirs(thumbnails_dir, exist_ok=True)
            
            with Image.open(image_path) as img:
                base_name = os.path.basename(image_path)
                thumbnail_path = self.generate_thumbnail(img, base_name, thumbnails_dir)
                
                if thumbnail_path and os.path.exists(thumbnail_path):
                    self.image_data[row]["thumbnail"] = thumbnail_path
                    # 关键优化：使用局部变量存储pixmap，避免长期引用
                    pixmap = QPixmap(thumbnail_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            thumbnail_label.size(), 
                            Qt.KeepAspectRatio, 
                            Qt.SmoothTransformation
                        )
                        thumbnail_label.setPixmap(scaled_pixmap)
                    else:
                        thumbnail_label.setText("无法加载")
        except Exception as e:
            print(f"生成缩略图失败: {str(e)}")
            thumbnail_label.setText("生成失败")
        # 显式删除临时pixmap引用
        finally:
            # 删除局部 QPixmap
            del pixmap
            QPixmapCache.clear()
            QApplication.processEvents()  # 强制刷新UI，释放pixmap资源
    
    def generate_thumbnail(self, img, original_filename, thumbnails_dir):
        """生成图片缩略图"""
        try:
            # 获取原始尺寸
            width, height = img.size
            
            # 计算缩略图尺寸
            new_width = max(1, width // 8)
            new_height = max(1, height // 8)
            
            # 调整图片分辨率
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 构建缩略图路径（包含material根目录）
            base_name, ext = os.path.splitext(original_filename)
            thumbnail_filename = f"{base_name}_thumb.jpg"
            thumbnail_path = os.path.join(thumbnails_dir, thumbnail_filename)
            
            # 保存缩略图
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            resized_img.save(thumbnail_path, "JPEG", quality=80)
            
            return thumbnail_path
        except Exception as e:
            print(f"生成缩略图失败: {str(e)}")
            return None
        
    def validate_board_parameters(self):
        """验证板宽、板高、板厚参数是否已填写"""
        self.board_width = self.width_input.text().strip()
        self.board_height = self.height_input.text().strip()
        self.board_thickness = self.thickness_input.text().strip()
        
        if not self.board_width or not self.board_height or not self.board_thickness:
            return False
            
        # 检查是否为数字
        try:
            float(self.board_width)
            float(self.board_height)
            float(self.board_thickness)
            return True
        except ValueError:
            return False

    def show_loading_status(self, message):
        """显示图片加载状态"""
        self.processing_status.setText(message)
    

    def process_all_images(self):
        """处理所有图片（修复线程销毁问题）"""
        if not self.validate_board_parameters():
            self.show_message("警告", "请先正确填写板宽、板高、板厚参数（必须为数字）")
            return
        images_to_process = []
        for row, info in enumerate(self.image_data):
            combo_box = self.table.cellWidget(row, 1)
            side_text = combo_box.currentText()
            if side_text != "未设置":
                info["side"] = side_text
                images_to_process.append((info["path"], side_text))
        
        if not images_to_process:
            self.show_message("警告", "没有设置A面或B面的图片")
            return
        
        # 停止现有线程
        self._stop_processing_thread()
        
        # 创建工作对象和线程
        self.processing_worker = ImageProcessingWorker(
            images_and_sides=images_to_process,
            material_id=self.material_id
        )

        self.processing_worker.moveToThread(self.processing_thread)
                # 断开旧信号连接，避免重复触发
        
        # 连接信号槽（核心：明确任务流程，避免中途销毁）
        self.processing_thread.started.connect(self.processing_worker.start_processing)  # 统一用start_work
        self.processing_worker.progress_updated.connect(self.processing_updated)
        self.processing_worker.processing_completed.connect(self.processing_completed)  # 所有任务完成后才触发
        self.processing_worker.processing_failed.connect(self.on_processing_failed)
        self.processing_worker.image_loading.connect(self.show_loading_status)
        
        # 修复：线程清理函数（确保线程停止后再销毁）
        def cleanup():
            # 关键：先确保线程已停止
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.quit()  # 强制停止
                self.processing_thread.wait()  # 等待1秒确保停止
            
            # 安全断开信号（避免重复触发）
            try:
                if self.processing_worker:
                    self.processing_worker.progress_updated.disconnect()
                    self.processing_worker.processing_completed.disconnect()
                    self.processing_worker.processing_failed.disconnect()
                    self.processing_worker.image_loading.disconnect()
                if self.processing_thread:
                    self.processing_thread.started.disconnect()
            except TypeError:
                pass  # 忽略已断开的连接
            
            # 销毁对象并清除引用
            if self.processing_worker:
                self.processing_worker.deleteLater()
            
            self.processing_worker = None

        
        # 修复：仅在所有任务完全结束后才触发清理（避免中途调用）
        self.processing_worker.processing_completed.connect(cleanup)  # 所有任务成功完成
        self.processing_worker.processing_failed.connect(cleanup)     # 任务出错终止
        self.processing_thread.finished.connect(cleanup)              # 线程自然结束
        
        # 更新UI状态
        self.processing_status.setText("正在批量解析图片...")
        self.processing_progress.setValue(0)
        self.process_all_btn.setEnabled(False)
        self.confirm_btn.setEnabled(False)
        self.processing_started.emit()
        
        # 启动线程
        self.processing_thread.start()
    
    def _stop_processing_thread(self):
        """安全停止处理线程（核心修复）"""
        # 使用try-except捕获已销毁对象的异常
        try:
            if self.processing_thread and self.processing_thread.isRunning():
                if self.processing_worker:
                    self.processing_worker.stop_processing()
                    self.processing_worker = None
                self.processing_thread.quit()
                self.processing_thread.wait()
        except RuntimeError as e:
            print(f"停止线程时出错: {e}")
        finally:
            self.processing_worker = None


    def _cleanup_resources(self):
        """清理图片资源和表格部件（核心优化）"""
        # 1. 清空表格，释放所有单元格部件（尤其是QPixmap）
        self.table.clearContents()  # 清除单元格内容
        self.table.setRowCount(0)   # 重置行数，销毁行对象
        
        # 2. 清除图片数据引用
        self.image_data = []
        
        # 3. 强制释放QPixmap缓存
        QPixmapCache.clear()
        
        # 4. 强制Python垃圾回收
        gc.collect(generation=0)

    def reject(self):
        """对话框关闭时确保线程资源释放"""
        self._stop_processing_thread()
        self._cleanup_resources()
        self.image_data.clear()
        gc.collect(generation=0)    
        super().reject()
    
    def update_processing_progress(self, progress, filename):
        """更新处理进度"""
        self.processing_progress.setValue(progress)
        self.processing_status.setText(f"正在处理: {filename} ({progress}%)")
    
    def on_processing_completed(self, results):
        """批量处理完成回调"""
        success_count = len(results["success"])
        failed_count = len(results["failed"])
        
        # 显示的路径包含material根目录
        full_material_path = os.path.join(MATERIAL_ROOT, self.material_id)
        message = f"图片解析与解析完成，已保存到 {full_material_path} 文件夹\n"
        message += f"成功: {success_count} 张\n"
        
        # 更新缩略图信息
        for success_info in results["success"]:
            original_path = success_info["original"]
            thumbnail_path = success_info["thumbnail"]
            slice_widths = success_info["slice_widths"]
            temp_height = success_info["temp_height"]
            
            # 查找对应的图片数据并更新缩略图路径
            for info in self.image_data:
                if info["path"] == original_path:
                    info["thumbnail"] = thumbnail_path
                    info["slice_widths"] = slice_widths
                    info["temp_height"] = temp_height
                    break
        
        if failed_count > 0:
            message += f"失败: {failed_count} 张"
            if not self.auto_confirm:
                error_list = "\n".join([f"{os.path.basename(f['path'])} ({f['side']}面): {f['error']}" for f in results["failed"]])
                message += f"\n\n失败详情:\n{error_list}"
        
        self.show_message("成功", message)
        self.process_all_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
        self.processing_status.setText("就绪")
        self.processing_progress.setValue(0)
    
    def on_processing_failed(self, error, filename):
        """处理失败回调"""
        self.show_message("错误", f"处理{'批量' if filename == '批量处理' else filename}时出错: {error}")
        self.process_all_btn.setEnabled(True)
        self.confirm_btn.setEnabled(True)
        self.processing_status.setText("就绪")
        self.processing_progress.setValue(0)
    
    def confirm_import(self):
        # 验证板材参数
        if not self.validate_board_parameters():
            self.show_message("警告", "请先正确填写板宽、板高、板厚参数（必须为数字）")
            return
        
        """确认导入"""
        if not self.material_id:
            self.show_message("警告", "请先设置料号!")
            return
        
        # 收集A面和B面图片（路径已包含material根目录）
        a_images = []
        b_images = []
        a_thumbnail = None
        b_thumbnail = None
        slice_widths =None
        temp_height =None
        
        for info in self.image_data:
            slice_widths = info["slice_widths"]
            temp_height = info["temp_height"]
            if info["side"] == "A":
                a_images.append(info["path"])
                if not a_thumbnail and info.get("thumbnail"):
                    a_thumbnail = info["thumbnail"]
            elif info["side"] == "B":
                b_images.append(info["path"])
                if not b_thumbnail and info.get("thumbnail"):
                    b_thumbnail = info["thumbnail"]
        
        if not a_images and not b_images:
            self.show_message("警告", "至少需要选择一张A面或B面图片!")
            return
        
        # 准备导入数据
        import_data = {
            "A面": a_images[0] if a_images else "",
            "B面": b_images[0] if b_images else "",
            "A面全部": a_images,
            "B面全部": b_images,
            "A面缩略图": a_thumbnail,
            "B面缩略图": b_thumbnail,
            "板宽(mm)": self.board_width,
            "板高(mm)": self.board_height,
            "板厚(mm)": self.board_thickness,
            "条带宽": slice_widths,
            "资料高": temp_height 
        }
        
        # 发送导入信号（传递不含material根目录的料号名称）
        self.material_imported.emit(self.material_id, import_data)
        self._cleanup_resources()  # 新增：清理资源
        self.accept()
    
    def show_message(self, title, message):
        """显示消息框"""
        if hasattr(self, 'auto_confirm') and self.auto_confirm:
            return
        QMessageBox.information(self, title, message)
    



class ImageProcessingWorker(QObject):
    """图片处理工作对象（使用QObject实现，配合moveToThread）"""
    progress_updated = pyqtSignal(int, str)  # 进度值, 处理中的图片名称
    processing_completed = pyqtSignal(dict)
    processing_failed = pyqtSignal(str, str)  # 错误信息, 图片名称
    image_loading = pyqtSignal(str)  # 图片加载进度信号
    
    def __init__(self, images_and_sides, material_id, generate_thumbnails=True):
        super().__init__()
        self.images_and_sides = images_and_sides  # 格式: [(path, side), ...]
        self.material_path = os.path.join(MATERIAL_ROOT, material_id)  # 完整料号路径
        self.generate_thumbnails = generate_thumbnails
        self.running = False  # 运行状态标志（用于安全停止）
        
        # 解析参数
        self.segment_mm = 128
        self.step_mm = 113.3
        self.start_mm = 104.7
        self.offset_mm = 0
    
    def start_processing(self):
        """批量处理入口（确保所有任务完成后再发送信号）"""
        self.running = True
        results = {"success": [], "failed": []}
        total = len(self.images_and_sides)
        
        try:
            self._create_necessary_dirs()
            
            # 循环处理所有图片（确保全部完成后再结束）
            for i, (image_path, side) in enumerate(self.images_and_sides):
                if not self.running:
                    break  # 若被停止，直接退出
                
                try:
                    filename = os.path.basename(image_path)
                    self.image_loading.emit(f"开始加载图片: {filename}")
                    
                    with Image.open(image_path) as img:
                        self.image_loading.emit(f"图片加载完成，开始处理: {filename}")
                        thumbnail_path = self._generate_thumbnail(img, filename) if self.generate_thumbnails else None
                        saved_paths, slice_widths, Height = self._slice_image(img, filename, side)
                    del img
                    results["success"].append({
                        "original": image_path, "processed": saved_paths, 
                        "side": side, "thumbnail": thumbnail_path, "slice_widths":slice_widths, "temp_height":Height
                    })
                
                except Exception as e:
                    results["failed"].append({"path": image_path, "error": str(e), "side": side})
                
                finally:
                    # 发送进度更新（确保线程仍在运行）
                    if self.running:
                        progress = int((i + 1) / total * 100)
                        self.progress_updated.emit(progress, filename)

                    gc.collect(generation=0)
            
            # 所有任务完成后才发送完成信号
            if self.running:
                self.processing_completed.emit(results)
        
        except Exception as e:
            self.processing_failed.emit(str(e), "批量处理")
    
    def stop_processing(self):
        """安全停止处理过程"""
        self.running = False
    
    def _create_necessary_dirs(self):
        """创建必要的目录结构"""
        # 主目录
        os.makedirs(self.material_path, exist_ok=True)
        os.chmod(self.material_path, 0o777)
        
        # 缩略图目录
        thumb_dir = os.path.join(self.material_path, "thumbnails")
        os.makedirs(thumb_dir, exist_ok=True)
        os.chmod(thumb_dir, 0o777)
    
    def _generate_thumbnail(self, img, original_filename):
        """生成缩略图（内部方法）"""
        try:
            # 计算缩略图尺寸
            width, height = img.size
            new_width = max(1, width // 8)
            new_height = max(1, height // 8)
            
            # 生成缩略图
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 保存缩略图
            base_name, ext = os.path.splitext(original_filename)
            thumb_path = os.path.join(self.material_path, "thumbnails", f"{base_name}_thumb.jpg")
            
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            resized_img.save(thumb_path, "JPEG", quality=80)

            del resized_img
            gc.collect()
            return thumb_path
            
        except Exception as e:
            print(f"生成缩略图失败: {str(e)}")
            return None
    
    def _slice_image_old(self, img, img_filename, side):
        """解析图片（内部方法）"""
        Image.MAX_IMAGE_PIXELS = None  # 禁用解压炸弹检查
        saved_paths = []
        
        try:
            # 转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')
                
            # 计算参数
            pixels_per_mm = 16384 / self.segment_mm
            W, H = img.size
            x_left = W
            i = -1
            
            # 创建侧面目录
            side_dir = os.path.join(self.material_path, side)
            os.makedirs(side_dir, exist_ok=True)
            os.chmod(side_dir, 0o777)
            
            # 切分图片
            base_name = os.path.splitext(img_filename)[0]
            while x_left != 0 and self.running:  # 循环中检查运行状态
                # 计算切分区域
                right_mm = (self.start_mm - self.offset_mm) + i * self.step_mm
                left_mm = self.start_mm + (i + 1) * self.step_mm
                
                right_mm = max(0, right_mm)
                x_right = int(W - right_mm * pixels_per_mm)
                x_left = int(W - left_mm * pixels_per_mm)
                x_left = max(0, x_left)
                
                # 裁剪并保存
                box = (x_left, 0, x_right, H)
                slice_img = img.crop(box)
                
                slice_path = os.path.join(side_dir, f"{base_name}_{i+1}.tif")
                slice_img.save(slice_path)
                saved_paths.append(slice_path)

                del slice_img
                gc.collect(generation=0)
                
                i += 1
                
        except Exception as e:
            print(f"图片切分失败: {str(e)}")
            raise  # 重新抛出异常让上层处理
        
        return saved_paths


 ##modify by jingshu 0720
    def _slice_image(self, img, img_filename, side):


        # 禁用解压炸弹检查
        Image.MAX_IMAGE_PIXELS = None
        saved_paths = []
        slice_widths = []

        # 转灰度
        if img.mode != 'L':
            img = img.convert('L')

        # 读取参数
        dpi       = 3250                # Gerber 导出 DPI，例如 3250
        step_mm   = 120            # 相机移动步进，mm， 120.0
        margin_mm = 5.0  #右侧首条带右侧留白
        seg_mm = 128 #   #对应16384的相机视野
          # 右侧未拍摄留白，mm

        # 计算像素／毫米
        px_per_mm = dpi / 25.4
        first_px  = int(round((seg_mm - margin_mm) * px_per_mm)) #修正切分逻辑 by
        step_px   = int(round(step_mm * px_per_mm))
        # margin_px = int(round(margin_mm * px_per_mm))

        # 图像尺寸
        W, H = img.size

        # 创建侧面目录
        side_dir = os.path.join(self.material_path, side)
        os.makedirs(side_dir, exist_ok=True)
        os.chmod(side_dir, 0o777)

        base_name = os.path.splitext(img_filename)[0]
        x_right = W
        x_left = max(0, W - first_px)
        i = 0

        # 第一个条带（右侧首段）
        while (x_right > 0) and getattr(self, 'running', True):
            current_width = x_right -x_left
            slice_widths.append(current_width)
            # 裁剪并保存
            box = (x_left, 0, x_right, H)
            slice_img = img.crop(box)
            slice_path = os.path.join(side_dir, f"{base_name}_{i+1}.tif")
            slice_img.save(slice_path)
            saved_paths.append(slice_path)
            del slice_img
            gc.collect(generation=0)

            # 更新边界，之后按 step_px 切分
            x_right = x_left
            x_left  = max(0, x_left - step_px)
            i += 1

        return saved_paths, slice_widths, H

    ## modified and