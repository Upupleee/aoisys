# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'SystemSettingWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.8.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(717, 492)
        self.gridLayout = QGridLayout(Dialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.tabWidget = QTabWidget(Dialog)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab_platform = QWidget()
        self.tab_platform.setObjectName(u"tab_platform")
        self.gridLayout_2 = QGridLayout(self.tab_platform)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFormAlignment(Qt.AlignCenter)
        self.label_x_speed = QLabel(self.tab_platform)
        self.label_x_speed.setObjectName(u"label_x_speed")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_x_speed)

        self.lineEdit_x_speed = QLineEdit(self.tab_platform)
        self.lineEdit_x_speed.setObjectName(u"lineEdit_x_speed")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.lineEdit_x_speed)

        self.label_x_acceleration = QLabel(self.tab_platform)
        self.label_x_acceleration.setObjectName(u"label_x_acceleration")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_x_acceleration)

        self.lineEdit_x_acceleration = QLineEdit(self.tab_platform)
        self.lineEdit_x_acceleration.setObjectName(u"lineEdit_x_acceleration")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.lineEdit_x_acceleration)

        self.label_x_distance = QLabel(self.tab_platform)
        self.label_x_distance.setObjectName(u"label_x_distance")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_x_distance)

        self.lineEdit_x_distance = QLineEdit(self.tab_platform)
        self.lineEdit_x_distance.setObjectName(u"lineEdit_x_distance")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.lineEdit_x_distance)

        self.label_x_absolute_position = QLabel(self.tab_platform)
        self.label_x_absolute_position.setObjectName(u"label_x_absolute_position")

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.label_x_absolute_position)

        self.lineEdit_x_absolute_position = QLineEdit(self.tab_platform)
        self.lineEdit_x_absolute_position.setObjectName(u"lineEdit_x_absolute_position")

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.lineEdit_x_absolute_position)

        self.lineEdit_x_relative_position = QLineEdit(self.tab_platform)
        self.lineEdit_x_relative_position.setObjectName(u"lineEdit_x_relative_position")

        self.formLayout.setWidget(8, QFormLayout.FieldRole, self.lineEdit_x_relative_position)

        self.label_x_relative_position = QLabel(self.tab_platform)
        self.label_x_relative_position.setObjectName(u"label_x_relative_position")

        self.formLayout.setWidget(8, QFormLayout.LabelRole, self.label_x_relative_position)

        self.verticalSpacer = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout.setItem(1, QFormLayout.FieldRole, self.verticalSpacer)

        self.verticalSpacer_2 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout.setItem(3, QFormLayout.FieldRole, self.verticalSpacer_2)

        self.verticalSpacer_3 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout.setItem(5, QFormLayout.FieldRole, self.verticalSpacer_3)

        self.verticalSpacer_4 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout.setItem(7, QFormLayout.FieldRole, self.verticalSpacer_4)


        self.gridLayout_2.addLayout(self.formLayout, 0, 0, 1, 1)

        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setFormAlignment(Qt.AlignCenter)
        self.label_y_speed = QLabel(self.tab_platform)
        self.label_y_speed.setObjectName(u"label_y_speed")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_y_speed)

        self.lineEdit_y_speed = QLineEdit(self.tab_platform)
        self.lineEdit_y_speed.setObjectName(u"lineEdit_y_speed")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.lineEdit_y_speed)

        self.label_y_acceleration = QLabel(self.tab_platform)
        self.label_y_acceleration.setObjectName(u"label_y_acceleration")

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.label_y_acceleration)

        self.lineEdit_y_acceleration = QLineEdit(self.tab_platform)
        self.lineEdit_y_acceleration.setObjectName(u"lineEdit_y_acceleration")

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.lineEdit_y_acceleration)

        self.label_y_distance = QLabel(self.tab_platform)
        self.label_y_distance.setObjectName(u"label_y_distance")

        self.formLayout_2.setWidget(4, QFormLayout.LabelRole, self.label_y_distance)

        self.lineEdit_y_distance = QLineEdit(self.tab_platform)
        self.lineEdit_y_distance.setObjectName(u"lineEdit_y_distance")

        self.formLayout_2.setWidget(4, QFormLayout.FieldRole, self.lineEdit_y_distance)

        self.label_y_absolute_position = QLabel(self.tab_platform)
        self.label_y_absolute_position.setObjectName(u"label_y_absolute_position")

        self.formLayout_2.setWidget(6, QFormLayout.LabelRole, self.label_y_absolute_position)

        self.lineEdit_y_absolute_position = QLineEdit(self.tab_platform)
        self.lineEdit_y_absolute_position.setObjectName(u"lineEdit_y_absolute_position")

        self.formLayout_2.setWidget(6, QFormLayout.FieldRole, self.lineEdit_y_absolute_position)

        self.lineEdit_y_relative_position = QLineEdit(self.tab_platform)
        self.lineEdit_y_relative_position.setObjectName(u"lineEdit_y_relative_position")

        self.formLayout_2.setWidget(8, QFormLayout.FieldRole, self.lineEdit_y_relative_position)

        self.label_y_relative_position = QLabel(self.tab_platform)
        self.label_y_relative_position.setObjectName(u"label_y_relative_position")

        self.formLayout_2.setWidget(8, QFormLayout.LabelRole, self.label_y_relative_position)

        self.verticalSpacer_5 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_2.setItem(1, QFormLayout.FieldRole, self.verticalSpacer_5)

        self.verticalSpacer_6 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_2.setItem(3, QFormLayout.FieldRole, self.verticalSpacer_6)

        self.verticalSpacer_7 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_2.setItem(5, QFormLayout.FieldRole, self.verticalSpacer_7)

        self.verticalSpacer_8 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_2.setItem(7, QFormLayout.FieldRole, self.verticalSpacer_8)


        self.gridLayout_2.addLayout(self.formLayout_2, 0, 1, 1, 1)

        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setFormAlignment(Qt.AlignCenter)
        self.label_z_speed = QLabel(self.tab_platform)
        self.label_z_speed.setObjectName(u"label_z_speed")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label_z_speed)

        self.lineEdit_z_speed = QLineEdit(self.tab_platform)
        self.lineEdit_z_speed.setObjectName(u"lineEdit_z_speed")

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.lineEdit_z_speed)

        self.label_z_acceleration = QLabel(self.tab_platform)
        self.label_z_acceleration.setObjectName(u"label_z_acceleration")

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.label_z_acceleration)

        self.lineEdit_z_acceleration = QLineEdit(self.tab_platform)
        self.lineEdit_z_acceleration.setObjectName(u"lineEdit_z_acceleration")

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.lineEdit_z_acceleration)

        self.label_z_distance = QLabel(self.tab_platform)
        self.label_z_distance.setObjectName(u"label_z_distance")

        self.formLayout_3.setWidget(4, QFormLayout.LabelRole, self.label_z_distance)

        self.lineEdit_z_distance = QLineEdit(self.tab_platform)
        self.lineEdit_z_distance.setObjectName(u"lineEdit_z_distance")

        self.formLayout_3.setWidget(4, QFormLayout.FieldRole, self.lineEdit_z_distance)

        self.label_z_absolute_position = QLabel(self.tab_platform)
        self.label_z_absolute_position.setObjectName(u"label_z_absolute_position")

        self.formLayout_3.setWidget(6, QFormLayout.LabelRole, self.label_z_absolute_position)

        self.lineEdit_z_absolute_position = QLineEdit(self.tab_platform)
        self.lineEdit_z_absolute_position.setObjectName(u"lineEdit_z_absolute_position")

        self.formLayout_3.setWidget(6, QFormLayout.FieldRole, self.lineEdit_z_absolute_position)

        self.lineEdit_z_relative_position = QLineEdit(self.tab_platform)
        self.lineEdit_z_relative_position.setObjectName(u"lineEdit_z_relative_position")

        self.formLayout_3.setWidget(8, QFormLayout.FieldRole, self.lineEdit_z_relative_position)

        self.label_z_relative_position = QLabel(self.tab_platform)
        self.label_z_relative_position.setObjectName(u"label_z_relative_position")

        self.formLayout_3.setWidget(8, QFormLayout.LabelRole, self.label_z_relative_position)

        self.verticalSpacer_9 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_3.setItem(1, QFormLayout.FieldRole, self.verticalSpacer_9)

        self.verticalSpacer_10 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_3.setItem(3, QFormLayout.FieldRole, self.verticalSpacer_10)

        self.verticalSpacer_11 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_3.setItem(5, QFormLayout.FieldRole, self.verticalSpacer_11)

        self.verticalSpacer_12 = QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.formLayout_3.setItem(7, QFormLayout.FieldRole, self.verticalSpacer_12)


        self.gridLayout_2.addLayout(self.formLayout_3, 0, 2, 1, 1)

        self.tabWidget.addTab(self.tab_platform, "")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.layoutWidget = QWidget(self.tab)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(0, 0, 258, 345))
        self.formLayout_9 = QFormLayout(self.layoutWidget)
        self.formLayout_9.setObjectName(u"formLayout_9")
        self.formLayout_9.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.layoutWidget)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_9.setWidget(0, QFormLayout.LabelRole, self.label_5)

        self.default_x_velocity_lineEdit = QLineEdit(self.layoutWidget)
        self.default_x_velocity_lineEdit.setObjectName(u"default_x_velocity_lineEdit")

        self.formLayout_9.setWidget(0, QFormLayout.FieldRole, self.default_x_velocity_lineEdit)

        self.verticalSpacer_31 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_9.setItem(1, QFormLayout.FieldRole, self.verticalSpacer_31)

        self.label_10 = QLabel(self.layoutWidget)
        self.label_10.setObjectName(u"label_10")

        self.formLayout_9.setWidget(2, QFormLayout.LabelRole, self.label_10)

        self.default_x_acceleration_lineEdit = QLineEdit(self.layoutWidget)
        self.default_x_acceleration_lineEdit.setObjectName(u"default_x_acceleration_lineEdit")

        self.formLayout_9.setWidget(2, QFormLayout.FieldRole, self.default_x_acceleration_lineEdit)

        self.verticalSpacer_32 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_9.setItem(3, QFormLayout.FieldRole, self.verticalSpacer_32)

        self.label_6 = QLabel(self.layoutWidget)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_9.setWidget(4, QFormLayout.LabelRole, self.label_6)

        self.default_x_position_lineEdit = QLineEdit(self.layoutWidget)
        self.default_x_position_lineEdit.setObjectName(u"default_x_position_lineEdit")

        self.formLayout_9.setWidget(4, QFormLayout.FieldRole, self.default_x_position_lineEdit)

        self.verticalSpacer_35 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_9.setItem(5, QFormLayout.FieldRole, self.verticalSpacer_35)

        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")

        self.formLayout_9.setWidget(6, QFormLayout.LabelRole, self.label)

        self.board_width_lineEdit = QLineEdit(self.layoutWidget)
        self.board_width_lineEdit.setObjectName(u"board_width_lineEdit")

        self.formLayout_9.setWidget(6, QFormLayout.FieldRole, self.board_width_lineEdit)

        self.verticalSpacer_37 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_9.setItem(7, QFormLayout.FieldRole, self.verticalSpacer_37)

        self.board_height_lineEdit = QLineEdit(self.layoutWidget)
        self.board_height_lineEdit.setObjectName(u"board_height_lineEdit")

        self.formLayout_9.setWidget(8, QFormLayout.FieldRole, self.board_height_lineEdit)

        self.label_3 = QLabel(self.layoutWidget)
        self.label_3.setObjectName(u"label_3")

        self.formLayout_9.setWidget(8, QFormLayout.LabelRole, self.label_3)

        self.layoutWidget1 = QWidget(self.tab)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(357, 10, 279, 381))
        self.formLayout_5 = QFormLayout(self.layoutWidget1)
        self.formLayout_5.setObjectName(u"formLayout_5")
        self.formLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_7 = QLabel(self.layoutWidget1)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_5.setWidget(0, QFormLayout.LabelRole, self.label_7)

        self.default_y_velocity_lineEdit = QLineEdit(self.layoutWidget1)
        self.default_y_velocity_lineEdit.setObjectName(u"default_y_velocity_lineEdit")

        self.formLayout_5.setWidget(0, QFormLayout.FieldRole, self.default_y_velocity_lineEdit)

        self.verticalSpacer_34 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_5.setItem(1, QFormLayout.FieldRole, self.verticalSpacer_34)

        self.label_8 = QLabel(self.layoutWidget1)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_5.setWidget(2, QFormLayout.LabelRole, self.label_8)

        self.default_y_acceleration_lineEdit = QLineEdit(self.layoutWidget1)
        self.default_y_acceleration_lineEdit.setObjectName(u"default_y_acceleration_lineEdit")

        self.formLayout_5.setWidget(2, QFormLayout.FieldRole, self.default_y_acceleration_lineEdit)

        self.verticalSpacer_36 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_5.setItem(3, QFormLayout.FieldRole, self.verticalSpacer_36)

        self.label_9 = QLabel(self.layoutWidget1)
        self.label_9.setObjectName(u"label_9")

        self.formLayout_5.setWidget(4, QFormLayout.LabelRole, self.label_9)

        self.default_y_position_lineEdit = QLineEdit(self.layoutWidget1)
        self.default_y_position_lineEdit.setObjectName(u"default_y_position_lineEdit")

        self.formLayout_5.setWidget(4, QFormLayout.FieldRole, self.default_y_position_lineEdit)

        self.verticalSpacer_33 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_5.setItem(5, QFormLayout.FieldRole, self.verticalSpacer_33)

        self.label_2 = QLabel(self.layoutWidget1)
        self.label_2.setObjectName(u"label_2")

        self.formLayout_5.setWidget(6, QFormLayout.LabelRole, self.label_2)

        self.movement_distance_lineEdit = QLineEdit(self.layoutWidget1)
        self.movement_distance_lineEdit.setObjectName(u"movement_distance_lineEdit")

        self.formLayout_5.setWidget(6, QFormLayout.FieldRole, self.movement_distance_lineEdit)

        self.verticalSpacer_38 = QSpacerItem(20, 38, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_5.setItem(7, QFormLayout.FieldRole, self.verticalSpacer_38)

        self.label_11 = QLabel(self.layoutWidget1)
        self.label_11.setObjectName(u"label_11")

        self.formLayout_5.setWidget(8, QFormLayout.LabelRole, self.label_11)

        self.lineEdit_crop_height = QLineEdit(self.layoutWidget1)
        self.lineEdit_crop_height.setObjectName(u"lineEdit_crop_height")

        self.formLayout_5.setWidget(8, QFormLayout.FieldRole, self.lineEdit_crop_height)

        self.confirm_button = QPushButton(self.layoutWidget1)
        self.confirm_button.setObjectName(u"confirm_button")

        self.formLayout_5.setWidget(10, QFormLayout.FieldRole, self.confirm_button)

        self.verticalSpacer_17 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.formLayout_5.setItem(9, QFormLayout.FieldRole, self.verticalSpacer_17)

        self.tabWidget.addTab(self.tab, "")
        self.tab_camera = QWidget()
        self.tab_camera.setObjectName(u"tab_camera")
        self.frame = QFrame(self.tab_camera)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(80, 50, 581, 44))
        self.frame.setFrameShape(QFrame.Box)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_gain = QLabel(self.frame)
        self.label_gain.setObjectName(u"label_gain")

        self.horizontalLayout.addWidget(self.label_gain)

        self.lineEdit_gain = QLineEdit(self.frame)
        self.lineEdit_gain.setObjectName(u"lineEdit_gain")

        self.horizontalLayout.addWidget(self.lineEdit_gain)

        self.horizontalSpacer_6 = QSpacerItem(60, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_6)


        self.horizontalLayout_3.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer_4 = QSpacerItem(60, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)

        self.label_times = QLabel(self.frame)
        self.label_times.setObjectName(u"label_times")

        self.horizontalLayout_2.addWidget(self.label_times)

        self.lineEdit_times = QLineEdit(self.frame)
        self.lineEdit_times.setObjectName(u"lineEdit_times")

        self.horizontalLayout_2.addWidget(self.lineEdit_times)


        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)

        self.label_4 = QLabel(self.tab_camera)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(21, 21, 72, 16))
        self.frame_2 = QFrame(self.tab_camera)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setGeometry(QRect(80, 180, 581, 44))
        self.frame_2.setFrameShape(QFrame.Box)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_light1 = QLabel(self.frame_2)
        self.label_light1.setObjectName(u"label_light1")

        self.horizontalLayout_4.addWidget(self.label_light1)

        self.lineEdit_light = QLineEdit(self.frame_2)
        self.lineEdit_light.setObjectName(u"lineEdit_light")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_light.sizePolicy().hasHeightForWidth())
        self.lineEdit_light.setSizePolicy(sizePolicy)

        self.horizontalLayout_4.addWidget(self.lineEdit_light)

        self.horizontalSpacer_8 = QSpacerItem(60, 20, QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_8)


        self.horizontalLayout_6.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer)

        self.label_light2 = QLabel(self.frame_2)
        self.label_light2.setObjectName(u"label_light2")

        self.horizontalLayout_5.addWidget(self.label_light2)

        self.comboBox_light_mode = QComboBox(self.frame_2)
        self.comboBox_light_mode.addItem("")
        self.comboBox_light_mode.addItem("")
        self.comboBox_light_mode.setObjectName(u"comboBox_light_mode")

        self.horizontalLayout_5.addWidget(self.comboBox_light_mode)

        self.horizontalSpacer_5 = QSpacerItem(60, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_5)


        self.horizontalLayout_6.addLayout(self.horizontalLayout_5)

        self.label_12 = QLabel(self.tab_camera)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(20, 140, 72, 16))
        self.frame_3 = QFrame(self.tab_camera)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setGeometry(QRect(80, 290, 581, 91))
        self.frame_3.setFrameShape(QFrame.Box)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.checkBox_Autosplit = QCheckBox(self.frame_3)
        self.checkBox_Autosplit.setObjectName(u"checkBox_Autosplit")
        self.checkBox_Autosplit.setTristate(False)

        self.horizontalLayout_9.addWidget(self.checkBox_Autosplit)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_10)

        self.label_15 = QLabel(self.frame_3)
        self.label_15.setObjectName(u"label_15")

        self.horizontalLayout_9.addWidget(self.label_15)

        self.spinBox_overlap_temp = QSpinBox(self.frame_3)
        self.spinBox_overlap_temp.setObjectName(u"spinBox_overlap_temp")
        self.spinBox_overlap_temp.setMaximum(1500)
        self.spinBox_overlap_temp.setValue(600)

        self.horizontalLayout_9.addWidget(self.spinBox_overlap_temp)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_11)

        self.label_20 = QLabel(self.frame_3)
        self.label_20.setObjectName(u"label_20")

        self.horizontalLayout_9.addWidget(self.label_20)

        self.spinBox_overlap_samp = QSpinBox(self.frame_3)
        self.spinBox_overlap_samp.setObjectName(u"spinBox_overlap_samp")
        self.spinBox_overlap_samp.setMaximum(1500)
        self.spinBox_overlap_samp.setValue(600)

        self.horizontalLayout_9.addWidget(self.spinBox_overlap_samp)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_7)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_13 = QLabel(self.frame_3)
        self.label_13.setObjectName(u"label_13")

        self.horizontalLayout_7.addWidget(self.label_13)

        self.spinBox_row = QSpinBox(self.frame_3)
        self.spinBox_row.setObjectName(u"spinBox_row")
        self.spinBox_row.setValue(3)

        self.horizontalLayout_7.addWidget(self.spinBox_row)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_7)

        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_14 = QLabel(self.frame_3)
        self.label_14.setObjectName(u"label_14")

        self.horizontalLayout_8.addWidget(self.label_14)

        self.spinBox_col = QSpinBox(self.frame_3)
        self.spinBox_col.setObjectName(u"spinBox_col")
        self.spinBox_col.setValue(2)

        self.horizontalLayout_8.addWidget(self.spinBox_col)


        self.horizontalLayout_9.addLayout(self.horizontalLayout_8)


        self.verticalLayout.addLayout(self.horizontalLayout_9)

        self.verticalSpacer_13 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.checkBox_SaveImage = QCheckBox(self.frame_3)
        self.checkBox_SaveImage.setObjectName(u"checkBox_SaveImage")

        self.horizontalLayout_12.addWidget(self.checkBox_SaveImage)

        self.horizontalSpacer_12 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer_12)

        self.label_17 = QLabel(self.frame_3)
        self.label_17.setObjectName(u"label_17")

        self.horizontalLayout_12.addWidget(self.label_17)

        self.spinBox_flag = QSpinBox(self.frame_3)
        self.spinBox_flag.setObjectName(u"spinBox_flag")
        self.spinBox_flag.setValue(10)

        self.horizontalLayout_12.addWidget(self.spinBox_flag)

        self.horizontalLayout_12.setStretch(2, 1)
        self.horizontalLayout_12.setStretch(3, 1)

        self.horizontalLayout_14.addLayout(self.horizontalLayout_12)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_14.addItem(self.horizontalSpacer_2)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_16 = QLabel(self.frame_3)
        self.label_16.setObjectName(u"label_16")

        self.horizontalLayout_11.addWidget(self.label_16)

        self.spinBox_slice = QSpinBox(self.frame_3)
        self.spinBox_slice.setObjectName(u"spinBox_slice")
        self.spinBox_slice.setMaximum(3200)
        self.spinBox_slice.setSingleStep(10)
        self.spinBox_slice.setValue(720)

        self.horizontalLayout_11.addWidget(self.spinBox_slice)

        self.horizontalLayout_11.setStretch(0, 1)
        self.horizontalLayout_11.setStretch(1, 1)

        self.horizontalLayout_14.addLayout(self.horizontalLayout_11)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_14.addItem(self.horizontalSpacer_3)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_18 = QLabel(self.frame_3)
        self.label_18.setObjectName(u"label_18")

        self.horizontalLayout_13.addWidget(self.label_18)

        self.spinBox_thres = QDoubleSpinBox(self.frame_3)
        self.spinBox_thres.setObjectName(u"spinBox_thres")
        self.spinBox_thres.setDecimals(3)
        self.spinBox_thres.setMaximum(1.000000000000000)
        self.spinBox_thres.setValue(0.050000000000000)

        self.horizontalLayout_13.addWidget(self.spinBox_thres)

        self.horizontalLayout_13.setStretch(0, 1)
        self.horizontalLayout_13.setStretch(1, 1)

        self.horizontalLayout_14.addLayout(self.horizontalLayout_13)


        self.verticalLayout.addLayout(self.horizontalLayout_14)


        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.label_19 = QLabel(self.tab_camera)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(20, 270, 72, 16))
        self.tabWidget.addTab(self.tab_camera, "")

        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)


        self.retranslateUi(Dialog)

        self.tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"\u7cfb\u7edf\u53c2\u6570\u8bbe\u7f6e", None))
        self.label_x_speed.setText(QCoreApplication.translate("Dialog", u"X\u8f74\u901f\u5ea6\uff1a", None))
        self.label_x_acceleration.setText(QCoreApplication.translate("Dialog", u"X\u8f74\u52a0\u901f\u5ea6\uff1a", None))
        self.label_x_distance.setText(QCoreApplication.translate("Dialog", u"X\u8f74\u8fd0\u52a8\u8ddd\u79bb\uff1a", None))
        self.label_x_absolute_position.setText(QCoreApplication.translate("Dialog", u"X\u8f74\u7edd\u5bf9\u4f4d\u7f6e\uff1a", None))
        self.label_x_relative_position.setText(QCoreApplication.translate("Dialog", u"X\u8f74\u76f8\u5bf9\u4f4d\u7f6e\uff1a", None))
        self.label_y_speed.setText(QCoreApplication.translate("Dialog", u"Y\u8f74\u901f\u5ea6\uff1a", None))
        self.label_y_acceleration.setText(QCoreApplication.translate("Dialog", u"Y\u8f74\u52a0\u901f\u5ea6\uff1a", None))
        self.label_y_distance.setText(QCoreApplication.translate("Dialog", u"Y\u8f74\u8fd0\u52a8\u8ddd\u79bb\uff1a", None))
        self.label_y_absolute_position.setText(QCoreApplication.translate("Dialog", u"Y\u8f74\u7edd\u5bf9\u4f4d\u7f6e\uff1a", None))
        self.label_y_relative_position.setText(QCoreApplication.translate("Dialog", u"Y\u8f74\u76f8\u5bf9\u4f4d\u7f6e\uff1a", None))
        self.label_z_speed.setText(QCoreApplication.translate("Dialog", u"Z\u8f74\u901f\u5ea6\uff1a", None))
        self.label_z_acceleration.setText(QCoreApplication.translate("Dialog", u"Z\u8f74\u52a0\u901f\u5ea6\uff1a", None))
        self.label_z_distance.setText(QCoreApplication.translate("Dialog", u"Z\u8f74\u8fd0\u52a8\u8ddd\u79bb\uff1a", None))
        self.label_z_absolute_position.setText(QCoreApplication.translate("Dialog", u"Z\u8f74\u7edd\u5bf9\u4f4d\u7f6e\uff1a", None))
        self.label_z_relative_position.setText(QCoreApplication.translate("Dialog", u"Z\u8f74\u76f8\u5bf9\u4f4d\u7f6e\uff1a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_platform), QCoreApplication.translate("Dialog", u"\u5e73\u53f0\u8bbe\u7f6e", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"X\u8f74\u901f\u5ea6\uff1a", None))
        self.label_10.setText(QCoreApplication.translate("Dialog", u"X\u8f74\u52a0\u901f\u5ea6\uff1a", None))
        self.label_6.setText(QCoreApplication.translate("Dialog", u"\u521d\u59cb\u4f4d\u7f6e\uff1a", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"\u677f\u5bbd\uff1a", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"\u677f\u9ad8\uff1a", None))
        self.label_7.setText(QCoreApplication.translate("Dialog", u"Y\u8f74\u901f\u5ea6\uff1a", None))
        self.label_8.setText(QCoreApplication.translate("Dialog", u"Y\u8f74\u52a0\u901f\u5ea6\uff1a", None))
        self.label_9.setText(QCoreApplication.translate("Dialog", u"\u521d\u59cb\u4f4d\u7f6e\uff1a", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"\u6a2a\u5411\u79fb\u52a8\u8ddd\u79bb\uff1a", None))
        self.label_11.setText(QCoreApplication.translate("Dialog", u"\u884c\u9ad8\uff1a", None))
        self.confirm_button.setText(QCoreApplication.translate("Dialog", u"\u786e\u5b9a", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Dialog", u"\u8fd0\u52a8\u8f68\u8ff9", None))
        self.label_gain.setText(QCoreApplication.translate("Dialog", u"\u76f8\u673a\u589e\u76ca\uff1a", None))
        self.label_times.setText(QCoreApplication.translate("Dialog", u"\u66dd\u5149\u65f6\u95f4\uff1a", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"\u76f8\u673a\u53c2\u6570\u8bbe\u7f6e", None))
        self.label_light1.setText(QCoreApplication.translate("Dialog", u"\u5149\u6e90\u4eae\u5ea6\uff1a", None))
        self.label_light2.setText(QCoreApplication.translate("Dialog", u"\u5149\u6e90\u6a21\u5f0f\uff1a", None))
        self.comboBox_light_mode.setItemText(0, QCoreApplication.translate("Dialog", u"\u5e38\u4eae", None))
        self.comboBox_light_mode.setItemText(1, QCoreApplication.translate("Dialog", u"\u5e38\u706d", None))

        self.label_12.setText(QCoreApplication.translate("Dialog", u"\u5149\u6e90\u53c2\u6570\u8bbe\u7f6e", None))
        self.checkBox_Autosplit.setText(QCoreApplication.translate("Dialog", u"\u624b\u52a8\u5207\u5206", None))
        self.label_15.setText(QCoreApplication.translate("Dialog", u"\u6a21\u677f\u91cd\u5408\u5ea6", None))
        self.label_20.setText(QCoreApplication.translate("Dialog", u"\u91c7\u6837\u91cd\u5408\u5ea6", None))
        self.label_13.setText(QCoreApplication.translate("Dialog", u"\u884c", None))
        self.label_14.setText(QCoreApplication.translate("Dialog", u"\u5217", None))
        self.checkBox_SaveImage.setText(QCoreApplication.translate("Dialog", u"\u5b58\u56fe", None))
        self.label_17.setText(QCoreApplication.translate("Dialog", u"\u6a21\u578b", None))
        self.label_16.setText(QCoreApplication.translate("Dialog", u"slice", None))
        self.label_18.setText(QCoreApplication.translate("Dialog", u"\u9608\u503c", None))
        self.label_19.setText(QCoreApplication.translate("Dialog", u"\u7b97\u6cd5\u53c2\u6570\u8bbe\u7f6e", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_camera), QCoreApplication.translate("Dialog", u"\u786c\u4ef6\u8bbe\u7f6e", None))
    # retranslateUi

