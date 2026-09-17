from modules import *
from widgets import *
from PIL.ImageShow import show
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from PySide6.QtWidgets import QMainWindow
from PySide6 import QtGui
from PySide6.QtCore import Qt
from PIL import Image
from correlation_analysis import cacu_cov
from tkinter import ttk
from CaculateHistogram import clustering
from numba import NumbaDeprecationWarning
from ivDataProcessUtils import IVDataProcessUtils, cacu_3fig

import sys
import new_function as nf
import os
import numpy as np
import CaculateHistogram as cacu
import tkinter as tk
import ctypes
import glob
import warnings
import umap
import multiprocessing
from multiprocessing import freeze_support

# 1. 编译使用 pyside6-uic main.ui -o ui_main.py,然后替换ui_main.py
# 2. pyside6与pyqt5只能用一种
# 3. pyside6-rcc resources.qrc -o resources_rc.py 更新图像
# pyinstaller -F --icon=icon.ico main.py  # 调试

# pyinstaller -Fw --icon=icon.ico --hidden-import=matplotlib.backends.backend_ps main.py  # !!!!!打包代码

# 如果没有某一个包的话替换掉
# 4. 打包 pyinstaller -Fw --icon=icon.ico main.py  # 推荐无终端版本
#        pyinstaller -Fw --icon=icon.ico main.py -D
#        增加图像在resources.qrc中添加并更新，注意格式
#        pyinstaller -Fw --add-data "C:\Users\Bruce\Desktop\Moel GUI1.0\images\images\logo.png;images" --icon=icon.ico main.py
#        --add-data 参数需要指定源文件的路径和目标文件夹的路径，并使用逗号 , 分隔它们
# 5. 使用tk创建多种窗口，会相互作用，找其他窗口类解决
# 6. 高斯拟合时数量过少会无法拟合，设置绘制阈值，增大迭代次数即可，

os.environ["QT_FONT_DPI"] = "96"  # FIX Problem for High DPI and Scale above 100%
# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
widgets = None


class MainWindow(QMainWindow):
    def __init__(self, show_startup_message=True):
        QMainWindow.__init__(self)
        # 初始化变量

        variables = [
            "divide_pic_peak", "hist_2d_data_peak", "image_path_2d_peak", "sorted_indices",
            "label", "redline", "hist_2d_data", "image_path_2d", "cluster_num_scan",
            "ch_fig_path", "image_path_For", "image_path_Reve", "hist_class", "iv_his_data",
            "dragPos", "single_fig_path", "min_max", "data_Reve", "merged_iv_cluster_image_path",
            "his_For", "data_For", "his", "data", "single_find_datas", "single_run_image_his_data",
            "single_find_merge_image", "single_run_image", "peak_pic", "cov_pic", "cov_data",
            "peak_run_data", "merged_image", "draw_mix_cluster", "divide_data", "data_save",
            "Primitive_data", "values", "object_data_iv_y", "object_data_iv_x", "his_Reve",
            "ch_score", "images", "datas", "his_data", "single_spin", "divide_pic", "Draw_iv_his_path",
            "meanCond_sourceFor_list", "meanCond_sourceReve_list", "list_integPSD", "list_gMean",
            "single_cut_png_path", "integPSD_png_path", "list_minN_png_path","conductance_fn_mean"
        ]

        # 使用循环将所有变量初始化为 None
        for var_name in variables:
            setattr(self, var_name, None)

        self.the_abs = True
        self.setAcceptDrops(True)
        self.accepted_urls = []  # 创建一个成员变量用于存储地址
        self.class_data = []
        self.file_path = ""
        self.file_path_2 = ""
        self.label_mode = False
        self.correlation = False

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX0
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "Molecular data processing -  GUI"
        description = "Data processing APP - Single-molecule research only."
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.toggleMenu(self, True)
        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_save.clicked.connect(self.buttonClick)
        widgets.btn_iv.clicked.connect(self.buttonClick)
        widgets.btn_psd.clicked.connect(self.buttonClick)

        # EXTRA LEFT BOX
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)

        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)

        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # SHOW APP
        # ///////////////////////////////////////////////////////////////

        self.showMinimized()
        if show_startup_message:
            cacu.signal_window('load complete!')
        self.showNormal()
        freeze_support()
        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)

        icon = QIcon(':images/images/images/PyDracula.png')  # 替换为您的图标文件路径
        self.setWindowIcon(icon)  # 设置任务栏图标
        nf.make_forder()  # 新建存图文件夹
        # button_clicked_thread(self.button_clicked, self.ui.progressBar, 0.1)

        button_slots = {  # 设置点击按钮
            self.ui.run: self.button_clicked,
            self.ui.save: self.save_button_clicked,
            self.ui.single_runpushButton: self.single_run,
            self.ui.single__savepushButton: self.single_save,
            self.ui.run_cov_pushButton: self.run_cov_cacu,
            self.ui.Redraw_Button: self.redraw,
            self.ui.open_file_button: self.open_file_button_clicked,
            self.ui.open_file_button_3: self.open_file_button_clicked,
            self.ui.open_file_button_4: self.open_file_button_clicked,
            self.ui.open_file_Button_6: self.open_file2_button_clicked,
            self.ui.open_file_button_iv: self.open_file3_button_clicked,
            self.ui.pushButton: self.CH_scan,
            self.ui.Cluster_run_iv: self.cluster_iv,
            self.ui.Draw_his_iv: self.Draw_iv_his,
            self.ui.saveall_iv: self.Save_iv,
            self.ui.choice_merge: self.little_window_for_merge,
            self.ui.mergeSaveButton2: self.little_window_for_rawData_merge,
            self.ui.peak_pushButton: self.peak_run,
            self.ui.divide_pushButton: self.divide_way,
            self.ui.div_clu_pushButton: self.dived_cluster,
            self.ui.save_Button_3: self.save_button_3clicked,
            self.ui.pushButton_3: self.save_button_clicked2,
            self.ui.Peak_Subsection: self.Peak_Subsection_clicked,
            self.ui.single_find_pushButton: self.single_find,
            self.ui.Correlation_find_pushButton: self.Correlation_find,
            self.ui.pushButton_2: self.single_preview,
            self.ui.run_file_button_iv: self.run_file_button_iv,
            self.ui.Single_pre_iv: self.iv_single_run,
            self.ui.pushButton_5: self.abs_sort,
            self.ui.Stable_sort_mean: self.stable_sort,
            self.ui.open_file_button_5: self.open_file_button_clicked,
            self.ui.Show_N_list: self.Show_N_list,
            self.ui.pushButton_7: self.Show_psd_list,
            self.ui.pushButton_4: self.pre_single_cut,
            self.ui.pushButton_6: self.psd_save,
        }

        for button, slot in button_slots.items():
            button.clicked.connect(slot)

        # 颜色设置
        spin_boxes = [
            self.ui.clusters, self.ui.single_spinBox, self.ui.red_set,
            self.ui.spinBox, self.ui.spinBox_2, self.ui.spinBox_3,
            self.ui.spinBox_4, self.ui.spinBox_5, self.ui.spinBox_6,
            self.ui.spinBox_7, self.ui.spinBox_8, self.ui.spinBox_9,
            self.ui.spinBox_10, self.ui.spinBox_11, self.ui.spinBox_12, self.ui.spinBox_13,
            self.ui.spinBox_14, self.ui.spinBox_15, self.ui.spinBox_16,
        ]

        for spin_box in spin_boxes:
            spin_box.setStyleSheet("QSpinBox { border: 1px solid rgb(53, 57, 63); }")

        self.ui.doubleSpinBox.setStyleSheet("QDoubleSpinBox { border: 1px solid rgb(53, 57, 63); }")
        self.ui.doubleSpinBox_16.setStyleSheet("QDoubleSpinBox { border: 1px solid rgb(53, 57, 63); }")
        for i in range(2, 15):
            getattr(self.ui, f"doubleSpinBox_{i}").setStyleSheet(
                "QDoubleSpinBox { border: 1px solid rgb(53, 57, 63); }")
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))  # 默认HOME页面亮起
        # 连接 stateChanged 信号到槽函数
        self.ui.ivMode.stateChanged.connect(self.iv_checkbox_state_changed)
        self.ui.checkBox.stateChanged.connect(self.label_checkbox_state_changed)
        # 然后调用函数来设置进度条的样式
        nf.set_progressbar_style(self.ui.progressBar, 4)
        nf.set_progressbar_style(self.ui.progressBar_3, 4)
        nf.set_progressbar_style(self.ui.progressBar_4, 20)
        nf.set_progressbar_style(self.ui.progressBar_5, 20)
        nf.set_progressbar_style(self.ui.progressBar_6, 20)

    def label_checkbox_state_changed(self):
        if self.ui.checkBox.isChecked():
            # 检测是否存在label
            if self.label is not None:
                self.label_mode = True
                self.ui.progressBar.setValue(100)
                cacu.signal_window('Label presence! Clicking on the cluster will be sorted by the label!')
            else:
                self.label_mode = False
                # 弹出小窗，请进行一次聚类！
                self.ui.progressBar.setValue(100)
                cacu.signal_window('no label data!  Please clustering!')
        else:
            self.label_mode = False

    def iv_checkbox_state_changed(self):
        # 处理复选框状态变化的操作
        print(self.ui.ivMode.isChecked())
        if self.ui.ivMode.isChecked():
            self.ui.comboBox_2.setCurrentIndex(0)
            self.ui.lineEdit_6.setText("-3.5")
            self.ui.lineEdit_7.setText("4")
            self.ui.lineEdit_5.setText("-1")
            self.ui.lineEdit_8.setText("1")
            self.ui.lineEdit_9.setText("-6")
            self.ui.lineEdit_10.setText("0")
            # 单曲线预处理
            self.ui.lineEdit_14.setText("-0.5")
            self.ui.lineEdit_16.setText("0.5")
            self.ui.lineEdit_15.setText("-2")
            self.ui.lineEdit_19.setText("2")

            self.ui.red_set.setValue(200)
        else:
            self.ui.comboBox_2.setCurrentIndex(2)
            self.ui.lineEdit_6.setText("-6.5")
            self.ui.lineEdit_7.setText("0.5")
            self.ui.lineEdit_5.setText("-0.2")
            self.ui.lineEdit_8.setText("1.2")
            self.ui.lineEdit_9.setText("0")
            self.ui.lineEdit_10.setText("1")
            # 单曲线预处理
            self.ui.lineEdit_14.setText("0")
            self.ui.lineEdit_16.setText("0.1")
            self.ui.lineEdit_15.setText("-6")
            self.ui.lineEdit_19.setText("0")
            self.ui.red_set.setValue(200)

    # 文件打开按钮
    def open_file_button_clicked(self):
        files, _ = QFileDialog.getOpenFileNames(self, caption='选择需要处理的.npz数据', filter="选择npz文件(*.npz)")
        if files:
            self.ui.lineEdit.setText(str(files[0]))
            self.ui.lineEdit_3.setText(str(files[0]))
            self.ui.lineEdit_4.setText(str(files[0]))
            self.ui.lineEdit_17.setText(str(files[0]))
            self.file_path = str(files[0])

    def open_file2_button_clicked(self):
        files, _ = QFileDialog.getOpenFileNames(self, caption='选择查找的.npz数据', filter="选择npz文件(*.npz)")
        if files:
            self.ui.lineEdit_2.setText(str(files[0]))
            self.file_path_2 = str(files[0])

    def open_file3_button_clicked(self):
        selected_directory = QFileDialog.getExistingDirectory(self, "选择文件夹", "默认路径")
        if selected_directory:
            self.ui.lineEdit_iv.setText(str(selected_directory))
            self.file_path = str(selected_directory)
            print('file_path_iv:' + self.file_path)

    # 提示窗口
    # BUTTONS CLICK
    # Post here your functions for clicked buttons
    # ///////////////////////////////////////////////////////////////
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))
            UIFunctions.resetStyle(self, btnName)
            # btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW WIDGETS PAGE
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            widgets.btn_widgets.setStyleSheet(UIFunctions.selectMenu(widgets.btn_widgets.styleSheet()))
            UIFunctions.resetStyle(self, btnName)
            # btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW NEW PAGE
        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page)  # SET PAGE
            widgets.btn_new.setStyleSheet(UIFunctions.selectMenu(widgets.btn_new.styleSheet()))
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            # btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))  # SELECT MENU

        if btnName == "btn_save":
            widgets.stackedWidget.setCurrentWidget(widgets.dl_page)  # SET PAGE
            widgets.btn_save.setStyleSheet(UIFunctions.selectMenu(widgets.btn_save.styleSheet()))
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            print("Save BTN clicked!")

        if btnName == "btn_iv":
            widgets.stackedWidget.setCurrentWidget(widgets.IV_page)  # SET PAGE
            widgets.btn_iv.setStyleSheet(UIFunctions.selectMenu(widgets.btn_iv.styleSheet()))
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            print("Save BTN clicked!")

        if btnName == "btn_psd":
            widgets.stackedWidget.setCurrentWidget(widgets.psd_page)  # SET PAGE
            widgets.btn_psd.setStyleSheet(UIFunctions.selectMenu(widgets.btn_psd.styleSheet()))
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            print("Save BTN clicked!")

        # PRINT BTN NAME
        print(f'Button "{btnName}" pressed!')

    # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

    def dragEnterEvent(self, a0: QtGui.QDragEnterEvent) -> None:
        # 判断有没有接受到内容
        if a0.mimeData().hasUrls():
            # 如果接收到内容了，就把它存在事件中
            urls = [url.toString() for url in a0.mimeData().urls()]
            self.accepted_urls = urls  # 存储接收到的地址
            a0.accept()
        else:
            # 没接收到内容就忽略
            a0.ignore()

    def dropEvent(self, a0: QtGui.QDropEvent) -> None:
        if a0:
            for i in a0.mimeData().urls():
                print(i.path())
                file_path = i.toLocalFile()
                self.ui.lineEdit.setText(file_path)
                self.ui.lineEdit_3.setText(file_path)
                self.ui.lineEdit_4.setText(file_path)
                self.ui.lineEdit_iv.setText(file_path)
                self.ui.lineEdit_17.setText(file_path)
                self.file_path = file_path

    def initUI(self, lbl, pic):
        if self.file_path != "" and lbl:
            pixmap = QtGui.QPixmap(pic)  # 按指定路径找到图片
            lbl.setPixmap(pixmap)  # 在label上显示图片
            lbl.setScaledContents(False)  # 让图片不自适应label大小
            self.show()

    def run_file_button_iv(self):
        """判断是不是接收到了IV文件夹
        如是则进行计算6种IV图像
        """
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_5.setValue(0)
            # 格式化和处理文件路径
            sourcePath = os.path.normpath(os.path.abspath(self.file_path))

            # 打印处理后的路径以检查
            print("Processed file path:", sourcePath)

            # 打开文件
            tdms_files = sorted(path for path in glob.glob(os.path.join(sourcePath, '*'))
                                if os.path.isfile(path) and path.lower().endswith('.tdms'))

            # 计算并打印原始数据文件的数量
            print("Number of TDMS files found:", len(tdms_files))

            if len(tdms_files) == 0:
                self.ui.progressBar_5.setValue(100)
                cacu.signal_window('no find iv data!Check whether the path contains illegal characters!!')
                self.showNormal()
                return

            biasVDataFor_list = []
            currentDataFor_list = []
            condDataFor_list = []
            currentData_sourceFor_list = []
            self.meanCond_sourceFor_list = []

            biasVDataReve_list = []
            currentDataReve_list = []
            condDataReve_list = []
            currentData_sourceReve_list = []
            self.meanCond_sourceReve_list = []

            threshold = self.ui.spinBox_10.value()
            bias_base = self.ui.doubleSpinBox_4.property("value")
            peakStart = self.ui.doubleSpinBox_5.property("value")
            peakEnd = self.ui.doubleSpinBox_6.property("value")
            butter_parameter = self.ui.spinBox_5.value()
            bin_2dhis = self.ui.spinBox_9.value()
            bin_1dhis = self.ui.spinBox_8.value()
            logI_min_max = [self.ui.doubleSpinBox_8.property("value"), self.ui.doubleSpinBox_7.property("value")]
            logdI_min_max = [self.ui.doubleSpinBox_10.property("value"), self.ui.doubleSpinBox_9.property("value")]
            logG_min_max = [self.ui.doubleSpinBox_11.property("value"), self.ui.doubleSpinBox_12.property("value")]
            I_min_max = [self.ui.doubleSpinBox_13.property("value") * 1e-6,
                         self.ui.doubleSpinBox_14.property("value") * 1e-6]
            sample_point = 1000
            # 创建进程池
            try:
                with multiprocessing.Pool(max(1, min(len(tdms_files), multiprocessing.cpu_count() - 1))) as pool:
                    results = []
                    # 遍历每个 tdms_file
                    for tdms_file in tdms_files:
                        result = pool.apply_async(func=IVDataProcessUtils.iv_process,
                                                  args=(tdms_file,),
                                                  kwds={"bias_base": bias_base,
                                                        "peakStart": peakStart,
                                                        "peakEnd": peakEnd,
                                                        "de_capicity": self.ui.checkBox_2.isChecked(),
                                                        "raise_on_error": True})

                        results.append(result)
                        print(f'Processing file: {tdms_file}')

                    failures = []
                    for tdms_file, result in zip(tdms_files, results):
                        try:
                            processed = result.get()
                        except Exception as exc:
                            failures.append(str(exc))
                            print(f"IV processing failed: {exc}")
                            continue
                        if processed and processed[0] and processed[3]:
                            # 解包返回值
                            (biasVDataFor, currentDataFor, condDataFor, biasVDataReve, currentDataReve, condDataReve,
                             currentData_sourceFor, currentData_sourceReve, meanCond_sourceFor_new,
                             meanCond_sourceReve) = processed

                            # 添加到相应的列表中
                            biasVDataFor_list.append(biasVDataFor)
                            currentDataFor_list.append(currentDataFor)
                            condDataFor_list.append(condDataFor)
                            currentData_sourceFor_list.append(currentData_sourceFor)
                            self.meanCond_sourceFor_list.append(meanCond_sourceFor_new)

                            biasVDataReve_list.append(biasVDataReve)
                            currentDataReve_list.append(currentDataReve)
                            condDataReve_list.append(condDataReve)
                            currentData_sourceReve_list.append(currentData_sourceReve)
                            self.meanCond_sourceReve_list.append(meanCond_sourceReve)

                        else:
                            failures.append(f"{os.path.basename(tdms_file)}: No complete forward/reverse scans")
                    if not biasVDataFor_list or not biasVDataReve_list:
                        details = "\n".join(failures[:5])
                        raise ValueError("No valid IV scans. Check bias and conductance limits.\n" + details)
                    if failures:
                        cacu.signal_window(f"Skipped {len(failures)} of {len(tdms_files)} files:\n" + "\n".join(failures[:5]))

                    # 打开两张图片。侧面写上名字，并合并绘制并显示
                    label1 = 'Forward(splicing)'
                    label2 = 'Reverse(continuous)'
                    folder_name = "png_images"

                    # 处理正扫
                    result1 = pool.apply_async(func=cacu_3fig,
                                               args=(biasVDataFor_list, currentDataFor_list, condDataFor_list,
                                                     currentData_sourceFor_list,),
                                               kwds={
                                                   # 传递其他参数
                                                   "butter_parameter": butter_parameter,
                                                   "sample_point": sample_point,
                                                   "label": label1,
                                                   "bin_2dhis": bin_2dhis,
                                                   "bin_1dhis": bin_1dhis,
                                                   "threshold": threshold,
                                                   "logI_min_max": logI_min_max,
                                                   "logdI_min_max": logdI_min_max,
                                                   "logG_min_max": logG_min_max,
                                                   "I_min_max": I_min_max,
                                                   "color_2d": self.ui.color_style_2d.currentText()
                                               })
                    result2 = pool.apply_async(func=cacu_3fig,
                                               args=(biasVDataReve_list, currentDataReve_list, condDataReve_list,
                                                     currentData_sourceReve_list,),
                                               kwds={
                                                   # 传递其他参数
                                                   "butter_parameter": butter_parameter,
                                                   "sample_point": sample_point,
                                                   "label": label2,
                                                   "bin_2dhis": bin_2dhis,
                                                   "bin_1dhis": bin_1dhis,
                                                   "threshold": threshold,
                                                   "logI_min_max": logI_min_max,
                                                   "logdI_min_max": logdI_min_max,
                                                   "logG_min_max": logG_min_max,
                                                   "I_min_max": I_min_max,
                                                   "color_2d": self.ui.color_style_2d.currentText()
                                               })

                    self.his_For, self.data_For = result1.get()
                    self.his_Reve, self.data_Reve = result2.get()

                    # 保存图像并显示在UI上
                    self.image_path_For = os.path.join(folder_name, f'IVfig{label1}.png')
                    self.image_path_Reve = os.path.join(folder_name, f'IVfig{label2}.png')
            except Exception as exc:
                self.his_For = self.his_Reve = self.data_For = self.data_Reve = None
                self.meanCond_sourceFor_list = []
                self.meanCond_sourceReve_list = []
                self.image_path_For = self.image_path_Reve = None
                self.ui.progressBar_5.setValue(0)
                cacu.signal_window(f"IV processing failed:\n{exc}")
                self.showNormal()
                return
            # 绘制大图
            fig, axes = plt.subplots(2, 1, figsize=(18, 8))
            for i in range(2):
                if i == 0:
                    image_path = self.image_path_For
                else:
                    image_path = self.image_path_Reve
                image = plt.imread(image_path)
                axes[i].imshow(image)
                axes[i].axis('off')

            # 调整子图间距
            plt.subplots_adjust(hspace=-0.3)
            # 显示图像叠加后的结果
            merged_image_path = os.path.join(folder_name, 'iv_merged_image.png')
            plt.savefig(merged_image_path, dpi=110, bbox_inches='tight')
            plt.close(fig)
            self.initUI(self.ui.label_iv, merged_image_path)

            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('Calculation complete!')
            self.showNormal()
            # 保存6种图片数据，以及处理后的数据至缓存
        else:
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('no data!  please input iv folder!')

    def Chose_data_iv(self):
        min_max = []
        Chose_data = self.ui.Chose_data_text.currentIndex()
        object_data_y = []
        object_data_x = []
        if Chose_data == 0:
            object_data_y = self.data_For[1]
            object_data_x = self.data_For[0]
            min_max = [self.ui.doubleSpinBox_8.property("value"), self.ui.doubleSpinBox_7.property("value")]

        elif Chose_data == 1:
            object_data_y = self.data_For[2]
            object_data_x = self.data_For[0]
            min_max = [self.ui.doubleSpinBox_10.property("value"), self.ui.doubleSpinBox_9.property("value")]

        elif Chose_data == 2:
            object_data_y = self.data_For[3]
            object_data_x = self.data_For[0]
            min_max = [self.ui.doubleSpinBox_11.property("value"), self.ui.doubleSpinBox_12.property("value")]

        elif Chose_data == 6:
            object_data_y = self.data_For[4]
            object_data_x = self.data_For[0]
            min_max = [self.ui.doubleSpinBox_13.property("value") * 1e-6,
                       self.ui.doubleSpinBox_14.property("value") * 1e-6]

        elif Chose_data == 3:
            object_data_y = self.data_Reve[1]
            object_data_x = self.data_Reve[0]
            min_max = [self.ui.doubleSpinBox_8.property("value"), self.ui.doubleSpinBox_7.property("value")]
        elif Chose_data == 4:
            object_data_y = self.data_Reve[2]
            object_data_x = self.data_Reve[0]
            min_max = [self.ui.doubleSpinBox_10.property("value"), self.ui.doubleSpinBox_9.property("value")]
        elif Chose_data == 5:
            object_data_y = self.data_Reve[3]
            object_data_x = self.data_Reve[0]
            min_max = [self.ui.doubleSpinBox_11.property("value"), self.ui.doubleSpinBox_12.property("value")]

        elif Chose_data == 7:
            object_data_y = self.data_Reve[4]
            object_data_x = self.data_Reve[0]
            min_max = [self.ui.doubleSpinBox_13.property("value") * 1e-6,
                       self.ui.doubleSpinBox_14.property("value") * 1e-6]
        self.object_data_iv_y = object_data_y
        self.object_data_iv_x = object_data_x
        self.min_max = min_max

    def cluster_iv(self):
        object_data = []
        if self.his_Reve:
            self.showMinimized()
            self.ui.progressBar_5.setValue(0)
            plt.close('all')
            self.Chose_data_iv()
            range_y = self.min_max
            hist_class = []
            # 聚类个数
            Cluster_num = self.ui.spinBox_7.value()
            # 2D bin
            bin_1dhis = self.ui.spinBox_8.value()
            bin_2dhis = self.ui.spinBox_9.value()
            threshold = self.ui.spinBox_10.value()
            #  选择logI, logDI, logG
            self.Chose_data_iv()
            object_data_y = self.object_data_iv_y
            object_data_x = self.object_data_iv_x

            # 选择预处理方法
            Chose_data_and_change_to_cluster = self.ui.Chose_data_and_change_to_cluster_text.currentIndex()
            if Chose_data_and_change_to_cluster == 0:
                object_data = object_data_y

            elif Chose_data_and_change_to_cluster == 1:
                object_data_y_array_cluster = []
                for i in range(len(object_data_y)):
                    object_data_y_array = object_data_y[i]
                    bin_counts2, _ = np.histogram(object_data_y_array, bins=50)  # 取目标范围
                    object_data_y_array_cluster.append(bin_counts2)

                object_data = object_data_y_array_cluster

            elif Chose_data_and_change_to_cluster == 2:
                his2d_clusters = []
                for i in range(len(object_data_y)):
                    # 二维直方图
                    H, _, _ = np.histogram2d(object_data_x[i], object_data_y[i], bins=50)
                    H = H.reshape(-1)  # 合并为一维数组
                    his2d_clusters.append(H)
                object_data = his2d_clusters

            # 选择聚类方法k/gauss
            iv_cluster_way = self.ui.iv_cluster_way_test.currentIndex()
            labels = cacu.chose_cluster(object_data, clusters_way=iv_cluster_way, n_clusters=Cluster_num)
            # 显示聚类
            # 获取数据点的分类索引
            class_indices = []  # 用于存储各个分类的索引
            class_object_data_y = []  # 用于存储各个分类的conductance
            class_object_data_x = []  # 用于存储各个分类的distance

            # 显示原始图像
            object_data_y = np.array(object_data_y)
            object_data_x = np.array(object_data_x)

            fig, axes = plt.subplots(1, Cluster_num + 1, figsize=(5 * (Cluster_num + 1), 3.7))
            axes[0].set_title(f'2D histogram')

            hist, extent, his2d_edg_list = cacu.plt_2dIV(object_data_x, object_data_y, bin_1dhis=bin_1dhis,
                                                         bin_2dhis=bin_2dhis,
                                                         threshold=threshold, range_y=range_y)
            im0 = axes[0].imshow(hist.T, origin='lower', extent=extent, aspect='auto',
                                 cmap=self.ui.color_style_2d.currentText())
            plt.colorbar(im0, ax=axes[0])
            axes[0].set_xlabel('V(V)')
            axes[0].set_ylabel(self.ui.Chose_data_text.currentText())
            axes[0].set_xlim(extent[0], extent[1])
            axes[0].set_ylim(range_y[0], range_y[1])

            for i in range(Cluster_num):
                class_i_indices = np.where(labels == i)[0]
                class_i_conductance = object_data_y[class_i_indices]
                class_i_distance = object_data_x[class_i_indices]
                result_rounded = round(len(class_i_distance) / len(object_data), 2)
                axes[i + 1].set_title(
                    f'2D histogram class :{i + 1} num = {len(class_i_distance)}, {[result_rounded * 100]}%')
                hist, extent, his2d_edg_list = cacu.plt_2dIV(class_i_distance, class_i_conductance, bin_1dhis=bin_1dhis,
                                                             bin_2dhis=bin_2dhis,
                                                             threshold=threshold, range_y=range_y)
                im0 = axes[i + 1].imshow(hist.T, origin='lower', extent=extent, aspect='auto',
                                         cmap=self.ui.color_style_2d.currentText())
                plt.colorbar(im0, ax=axes[i + 1])
                axes[i + 1].set_xlabel('V(V)')
                axes[i + 1].set_ylabel(self.ui.Chose_data_text.currentText())
                axes[i + 1].set_xlim(extent[0], extent[1])
                axes[i + 1].set_ylim(range_y[0], range_y[1])
                # 存储每次循环的结果
                class_indices.append(class_i_indices)
                class_object_data_y.append(class_i_conductance)
                class_object_data_x.append(class_i_distance)
                hist_class.append([his2d_edg_list, hist])

                # 创建图表
            self.hist_class = hist_class
            # 调整布局
            plt.tight_layout()
            # 显示图像叠加后的结果
            folder_name = "png_images"
            self.merged_iv_cluster_image_path = os.path.join(folder_name, 'merged_iv_cluster_image.png')
            plt.savefig(self.merged_iv_cluster_image_path, dpi=100)

            self.initUI(self.ui.label_iv, self.merged_iv_cluster_image_path)
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('Calculation complete!')
            self.showNormal()
        else:
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('no data!  please clink base_run!')

    def Draw_iv_his(self):

        if self.his_Reve:

            self.showMinimized()
            self.ui.progressBar_5.setValue(0)
            plt.close('all')
            self.Chose_data_iv()
            bin_1dhis = self.ui.spinBox_8.value()
            bin_2dhis = self.ui.spinBox_9.value()
            range_y = self.min_max
            threshold = self.ui.spinBox_10.value()

            object_data_y = self.object_data_iv_y
            object_data_x = self.object_data_iv_x
            self.Draw_iv_his_path, self.iv_his_data, self.redline, _ = cacu.plt_2dIV_1div(object_data_x, object_data_y,
                                                                                          bin_1dhis=bin_1dhis,
                                                                                          bin_2dhis=bin_2dhis,
                                                                                          threshold=threshold,
                                                                                          color_1d=self.ui.color_style_1d.currentText(),
                                                                                          color_2d=self.ui.color_style_2d.currentText(),
                                                                                          label=self.ui.Chose_data_text.currentText(),
                                                                                          range_y=range_y,
                                                                                          gauss_fit=self.ui.radioButton.isChecked(),
                                                                                          file_path2=self.file_path)

            self.initUI(self.ui.label_iv, self.Draw_iv_his_path)
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('Calculation complete!')
            self.showNormal()
        else:
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('no data!  please clink base_run!')

    def iv_single_run(self):

        if self.his_Reve is not None:
            self.ui.progressBar_5.setValue(0)
            plt.close('all')
            self.Chose_data_iv()

            Chose_data = self.ui.Chose_data_text.currentIndex()
            object_data_y = self.object_data_iv_y
            object_data_x = self.object_data_iv_x
            i = self.ui.spinBox_4.value() - 1
            self.single_fig_path = cacu.single_trance_iv(object_data_x, object_data_y, i=i, bins=50, )
            self.initUI(self.ui.label_iv, self.single_fig_path)
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('Calculation complete!')
        else:
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('no data!  please clink base_run!')

    def savefig(self, new_folder_name='base_data', temp_fig_path='Forward(splicing).png'):
        # 目标文件夹
        folder_path = self.file_path
        # 临时文件夹,去除绝对路径
        temp_fig_path = os.path.basename(temp_fig_path)
        folder_name = "png_images"
        # 要放入的新建文件夹
        base_data_folder = os.path.join(folder_path, new_folder_name)
        if not os.path.exists(base_data_folder):
            os.makedirs(base_data_folder)
        # 临时图像路径
        image_path = os.path.join(folder_name, temp_fig_path)
        # 保存路径
        save_path_image1 = os.path.join(base_data_folder, temp_fig_path)
        # 打开图像并保存
        image1 = Image.open(image_path)
        image1.save(save_path_image1)
        # 目标文件夹,去除绝对路径
        folder_path = os.path.dirname(save_path_image1)
        return folder_path

    def Save_iv(self):
        if self.his_Reve:
            self.showMinimized()
            self.ui.progressBar_5.setValue(0)
            Chose_data = self.ui.Chose_data_text.currentText()
            folder_path = self.savefig(new_folder_name='base_data_for',
                                       temp_fig_path=self.image_path_For)
            # self.hist_For : hist_logI, hist_logDI, hist_logG, hist_I, edg_logI, edg_logDI, edg_logG, edg_I

            # txt地址
            # 在data 中保存了I_data ,需完善新增I_data
            name_txt = ['hist_logI', 'hist_log_dI_dv', 'hist_logG', 'hist_I', 'edg_logI', 'edg_log_dI_dv', 'edg_logG',
                        'edg_I']
            for i in range(4):
                base_data_folder = os.path.join(folder_path, name_txt[i])
                if not os.path.exists(base_data_folder):
                    os.makedirs(base_data_folder)
                # Creating the full file path by joining 'base_data_folder' with a specific file name
                base_data_hist_logI_path = os.path.join(base_data_folder, f'base_data_{name_txt[i]}_For.txt')
                base_data_edg_logI_path = os.path.join(base_data_folder, f'base_data_{name_txt[i + 4]}_For.txt')
                nf.save_data_iv(histX=self.his_For[i],
                                hist_pathX=base_data_hist_logI_path, listX=self.his_For[i + 4],
                                list_pathX=base_data_edg_logI_path)
                # 在该文件夹中保存正扫数据.npz
                save_path = os.path.join(base_data_folder, f'base_data_{name_txt[i]}_For.npz')
                meanCond_sourceFor_list = [item for sublist in self.meanCond_sourceFor_list for item in sublist]

                # V_data_sample, logI_data_sample, I_data_sample_derivative_list, logG_sample
                np.savez(save_path, distance_array=self.data_For[0], conductance_array=self.data_For[i + 1],
                         length_array=meanCond_sourceFor_list,
                         additional_length=len(self.data_For[0][0]))

            folder_path = self.savefig(new_folder_name='base_data_Reve',
                                       temp_fig_path=self.image_path_Reve)  # self.data_Reve = [hist_logI, hist_logDI, hist_logG, edg_logI, edg_logDI, edg_logG]

            for i in range(4):
                base_data_folder = os.path.join(folder_path, name_txt[i])
                if not os.path.exists(base_data_folder):
                    os.makedirs(base_data_folder)
                base_data_hist_logI_path = os.path.join(base_data_folder, f'base_data_{name_txt[i]}_Reve.txt')
                base_data_edg_logI_path = os.path.join(base_data_folder, f'base_data_{name_txt[i + 4]}_Reve.txt')
                nf.save_data_iv(histX=self.his_Reve[i],
                                hist_pathX=base_data_hist_logI_path, listX=self.his_Reve[i + 4],
                                list_pathX=base_data_edg_logI_path)
                # .npz
                save_path = os.path.join(base_data_folder, f'base_data_{name_txt[i]}_Reve.npz')
                meanCond_sourceReve_list = [item for sublist in self.meanCond_sourceReve_list for item in sublist]
                np.savez(save_path, distance_array=self.data_Reve[0], conductance_array=self.data_Reve[i + 1],
                         length_array=meanCond_sourceReve_list,
                         additional_length=len(self.data_Reve[0][0]))

                # 在该文件夹中保存反扫数据.npz

        # if self.Draw_iv_his_path:  # 保存1dhis图像以及数据  # self.iv_his_data  = [1d_edge,1d_hist, his2d_edg_list, hist]  +Chose_data
        #     folder_path = self.savefig(new_folder_name=f'{Chose_data}_Draw_iv_his',
        #                                temp_fig_path=self.Draw_iv_his_path)
        #     name_txt = [f'{Chose_data}_1d_edge', f'{Chose_data}_1d_hist', f'{Chose_data}_his2d_edg_list',
        #                 f'{Chose_data}_hist']
        #     save_path_con = os.path.join(folder_path, f'{name_txt[1]}.txt')
        #     with open(save_path_con, 'w') as file:
        #         for item in self.iv_his_data[1]:
        #             line = f'{item[0]:<20}\t{item[1]:<20}\n'  # 使用格式化操作对齐每个元素
        #             file.write(line)
        #
        #     save_path_redLine_path = os.path.join(folder_path, f'meanLine.txt')
        #     with open(save_path_redLine_path, 'w') as file:
        #         for i in range(len(self.redline[0])):
        #             line = f'{self.redline[0][i]:<20}\t{self.redline[1][i]:<20}\n'  # 使用格式化操作对齐每个元素
        #             file.write(line)
        #     base_data_hist_logI_path = os.path.join(folder_path, f'{name_txt[3]}.txt')
        #     base_data_edg_logI_path = os.path.join(folder_path, f'{name_txt[2]}.txt')
        #     nf.save_data_iv(histX=self.iv_his_data[3],
        #                     hist_pathX=base_data_hist_logI_path, listX=self.iv_his_data[2],
        #                     list_pathX=base_data_edg_logI_path)
        #
        # if self.merged_iv_cluster_image_path:  # self.hist_class=hist_class.append([his2d_edg_list, hist]) + Chose_data
        #     folder_path = self.savefig(new_folder_name=f'{Chose_data}_iv_cluster',
        #                                temp_fig_path=self.merged_iv_cluster_image_path)
        #     for i in range(len(self.hist_class)):
        #         base_data_hist_logI_path = os.path.join(folder_path, f'{Chose_data}_hist_class_{i + 1}.txt')
        #         base_data_edg_logI_path = os.path.join(folder_path, f'{Chose_data}_edg_class_{i + 1}.txt')
        #         nf.save_data_iv(histX=self.hist_class[i][1],
        #                         hist_pathX=base_data_hist_logI_path, listX=self.hist_class[i][0],
        #                         list_pathX=base_data_edg_logI_path)
        #
        # if self.single_fig_path:
        #     self.savefig(new_folder_name='single_fig', temp_fig_path=self.single_fig_path)
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('successfully saved!!')
            self.showNormal()
        else:
            self.ui.progressBar_5.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def run_cov_cacu(self):
        plt.close('all')
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_3.setValue(0)

            bins = int(self.ui.lineEdit_12.text())
            vmax = float(self.ui.lineEdit_20.text())
            vmin = -vmax
            low_cut_1Dcon = float(self.ui.lineEdit_6.text())
            conductance_high_cut = float(self.ui.lineEdit_7.text())
            data = np.load(self.file_path)
            result, extent, self.cov_pic = cacu_cov(data, color_2d=self.ui.color_style_2d.currentText(), bins=bins,
                                                    vmin=vmin, vmax=vmax, low_cut_1Dcon=low_cut_1Dcon,
                                                    conductance_high_cut=conductance_high_cut,
                                                    file_path2=self.file_path)
            self.cov_data = result, extent
            self.initUI(self.ui.label_sub, self.cov_pic)
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('ok!')
            self.showNormal()
        else:
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def peak_run(self):
        if self.file_path != "":
            self.ui.progressBar_3.setValue(0)
            plt.close('all')
            self.get_parameters()
            values = self.values
            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            bins_1Dlen, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, his_1d_dis_0, his_1d_dis_1 = values
            peak = self.ui.doubleSpinBox.property("value")
            conductance_cut, distance_cut = cacu.new_data(distance, conductance, ivData=self.ui.ivMode.isChecked())
            his = nf.research_peak_length_his(distance_cut, conductance_cut, peak=peak)

            # peak_value, x, y, length, bins_1Dlen, len_his_list = cacu.calculate_1d_len(his, bins_1Dlen=bins_1Dlen)

            plt.hist(his, bins=bins_1Dlen, alpha=0.6, label='Data',
                     color=self.ui.color_style_1d.currentText(), range=[his_1d_dis_0, his_1d_dis_1])
            # plt.plot(x, y, 'r-')
            # plt.axvline(peak_value, color='g', linestyle='--')
            # 注明峰值横坐标
            # plt.text(peak_value, max(y), f'Peak: {peak_value:.2f}', color='r', ha='center', va='bottom')
            plt.title('1D distance')
            plt.xlabel('distance')
            plt.ylabel('count')
            plt.grid(True)  # 显示网格线
            folder_name = "png_images"
            image_path = os.path.join(folder_name, 'peak_run.png')
            plt.savefig(image_path)
            self.initUI(self.ui.label_sub, image_path)
            self.peak_pic = image_path
            self.peak_run_data = his
            hist, bin_edges = np.histogram(his, bins=bins_1Dlen, range=[his_1d_dis_0, his_1d_dis_1])
            self.peak_run_his =[hist, bin_edges]
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('ok')
        else:
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def divide_way(self):

        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_3.setValue(0)
            plt.close('all')
            self.get_parameters()
            values = self.values
            cacu_num = self.ui.spinBox_3.value()
            bins_1Dlen, bins_1Dcon, bins_2dhis_x, bins_2dhis_y, low_cut_1Dcon, high_cut_1Dcon, distance_0, distance_1, conductance_0, \
                conductance_1, single_1d_his_bin, single_2d_his_bin, low_cut_length, high_cut_length, low_cut_conductance, high_cut_conductance, _, _ = values
            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            length = data['length_array']

            self.divide_data, self.divide_pic, self.hist_2d_data = nf.draw_divide(conductance,
                                                                                  distance,
                                                                                  length,
                                                                                  bins_1Dcon=bins_1Dcon,
                                                                                  low_cut_1Dcon=low_cut_1Dcon,
                                                                                  high_cut_1Dcon=high_cut_1Dcon,
                                                                                  cacu_num=cacu_num,
                                                                                  color_1d=self.ui.color_style_1d.currentText(),
                                                                                  color_2d=self.ui.color_style_2d.currentText(),
                                                                                  red=self.ui.red_set.value(),
                                                                                  file_path2=self.file_path)

            self.initUI(self.ui.label_sub, self.divide_pic)
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('Calculation complete!')
            self.showNormal()
        else:
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def dived_cluster(self):
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_3.setValue(0)
            plt.close('all')
            self.get_parameters()
            values = self.values
            clusters_way = self.ui.comboBox.currentIndex()
            n_clusters = self.ui.spinBox_2.value()
            cacu_num = self.ui.spinBox_3.value()
            bins_1Dlen, bins_1Dcon, bins_2dhis_x, bins_2dhis_y, low_cut_1Dcon, high_cut_1Dcon, distance_0, distance_1, conductance_0, \
                conductance_1, single_1d_his_bin, single_2d_his_bin, low_cut_length, high_cut_length, low_cut_conductance, high_cut_conductance, _, _ = values
            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            length = data['length_array']

            data, _, cacu_time, _ = nf.divide(conductance, distance, length, bins_1Dcon=bins_1Dcon,
                                              low_cut_1Dcon=low_cut_1Dcon, high_cut_1Dcon=high_cut_1Dcon,
                                              cacu_num=cacu_num)

            # 特征提取
            object_data = self.object_data_prepro(distance, conductance, length, low_cut_length, high_cut_length,
                                                  single_1d_his_bin,
                                                  low_cut_conductance, high_cut_conductance, single_2d_his_bin)

            # 降维
            object_data = self.object_data_dimension_reduction(object_data)

            self.draw_mix_cluster = nf.cacu_mix_his(data, cacu_time, object_data, bins_1Dcon=bins_1Dcon,
                                                    low_cut_1Dcon=low_cut_1Dcon,
                                                    high_cut_1Dcon=high_cut_1Dcon,
                                                    n_clusters=n_clusters,
                                                    clusters_way=clusters_way, cacu_num=cacu_num,
                                                    len_conductance=len(conductance))
            self.initUI(self.ui.label_sub, self.draw_mix_cluster)
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('Calculation complete!')
            self.showNormal()
        else:
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def single_preview(self):
        self.ui.progressBar_4.setValue(0)
        if self.file_path_2 != "":

            self.show_single(self.file_path_2)
            self.ui.progressBar_4.setValue(100)
        elif self.file_path != "":

            self.show_single(self.file_path)
            self.ui.progressBar_4.setValue(100)
        else:
            self.ui.progressBar_4.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def show_single(self, file_path):
        self.ui.label_42.clear()
        plt.close('all')
        data = np.load(file_path)
        conductance = data['conductance_array']
        distance = data['distance_array']
        i = self.ui.spinBox.value() - 1
        if self.ui.ivMode.isChecked():
            conductance_cut = conductance[i]
            distance_cut = distance[i]
        else:
            conductance_cut, min_index = cacu.single_new_data(conductance[i])
            distance_cut = distance[i][:min_index]
        plt.plot(distance_cut, conductance_cut)
        plt.title('Find his friend')
        plt.xlabel('distance')
        plt.ylabel(f'conductance number : {i + 1}')
        folder_name = "png_images"
        image_path = os.path.join(folder_name, 'cacu_cov.png')
        plt.savefig(image_path)
        self.initUI(self.ui.label_42, image_path)

    def single_find(self):
        self.ui.label_42.clear()
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_4.setValue(0)
            if self.file_path_2 != "":
                data_2 = np.load(self.file_path_2)
                conductance_2 = data_2['conductance_array']
                distance_2 = data_2['distance_array']
            else:
                conductance_2 = None
                distance_2 = None

            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            length = data['length_array']
            plt.close('all')
            self.get_parameters()
            values = self.values
            bins_1Dlen, bins_1Dcon, bins_2dhis_x, bins_2dhis_y, low_cut_1Dcon, high_cut_1Dcon, distance_0, distance_1, conductance_0, \
                conductance_1, single_1d_his_bin, single_2d_his_bin, low_cut_length, high_cut_length, low_cut_conductance, high_cut_conductance, _, _ = values
            red = self.ui.red_set.value()

            datas = nf.class_for_one(distance, conductance, length,
                                     i=self.ui.spinBox.value() - 1,
                                     low_cut_1Dcon=low_cut_conductance,
                                     similar=self.ui.doubleSpinBox_3.property("value"),
                                     bins=100, conductance_high_cut=high_cut_conductance,
                                     p=self.ui.spinBox_6.value(),
                                     conductance_2=conductance_2, distance_2=distance_2,
                                     correlation=self.correlation)

            self.single_run_image_his_data, self.single_run_image, self.single_find_merge_image = cacu.draw_merge_figure(
                datas, distance,
                conductance, length, 2,
                bins_1Dcon=bins_1Dcon,
                low_cut_1Dcon=low_cut_1Dcon,
                high_cut_1Dcon=high_cut_1Dcon,
                bins_2dhis_x=bins_2dhis_x,
                bins_2dhis_y=bins_2dhis_y,
                red=red,
                distance_0=distance_0,
                distance_1=distance_1,
                conductance_0=conductance_1,
                conductance_1=conductance_0,
                bins_1Dlen=bins_1Dlen,
                color_1d=self.ui.color_style_1d.currentText(),
                color_2d=self.ui.color_style_2d.currentText(),
                ivData=self.ui.ivMode.isChecked())
            self.single_find_datas = datas
            self.initUI(self.ui.label_42, self.single_find_merge_image)
            self.ui.progressBar_4.setValue(100)
            cacu.signal_window('Data screening successfully')
            self.showNormal()
        else:
            self.ui.progressBar_4.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def Correlation_find(self):
        self.correlation = True
        self.single_find()
        self.correlation = False

    def stable_sort(self):
        self.ui.label_42.clear()
        self.ui.progressBar_4.setValue(0)
        if self.file_path != "":
            plt.close('all')
            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            i = self.ui.spinBox.value()
            # 原始单个曲线
            self.distance_i = distance[i]
            self.conductance_i = conductance[i]
            # 原始的单个曲线的线性电导
            self.conductance_new = 10 ** conductance[i] * distance[i] / np.abs(distance[i])
            # 原始单个曲线的FN x lim= -20 20
            self.conductance_fn = np.log(abs(self.conductance_new) / (distance[i] ** 2))  # Corrected V**2
            self.distance_new = 1 / distance[i]
            # 求均值
            self.conductance_mean = np.mean(conductance, axis=0)
            # 求线性
            self.conductance_mean_new = 10 ** self.conductance_mean * distance[i] / np.abs(distance[i])
            #求fn
            self.conductance_fn_mean = np.log(abs(self.conductance_mean_new) / (distance[i] ** 2))  # Corrected V**2

            # 创建 2 行 3 列子图
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.ravel()  # 将子图数组展平成一维数组以便迭代访问

            # 子图 1: 原始单个曲线 (distance vs. conductance)
            axes[0].plot(self.distance_i, self.conductance_i, label='Raw Conductance Curve', color='blue')
            axes[0].set_title('Raw Single Curve')
            axes[0].set_xlabel('Distance')
            axes[0].set_ylabel('Conductance')
            axes[0].legend()
            axes[0].grid(True)

            # 子图 2: 原始单个曲线的线性电导
            axes[1].plot(self.distance_i, self.conductance_new, label='Linear Conductance', color='green')
            axes[1].set_title('Linear Conductance (Single)')
            axes[1].set_xlabel('Distance')
            axes[1].set_ylabel('Linear Conductance')
            axes[1].legend()
            axes[1].grid(True)

            # 子图 3: 原始单个曲线的 FN 图 (只连接 lim 内点)
            xlim_min, xlim_max = -20, 20

            axes[2].scatter(self.distance_new, self.conductance_fn, color='red', label='FN Points')
            axes[2].set_xlim(xlim_min, xlim_max)
            axes[2].set_title('FN Plot (Single)')
            axes[2].set_xlabel('1/V')
            axes[2].set_ylabel('ln(I/V^2)')
            axes[2].legend()
            axes[2].grid(True)

            # 子图 4: 均值曲线 (distance vs. conductance_mean)
            axes[3].plot(self.distance_i, self.conductance_mean, label='Mean Conductance Curve', color='purple')
            axes[3].set_title('Mean Curve')
            axes[3].set_xlabel('voltage')
            axes[3].set_ylabel('Conductance')
            axes[3].legend()
            axes[3].grid(True)

            # 子图 5: 均值的线性电导
            axes[4].plot(self.distance_i,  self.conductance_mean_new, label='Linear Conductance (Mean)', color='orange')
            axes[4].set_title('Linear Conductance (Mean)')
            axes[4].set_xlabel('voltage')
            axes[4].set_ylabel('Linear Conductance')
            axes[4].legend()
            axes[4].grid(True)

            # 子图 6: 均值的 FN 图 (只连接 lim 内点)
            axes[5].scatter(self.distance_new,  self.conductance_fn_mean, color='red', label='FN Mean Points')
            axes[5].set_xlim(xlim_min, xlim_max)
            axes[5].set_title('FN Plot (Mean)')
            axes[5].set_xlabel('1/V')
            axes[5].set_ylabel('ln(I/V^2)')
            axes[5].legend()
            axes[5].grid(True)

            # 调整布局
            plt.tight_layout()

            folder_name = "png_images"
            self.FN_image_path = os.path.join(folder_name, f'FN.png')

            plt.savefig(self.FN_image_path, dpi=100)

            self.initUI(self.ui.label_42, self.FN_image_path)
            self.ui.progressBar_4.setValue(100)


    def abs_sort(self):
        self.ui.label_42.clear()
        self.ui.progressBar_4.setValue(0)
        if self.file_path != "":
            self.showMinimized()
            plt.close('all')
            self.get_parameters()
            values = self.values
            bins_1Dlen, bins_1Dcon, bins_2dhis_x, bins_2dhis_y, low_cut_1Dcon, high_cut_1Dcon, distance_0, distance_1, conductance_0, \
                conductance_1, single_1d_his_bin, single_2d_his_bin, low_cut_length, high_cut_length, low_cut_conductance, high_cut_conductance, _, _ = values
            data = np.load(self.file_path)
            object_data_y = data['conductance_array']
            object_data_x = data['distance_array']

            range_y = [conductance_0, conductance_1]
            range_x = [distance_0, distance_1]
            threshold = self.ui.red_set.value()

            the_abs = self.the_abs
            self.Draw_iv_his_path, self.iv_his_data, self.redline, self.sorted_indices = cacu.plt_2dIV_1div(
                object_data_x, object_data_y,
                bin_1dhis=bins_2dhis_x,
                bin_2dhis=bins_2dhis_y,
                threshold=threshold,
                color_1d=self.ui.color_style_1d.currentText(),
                color_2d=self.ui.color_style_2d.currentText(),
                label="the sort data",
                range_y=range_y, range_x=range_x,
                the_abs=the_abs,
                sort_range=self.ui.doubleSpinBox_3.property(
                    "value"),
                gauss_fit=self.ui.radioButton.isChecked(), file_path2=self.file_path)

            self.initUI(self.ui.label_42, self.Draw_iv_his_path)
            self.ui.progressBar_4.setValue(100)
            cacu.signal_window('Calculation complete!')
            self.showNormal()
        else:
            self.ui.progressBar_4.setValue(100)
            cacu.signal_window('no data!  please input it!!')

    def single_run(self):
        self.ui.label_png2.clear()
        if self.file_path != "":
            self.ui.progressBar.setValue(0)
            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            length = data['length_array']

            # 对图表所有的参数进行读取与定义
            single_1d_his_bin = int(self.ui.lineEdit_18.text())
            single_2d_his_bin = single_1d_his_bin

            low_cut_length = float(self.ui.lineEdit_14.text())
            high_cut_length = float(self.ui.lineEdit_16.text())
            low_cut_conductance = float(self.ui.lineEdit_15.text())
            high_cut_conductance = float(self.ui.lineEdit_19.text())

            i = self.ui.single_spinBox.value() - 1
            self.single_spin, self.data_save = cacu.single_trance(distance, conductance, length, i, low_cut_length,
                                                                  high_cut_length,
                                                                  low_cut_conductance, high_cut_conductance,
                                                                  single_1d_his_bin=single_1d_his_bin,
                                                                  single_2d_his_bin=single_2d_his_bin,
                                                                  color_1d=self.ui.color_style_1d.currentText(),
                                                                  color_2d=self.ui.color_style_2d.currentText(),
                                                                  ivData=self.ui.ivMode.isChecked())

            self.initUI(self.ui.label_png2, self.single_spin)
            plt.close('all')
            self.ui.progressBar.setValue(100)
            cacu.signal_window('single trance has updated')
        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def single_save(self):

        if self.single_spin is not None:
            self.ui.progressBar.setValue(0)
            file_path = self.file_path
            folder_path = os.path.dirname(file_path)  # 提取single trance所在路径

            single_folder = os.path.join(folder_path, f'single_trance_{self.ui.spinBox.value()}')
            os.makedirs(single_folder, exist_ok=True)

            save_path = os.path.join(single_folder, f'single_trance_{self.ui.spinBox.value()}.png')
            save_path = os.path.normpath(save_path)
            # 打开图像文件
            image = Image.open(self.single_spin)
            # 保存图像到指定路径
            image.save(save_path)

            # 保存直方图，创建文件路径
            save_path_con = os.path.join(single_folder, 'con_1d_his.txt')
            save_path_2d_hist = os.path.join(single_folder, 'save_path_2d_hist.txt')
            save_path_2d_edges = os.path.join(single_folder, 'save_path_2d_edges.txt')

            with open(save_path_con, 'w') as file:
                for item in self.data_save[0]:
                    line = f'{item[0]:<20}\t{item[1]:<20}\n'  # 使用格式化操作对齐每个元素
                    file.write(line)
            # 保存下降沿后的单个曲线
            save_path_Line_path = os.path.join(single_folder, f'single_trace_for_oringin.txt')
            conductance_cut, min_index = cacu.single_new_data(self.data_save[3][1])
            distance_cut = self.data_save[3][0][:min_index]
            # 不进行分割
            # conductance_cut = self.data_save[3][1]
            # distance_cut = self.data_save[3][0]

            with open(save_path_Line_path, 'w') as file:
                for i in range(len(distance_cut)):
                    line = f'{distance_cut[i]:<20}\t{conductance_cut[i]}\n'  # 使用格式化操作对齐每个元素
                    file.write(line)

            nf.save_data_iv(listX=self.data_save[2], list_pathX=save_path_2d_edges, histX=self.data_save[1],
                            hist_pathX=save_path_2d_hist)

            self.ui.progressBar.setValue(100)
            cacu.signal_window('single trance has saved')
        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def save_button_3clicked(self):
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_3.setValue(0)
            file_path = self.file_path
            folder_path = os.path.dirname(file_path)  # 提取single trance所在路径
            # 新建文件夹保存所有数据
            # time分割文件(时间分割/时间聚类/峰时间分割）
            if self.divide_pic_peak is not None:
                self.ui.progressBar_3.setValue(10)
                divide_data_folder = os.path.join(folder_path, f'separated_peak_data')
                os.makedirs(divide_data_folder, exist_ok=True)

                save_path_image = os.path.join(divide_data_folder, f'separated_data_peak.png')
                save_path_image = os.path.normpath(save_path_image)
                # 存图
                image = Image.open(self.divide_pic_peak)
                image.save(save_path_image)
                # 存his  hist_2d_data

                divide_path = os.path.join(divide_data_folder, 'divide_time_hist_peak.txt')
                with open(divide_path, 'w') as file:
                    for row in self.hist_2d_data_peak:
                        formatted_row = [f'{element:<6.3f}' for element in row]  # 使用格式化字符串对齐每个元素
                        line = ' '.join(formatted_row) + '\n'
                        file.write(line)

            if self.divide_data is not None:

                divide_data_folder = os.path.join(folder_path, f'separated_data')
                os.makedirs(divide_data_folder, exist_ok=True)

                save_path_image = os.path.join(divide_data_folder, f'separated_data.png')
                save_path_image = os.path.normpath(save_path_image)
                # 存图
                image = Image.open(self.divide_pic)
                image.save(save_path_image)
                # 存his  hist_2d_data

                divide_path = os.path.join(divide_data_folder, 'divide_time_hist.txt')
                with open(divide_path, 'w') as file:
                    for row in self.hist_2d_data[0]:
                        formatted_row = [f'{element:<6.3f}' for element in row]  # 使用格式化字符串对齐每个元素
                        line = ' '.join(formatted_row) + '\n'
                        file.write(line)

                # 存数据
                datas = self.divide_data
                for i in range(len(datas)):
                    save_path = os.path.join(divide_data_folder, f'separated_data{i + 1}.npz')
                    np.savez(save_path, distance_array=datas[i][0], conductance_array=datas[i][1],
                             length_array=datas[i][2], additional_length=len(datas[i][0]))

            # time时间聚类
            if self.draw_mix_cluster is not None:
                separated_data_clustering_folder = os.path.join(folder_path, 'separated_data_clustering')
                os.makedirs(separated_data_clustering_folder, exist_ok=True)

                save_path_image = os.path.join(separated_data_clustering_folder, 'separated_data_clustering.png')
                save_path_image = os.path.normpath(save_path_image)
                # 存图
                image = Image.open(self.draw_mix_cluster)
                image.save(save_path_image)

            # cov文件
            if self.cov_data is not None:
                cov_data_folder = os.path.join(folder_path, 'covariance')
                os.makedirs(cov_data_folder, exist_ok=True)
                cov_data_image = os.path.join(cov_data_folder, 'covariance.png')
                cov_data_image = os.path.normpath(cov_data_image)
                # 存图
                image = Image.open(self.cov_pic)
                image.save(cov_data_image)
                # 存数据
                result, extent = self.cov_data
                result_path = os.path.join(cov_data_folder, 'cov_result.txt')
                extent_path = os.path.join(cov_data_folder, 'cov_extent.txt')
                with open(extent_path, 'w') as file:
                    for item in extent:
                        line = f'{item:<10}\n'
                        file.write(line)
                with open(result_path, 'w') as file:
                    for row in result:
                        formatted_row = [f'{element:<6.3f}' for element in row]  # 使用格式化字符串对齐每个元素
                        line = ' '.join(formatted_row) + '\n'
                        file.write(line)

            if self.peak_run_data is not None:
                cov_data_folder = os.path.join(folder_path, f'peak_data')
                os.makedirs(cov_data_folder, exist_ok=True)
                cov_data_image = os.path.join(cov_data_folder, f'peak.png')
                cov_data_image = os.path.normpath(cov_data_image)
                # 存图
                image = Image.open(self.peak_pic)
                image.save(cov_data_image)
                # 存数据
                result = self.peak_run_data
                result_path = os.path.join(cov_data_folder, 'peak_result.txt')
                with open(result_path, 'w') as file:
                    for item in result:
                        line = f'{item:<10}\n'
                        file.write(line)
                # 存his
                hist, bin_edges = self.peak_run_his
                his_txt_path = os.path.join(cov_data_folder, 'his.txt')

                with open(his_txt_path, 'w') as file:
                    for i in range(len(hist)):
                        line = f'{bin_edges[i]}\t{hist[i]}\n'  # 使用制表符分隔
                        file.write(line)

            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('successfully saved!!')
            self.showNormal()
        else:
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def get_parameters(self):

        length_bin = int(self.ui.lineEdit_13.text())
        con_bin = int(self.ui.lineEdit_12.text())
        dis_bin = int(self.ui.lineEdit_11.text())
        con_min = float(self.ui.lineEdit_6.text())
        con_max = float(self.ui.lineEdit_7.text())
        dis_min = float(self.ui.lineEdit_5.text())
        dis_max = float(self.ui.lineEdit_8.text())

        # 单曲线预处理
        sin_dis_bin = int(self.ui.lineEdit_13.text())
        sin_con_bin = sin_dis_bin
        sin_dis_min = float(self.ui.lineEdit_14.text())
        sin_dis_max = float(self.ui.lineEdit_16.text())
        sin_con_min = float(self.ui.lineEdit_15.text())
        sin_con_max = float(self.ui.lineEdit_19.text())

        length_min = float(self.ui.lineEdit_9.text())
        length_max = float(self.ui.lineEdit_10.text())

        values = [length_bin, con_bin, dis_bin, con_bin,
                  con_min, con_max,
                  dis_min, dis_max, con_min,
                  con_max, sin_dis_bin,
                  sin_con_bin, sin_dis_min,
                  sin_dis_max, sin_con_min,
                  sin_con_max, length_min,
                  length_max]
        self.values = values

    def Peak_Subsection_clicked(self):

        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_3.setValue(0)
            plt.close('all')
            self.get_parameters()
            values = self.values
            cacu_num = self.ui.spinBox_3.value()
            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            conductance, distance = cacu.new_data(distance, conductance, ivData=self.ui.ivMode.isChecked())

            bins_1Dlen, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, his_1d_dis_0, his_1d_dis_1 = values
            peak = self.ui.doubleSpinBox.property("value")
            self.divide_pic_peak, self.hist_2d_data_peak = nf.peak_draw_divide(conductance,
                                                                               distance,
                                                                               bins_1Dlen=bins_1Dlen,
                                                                               his_1d_dis_0=his_1d_dis_0,
                                                                               his_1d_dis_1=his_1d_dis_1,
                                                                               cacu_num=cacu_num,
                                                                               peak=peak,
                                                                               color_1d=self.ui.color_style_1d.currentText(),
                                                                               color_2d=self.ui.color_style_2d.currentText(),
                                                                               red=self.ui.red_set.value(),
                                                                               file_path2=self.file_path)


            self.initUI(self.ui.label_sub, self.divide_pic_peak)
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('Calculation complete!')
            self.showNormal()
        else:
            self.ui.progressBar_3.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def redraw(self):
        self.ui.label_png2.clear()
        if self.his_data is not None:
            self.showMinimized()
            self.ui.progressBar.setValue(0)
            self.get_parameters()
            values = self.values
            clusters = self.ui.clusters.property("value")
            datas = self.datas
            red = self.ui.red_set.value()
            data = np.load(self.file_path)

            # 对图表所有的参数进行读取与定义
            bins_1Dlen, bins_1Dcon, bins_2dhis_x, bins_2dhis_y, low_cut_1Dcon, high_cut_1Dcon, distance_0, distance_1, conductance_0, \
                conductance_1, single_1d_his_bin, single_2d_his_bin, low_cut_length, high_cut_length, low_cut_conductance, high_cut_conductance, his_1d_dis_0, his_1d_dis_1 = values

            conductance = data['conductance_array']
            distance = data['distance_array']
            length = data['length_array']

            self.his_data, self.images, self.merged_image = cacu.draw_merge_figure(datas, distance,
                                                                                   conductance,
                                                                                   length,
                                                                                   n_clusters=clusters,
                                                                                   bins_1Dcon=bins_1Dcon,
                                                                                   low_cut_1Dcon=low_cut_1Dcon,
                                                                                   high_cut_1Dcon=high_cut_1Dcon,
                                                                                   bins_2dhis_x=bins_2dhis_x,
                                                                                   bins_2dhis_y=bins_2dhis_y,
                                                                                   distance_0=distance_0,
                                                                                   distance_1=distance_1,
                                                                                   conductance_0=conductance_1,
                                                                                   conductance_1=conductance_0,
                                                                                   bins_1Dlen=bins_1Dlen,
                                                                                   color_1d=self.ui.color_style_1d.currentText(),
                                                                                   color_2d=self.ui.color_style_2d.currentText(),
                                                                                   red=red,
                                                                                   ivData=self.ui.ivMode.isChecked(),
                                                                                   his_1d_dis_0=his_1d_dis_0,
                                                                                   his_1d_dis_1=his_1d_dis_1)

            # 更新数据
            # 合并图
            self.initUI(self.ui.label_png2, self.merged_image)
            self.ui.progressBar.setValue(100)
            cacu.signal_window('Redrawn successfully')
            self.showNormal()
        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def save_button_clicked(self):
        if self.his_data is not None:
            self.showMinimized()
            self.ui.progressBar.setValue(0)
            datas = self.datas
            file_path = self.file_path
            folder_path = os.path.dirname(file_path)  # 提取single trance所在路径
            clusters = self.ui.clusters.property("value")
            # 保存图片至folder_path/fig文件夹，保存数据至folder_path/data文件夹
            his_data = self.his_data
            # 创建"kmeans"文件夹
            kmeans_folder = os.path.join(folder_path, 'kmeans')
            os.makedirs(kmeans_folder, exist_ok=True)

            # 保存分类结果
            for i in range(clusters + 1):
                # 创建类别文件夹
                class_folder_i = os.path.join(kmeans_folder, f'class_{i}')
                # 确保保存文件夹存在
                os.makedirs(class_folder_i, exist_ok=True)
                file_path = self.images[i]
                save_path = os.path.join(class_folder_i, f'image_{i}.png')
                # 打开图像文件
                images = Image.open(file_path)
                # 保存图像到指定路径
                images.save(save_path)
                # 保存直方图
                save_path_con = os.path.join(class_folder_i, f'con_1d_his_{i}.txt')
                save_path_len = os.path.join(class_folder_i, f'len_1d_his_{i}.txt')
                save_path_2d_edges = os.path.join(class_folder_i, f'save_path_2d_edges_{i}.txt')
                save_path_2d_hist = os.path.join(class_folder_i, f'save_path_2d_hist_{i}.txt')

                data = [his_data[0][i], his_data[1][i], his_data[2][i], his_data[3][i]]
                path = [save_path_con, save_path_len, save_path_2d_edges, save_path_2d_hist]
                cacu.save_data(data, path)
                # 保存single_trance
                if i != 0:
                    save_path = os.path.join(class_folder_i, 'single_trance.npz')
                    np.savez(save_path, distance_array=datas[0][i - 1], conductance_array=datas[1][i - 1],
                             length_array=datas[2][i - 1], additional_length=len(datas[2][i - 1]))
            # 保存合并图
            image_merged = Image.open(self.merged_image)
            image_merged_path = os.path.join(kmeans_folder, 'merged_image.png')
            image_merged.save(image_merged_path)
            self.ui.progressBar.setValue(100)
            cacu.signal_window('Every data has saved')
            self.showNormal()
        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def save_button_clicked2(self):
        self.showMinimized()
        file_path = self.file_path
        folder_path = os.path.dirname(file_path)  # 提取single trance所在路径
        if self.file_path != "":

            if self.single_find_datas is not None:
                self.ui.progressBar_4.setValue(0)
                datas = self.single_find_datas
                # 保存图片至folder_path/fig文件夹，保存数据至folder_path/data文件夹
                his_data = self.single_run_image_his_data
                # 创建"kmeans"文件夹
                kmeans_folder = os.path.join(folder_path, 'find_or_denoise')
                os.makedirs(kmeans_folder, exist_ok=True)

                # 保存分类结果
                for i in range(3):
                    # 创建类别文件夹
                    class_folder_i = os.path.join(kmeans_folder, f'class_{i}')
                    # 确保保存文件夹存在
                    os.makedirs(class_folder_i, exist_ok=True)
                    file_path = self.single_run_image[i]
                    save_path = os.path.join(class_folder_i, f'image_{i}.png')
                    # 打开图像文件
                    image = Image.open(file_path)
                    # 保存图像到指定路径
                    image.save(save_path)
                    # 保存直方图
                    save_path_con = os.path.join(class_folder_i, f'con_1d_his_{i}.txt')
                    save_path_len = os.path.join(class_folder_i, f'len_1d_his_{i}.txt')
                    save_path_2d_edges = os.path.join(class_folder_i, f'save_path_2d_edges_{i}.txt')
                    save_path_2d_hist = os.path.join(class_folder_i, f'save_path_2d_hist_{i}.txt')

                    data = [his_data[0][i], his_data[1][i], his_data[2][i], his_data[3][i]]
                    path = [save_path_con, save_path_len, save_path_2d_edges, save_path_2d_hist]
                    cacu.save_data(data, path)

                    # 保存single_trance
                    if i != 0:
                        save_path = os.path.join(class_folder_i, 'single_trance.npz')
                        np.savez(save_path, distance_array=datas[0][i - 1], conductance_array=datas[1][i - 1],
                                 length_array=datas[2][i - 1], additional_length=len(datas[2][i - 1]))
                # 保存合并图
                image_merged = Image.open(self.single_find_merge_image)
                image_merged_path = os.path.join(kmeans_folder, 'merged_image.png')
                image_merged.save(image_merged_path)

            if self.Draw_iv_his_path:  # 保存1dhis图像以及数据  # self.iv_his_data  = [1d_edge,1d_hist, his2d_edg_list, hist]  +Chose_data
                Chose_data = "sort_data"
                folder_path_2 = os.path.join(folder_path, 'sort_data')
                os.makedirs(folder_path_2, exist_ok=True)

                # 文件路径 保存路径
                file_path = self.Draw_iv_his_path
                save_path = os.path.join(folder_path_2, 'Source_figure.png')
                # 打开图像文件
                image = Image.open(file_path)
                # 保存图像到指定路径
                image.save(save_path)

                name_txt = [f'{Chose_data}_1d_edge', f'{Chose_data}_1d_hist', f'{Chose_data}_his2d_edg_list',
                            f'{Chose_data}_hist']
                save_path_con = os.path.join(folder_path_2, f'{name_txt[1]}.txt')
                with open(save_path_con, 'w') as file:
                    for item in self.iv_his_data[1]:
                        line = f'{item[0]:<20}\t{item[1]:<20}\n'  # 使用格式化操作对齐每个元素
                        file.write(line)

                save_path_redLine_path = os.path.join(folder_path_2, f'meanLine.txt')
                with open(save_path_redLine_path, 'w') as file:
                    for i in range(len(self.redline[0])):
                        line = f'{self.redline[0][i]:<20}\t{self.redline[1][i]:<20}\n'  # 使用格式化操作对齐每个元素
                        file.write(line)
                base_data_hist_logI_path = os.path.join(folder_path_2, f'{name_txt[3]}.txt')
                base_data_edg_logI_path = os.path.join(folder_path_2, f'{name_txt[2]}.txt')
                nf.save_data_iv(histX=self.iv_his_data[3],
                                hist_pathX=base_data_hist_logI_path, listX=self.iv_his_data[2],
                                list_pathX=base_data_edg_logI_path)

                data = np.load(self.file_path)
                object_data_y = np.array(data['conductance_array'])
                object_data_x = np.array(data['distance_array'])

                # # 保存一下绘制误差棒的数据
                # save_path_err = os.path.join(folder_path_2, 'err_bar.txt')
                # # 将数组写入文本文件
                # np.savetxt(save_path_err, object_data_y, fmt='%d', delimiter='\n')

                # 保存single_trance
                save_path = os.path.join(folder_path_2, 'single_trance.npz')

                np.savez(save_path, distance_array=object_data_x[self.sorted_indices],
                         conductance_array=object_data_y[self.sorted_indices],
                         length_array=data['length_array'], additional_length=len(object_data_x))

            if self.conductance_fn_mean is not None:
                # 创建保存数据的文件夹
                folder_path_FN = os.path.join(folder_path, 'fn_TEST')
                os.makedirs(folder_path_FN, exist_ok=True)

                single_folder = os.path.join(folder_path_FN, 'single_curve_data')
                mean_folder = os.path.join(folder_path_FN, 'mean_curve_data')
                os.makedirs(single_folder, exist_ok=True)
                os.makedirs(mean_folder, exist_ok=True)

                # 保存单个曲线数据
                np.savetxt(os.path.join(single_folder, "distance_vs_conductance.txt"),
                           np.column_stack((self.distance_i, self.conductance_i)), header="Distance\tConductance", fmt="%.6e")
                np.savetxt(os.path.join(single_folder, "distance_vs_linear_conductance.txt"),
                           np.column_stack((self.distance_i, self.conductance_new)), header="Distance\tLinear Conductance",
                           fmt="%.6e")
                np.savetxt(os.path.join(single_folder, "FN_plot.txt"), np.column_stack((self.distance_new, self.conductance_fn)),
                           header="1/V\tln(I/V^2)", fmt="%.6e")

                # 保存均值曲线数据
                np.savetxt(os.path.join(mean_folder, "distance_vs_mean_conductance.txt"),
                           np.column_stack((self.distance_i, self.conductance_mean)), header="Distance\tMean Conductance", fmt="%.6e")
                np.savetxt(os.path.join(mean_folder, "distance_vs_mean_linear_conductance.txt"),
                           np.column_stack((self.distance_i, self.conductance_mean_new)), header="Distance\tMean Linear Conductance",
                           fmt="%.6e")
                np.savetxt(os.path.join(mean_folder, "FN_mean_plot.txt"),
                           np.column_stack((self.distance_new, self.conductance_fn_mean)), header="1/V\tln(I/V^2) (Mean)",
                           fmt="%.6e")
                # 文件路径 保存路径

                save_path = os.path.join(folder_path_FN, 'FN_image_path.png')
                # 打开图像文件
                image = Image.open(self.FN_image_path)
                # 保存图像到指定路径
                image.save(save_path)

            self.ui.progressBar_4.setValue(100)
            cacu.signal_window('Every data has saved')
            self.showNormal()
        else:
            self.ui.progressBar_4.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def object_data_prepro(self, distance, conductance, length, low_cut_length, high_cut_length, single_1d_his_bin,
                           low_cut_conductance, high_cut_conductance, single_2d_his_bin):
        # 单个数据预处理
        object_data = []
        if self.ui.comboBox_2.currentIndex() == 0:
            object_data = conductance  # 模块0 默认对电导进行聚类分析/不进行处理
        elif self.ui.comboBox_2.currentIndex() == 1:
            object_data, _ = cacu.single_tailor(distance, conductance,
                                                length, low_cut_length,
                                                high_cut_length)  # 模块1，裁剪后分析距离为从0到np.mean(data['length_array']))
        elif self.ui.comboBox_2.currentIndex() == 2:
            object_data = cacu.single_1d_his(conductance, bins=single_1d_his_bin,
                                             conductance_low_cut=low_cut_conductance,
                                             conductance_high_cut=high_cut_conductance)  # 模块2 投影至电导维度分析
        elif self.ui.comboBox_2.currentIndex() == 3:
            object_data = cacu.single_2d_his(distance, conductance, length, low_cut_length, high_cut_length,
                                             bins=single_2d_his_bin, conductance_low_cut=low_cut_conductance,
                                             conductance_high_cut=high_cut_conductance)  # 模块3 进行投影至2维分析
        elif self.ui.comboBox_2.currentIndex() == 4:
            object_data = cacu.single_linear_interpolation(distance, conductance, length, low_cut_length,
                                                           high_cut_length,
                                                           point_numbers=single_2d_his_bin)  # 模块4 取散点分析不加权重
        elif self.ui.comboBox_2.currentIndex() == 5:
            object_data = cacu.single_linear_interpolation(distance, conductance, length,
                                                           low_cut_length, high_cut_length,
                                                           use_weight=True,
                                                           point_numbers=single_2d_his_bin)  # 模块4 取散点分析加权重
        # elif self.ui.comboBox_2.currentIndex() == 6:
        #     object_data = nf.TSCFE(conductance)
        return object_data

    def object_data_dimension_reduction(self, object_data):
        # 进行至三降维
        if self.ui.comboBox_3.currentIndex() == 0:
            object_data = object_data
        elif self.ui.comboBox_3.currentIndex() == 1:
            # 使用 t-SNE 对数据进行降维------------------------------------------
            from sklearn.manifold import TSNE
            # 创建 t-SNE 实例
            tsne = TSNE(n_components=3, perplexity=60)
            object_data = np.array(object_data)
            object_data = tsne.fit_transform(object_data)
            # 使用 t-SNE 对数据进行降维------------------------------------------
        elif self.ui.comboBox_3.currentIndex() == 2:
            # 使用PCA行降维----------------------------------------------
            pca = PCA(n_components=3)
            object_data = np.array(object_data)
            object_data = pca.fit_transform(object_data)
            # 使用PCA行降维----------------------------------------------
        elif self.ui.comboBox_3.currentIndex() == 3:
            # 使用umap进行降维----------------------------------------------
            warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
            umap_model = umap.UMAP(n_components=3)
            object_data = np.array(object_data)
            # 使用UMAP对数据进行降维
            object_data = umap_model.fit_transform(object_data)
            # 使用umap进行降维----------------------------------------------
        elif self.ui.comboBox_3.currentIndex() == 4:
            # 使用MPVC进行降维
            object_data = cacu.MPVC_parameter(object_data)
        return object_data

    def button_clicked(self, ch_scan=False):
        self.ui.label_png2.clear()
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar.setValue(0)
            if ch_scan:
                clusters = self.cluster_num_scan
            else:
                plt.close('all')
                clusters = self.ui.clusters.property("value")
            clusters_way = self.ui.comboBox.currentIndex()
            data = np.load(self.file_path)
            conductance = data['conductance_array']
            distance = data['distance_array']
            length = data['length_array']
            self.get_parameters()
            values = self.values
            red = self.ui.red_set.value()
            # 对图表所有的参数进行读取与定义
            bins_1Dlen, bins_1Dcon, bins_2dhis_x, bins_2dhis_y, low_cut_1Dcon, high_cut_1Dcon, distance_0, distance_1, conductance_0, \
                conductance_1, single_1d_his_bin, single_2d_his_bin, low_cut_length, high_cut_length, low_cut_conductance, high_cut_conductance, his_1d_dis_0, his_1d_dis_1 = values
            if self.label_mode:
                object_data = None
            else:
                # 特征提取
                object_data = self.object_data_prepro(distance, conductance, length, low_cut_length, high_cut_length,
                                                      single_1d_his_bin,
                                                      low_cut_conductance, high_cut_conductance, single_2d_his_bin)
                # 降维
                object_data = self.object_data_dimension_reduction(object_data)

            label = self.label
            if not ch_scan:
                if self.label_mode:
                    if len(self.label) == len(conductance):
                        label = self.label
                    else:
                        self.ui.progressBar.setValue(100)
                        cacu.signal_window('The number of labels is not the same!')
                else:
                    label = None

                datas, plot_paths, his_data, self.merged_image, self.label = clustering(
                    distance, conductance, length,
                    object_data=object_data,
                    n_clusters=clusters, bins_1Dlen=bins_1Dlen,
                    bins_1Dcon=bins_1Dcon, bins_2dhis_x=bins_2dhis_x,
                    bins_2dhis_y=bins_2dhis_y,
                    high_cut_1Dcon=high_cut_1Dcon,
                    low_cut_1Dcon=low_cut_1Dcon,
                    distance_0=distance_0,
                    distance_1=distance_1,
                    conductance_0=conductance_1,
                    conductance_1=conductance_0,
                    clusters_way=clusters_way, color_1d=self.ui.color_style_1d.currentText(),
                    color_2d=self.ui.color_style_2d.currentText(), red=red, ivData=self.ui.ivMode.isChecked(),
                    label=label,
                    his_1d_dis_0=his_1d_dis_0, his_1d_dis_1=his_1d_dis_1)
                self.his_data = his_data
                self.datas = datas
                self.images = plot_paths
                # 合并图
                self.initUI(self.ui.label_png2, self.merged_image)
                self.ui.progressBar.setValue(100)
                cacu.signal_window('Calculation complete!')
                self.showNormal()

            else:
                labels = cacu.chose_cluster(object_data, clusters_way=clusters_way, n_clusters=clusters)
                self.ch_fig_path = cacu.cacu_chfig(labels, clusters, object_data)
                # 计算ch指数
                self.ch_score = calinski_harabasz_score(object_data, labels)

        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def CH_scan(self):
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar.setValue(0)
            plt.close('all')
            ch_score_list = []
            image_list = []
            # 聚类n-2次曲线绘图
            range_list = list(range(2, self.ui.clusters.property("value") + 1))

            for i in range(len(range_list)):
                self.cluster_num_scan = range_list[i]
                self.button_clicked(ch_scan=True)
                ch_score_list.append(self.ch_score)
                image_list.append(self.ch_fig_path)
                self.ui.progressBar.setValue(self.ui.progressBar.value() + 100 / len(range_list))

            max_index = ch_score_list.index(max(ch_score_list))
            fit_class = range_list[max_index]

            fig, axes = plt.subplots(1, len(range_list) + 1, figsize=(4 * (len(range_list) + 1), 4))

            axes[0].plot(range_list, ch_score_list)
            axes[0].set_ylabel('calinski_harabasz_score')
            axes[0].set_xlabel('class num')
            axes[0].set_title(f'MAX CH = {max(ch_score_list)} & class = {fit_class}')

            for i in range(len(ch_score_list)):
                image = plt.imread(image_list[i])
                axes[i + 1].imshow(image)
                axes[i + 1].set_title(f'class num = {i+2}')
                axes[i + 1].axis('off')
            # 调整子图间距
            plt.subplots_adjust(hspace=0.4)
            # 调整布局
            plt.tight_layout()

            folder_name = "png_images"
            path = os.path.join(folder_name, 'CH_scan.png')
            plt.savefig(path)
            self.initUI(self.ui.label_png2, path)
            self.ui.progressBar.setValue(100)
            cacu.signal_window('The calculation has completed')
            self.showNormal()
        else:

            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def little_window_for_merge(self):
        if self.his_data is not None:
            self.ui.progressBar.setValue(0)
            datas = self.datas
            class_number = len(datas[0])
            checkbox_vars = []

            # 获得每一个选择的值
            root = tk.Tk()
            # 设置窗口标题
            root.title("Merge Window")
            # 建立n类个选项
            for k in range(class_number):
                var = tk.IntVar()
                checkbox_vars.append(var)
                checkbox = tk.Checkbutton(root, text=f"Class {k + 1}", variable=var)
                checkbox.pack()

            # 判断值并计算合并后数据
            def checkbox_state():
                merge_class_distance = []
                merge_class_conductance = []
                merge_class_length = []
                merge_class = []
                numbers = 0
                numbers_merge_class = 0

                for i in range(class_number):
                    numbers += len(datas[0][i])
                    if checkbox_vars[i].get() == 1:
                        merge_class_distance.extend(datas[0][i])
                        merge_class_conductance.extend(datas[1][i])
                        merge_class_length.extend(datas[2][i])
                        merge_class.extend([i + 1])
                        numbers_merge_class += len(datas[0][i])

                    else:
                        print('Not add' + f'class{i + 1}  + in Merge drawing')
                value = f"{float(numbers_merge_class / numbers) * 100:.1f}"
                title = 'class' + str(merge_class) + '_' + str(value) + '%'
                red = self.ui.red_set.value()
                self.get_parameters()
                (
                    bins_1Dlen, bins_1Dcon, bins_2dhis, bins_2dhis_y, low_cut_1Dcon, high_cut_1Dcon, distance_0,
                    distance_1,
                    conductance_0,
                    conductance_1, single_1d_his_bin, single_2d_his_bin, low_cut_length, high_cut_length,
                    low_cut_conductance,
                    high_cut_conductance, his_1d_dis_0, his_1d_dis_1) = self.values

                merge_class_distance = np.array(merge_class_distance)
                merge_class_conductance = np.array(merge_class_conductance)
                merge_class_length = np.array(merge_class_length)

                his_data = cacu.draw_figure(merge_class_distance, merge_class_conductance, merge_class_length,
                                            bins_1Dcon=bins_1Dcon,
                                            low_cut_1Dcon=low_cut_1Dcon,
                                            high_cut_1Dcon=high_cut_1Dcon, bins_2dhis_x=bins_2dhis,
                                            bins_2dhis_y=bins_2dhis_y,
                                            distance_0=distance_0,
                                            distance_1=distance_1,
                                            conductance_0=conductance_1, conductance_1=conductance_0,
                                            bins_1Dlen=bins_1Dlen,
                                            label=title, color_1d=self.ui.color_style_1d.currentText(),
                                            color_2d=self.ui.color_style_2d.currentText(), red=red,
                                            ivData=self.ui.ivMode.isChecked(),
                                            his_1d_dis_0=his_1d_dis_0, his_1d_dis_1=his_1d_dis_1)
                # 再主界面显示图片
                # 创建"merge_class_folder"文件夹
                file_path = self.file_path
                folder_path = os.path.dirname(file_path)  # 提取single trance所在路径
                merge_class_folder = os.path.join(folder_path, f'{title}')
                os.makedirs(merge_class_folder, exist_ok=True)

                save_path = os.path.join(merge_class_folder, 'merge_class_single_trance.npz')
                np.savez(save_path, distance_array=merge_class_distance, conductance_array=merge_class_conductance,
                         length_array=merge_class_length, additional_length=len(merge_class_length))

                folder_name = "png_images"
                path = os.path.join(folder_name, f'{title}.png')

                image_save_path = os.path.join(merge_class_folder, f'{title}.png')
                # 打开图像文件
                self.initUI(self.ui.label_png2, path)
                image = Image.open(path)

                # 保存图像到指定路径
                image.save(image_save_path)
                # 保存直方图
                save_path_con = os.path.join(merge_class_folder, f'con_1d_his_merge_class.txt')
                save_path_len = os.path.join(merge_class_folder, f'len_1d_his_merge_class.txt')
                save_path_2d_edges = os.path.join(merge_class_folder, f'save_path_2d_edges_merge_class.txt')
                save_path_2d_hist = os.path.join(merge_class_folder, f'save_path_2d_hist_merge_class.txt')
                data = [his_data[0], his_data[1], his_data[2], his_data[3]]
                path = [save_path_con, save_path_len, save_path_2d_edges, save_path_2d_hist]
                cacu.save_data(data, path)
                self.ui.progressBar.setValue(100)

                root.destroy()  # 关闭窗口
                cacu.signal_window('The calculation has completed and saved in the root directory')

            # 创建按钮框容器
            button_box = ttk.Frame(root, padding=10)
            button_box.pack()

            # 创建按钮
            button_ok = ttk.Button(button_box, text="Merge drawing!", command=checkbox_state)
            button_ok.pack(side=tk.LEFT, padx=5)
            # 进入主循环
            root.mainloop()

        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def little_window_for_rawData_merge(self):
        if self.divide_data is not None:
            self.ui.progressBar.setValue(0)
            file_path = self.file_path
            folder_path = os.path.dirname(file_path)  # 提取single trance所在路径

            def get_values():
                choice_value = choice_entry.get()
                data_var_value = data_var_entry.get()
                print("max_var:", choice_value)
                print("min_var:", data_var_value)

                row_variances = self.hist_2d_data[1]

                divide_data_folder = os.path.join(folder_path, f'separated_data')
                os.makedirs(divide_data_folder, exist_ok=True)
                # 存数据
                # 判断条件
                judgment_list = (float(data_var_value) < row_variances) & (row_variances < float(choice_value))

                data = np.load(self.file_path)
                conductance = data['conductance_array']
                distance = data['distance_array']
                length = data['length_array']
                newData = nf.divide_merge(judgment_list,
                                          conductance, distance, length,
                                          cacu_num=self.ui.spinBox_3.value())

                save_path = os.path.join(divide_data_folder, f'mergeNew_data.npz')
                np.savez(save_path, distance_array=newData[0], conductance_array=newData[1],
                         length_array=newData[2], additional_length=newData[3])

                root.destroy()  # 关闭窗口
                self.ui.progressBar.setValue(100)
                cacu.signal_window('The calculation has completed and saved in the root directory')

            # 创建主窗口
            root = tk.Tk()
            root.title("Merge Window")

            # 创建标签和输入框1（Choice）
            choice_label = tk.Label(root, text="max_var:")
            choice_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

            choice_entry = tk.Entry(root)
            choice_entry.grid(row=0, column=1, padx=10, pady=10)

            # 创建标签和输入框2（Data Var）
            data_var_label = tk.Label(root, text="min_var:")
            data_var_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

            data_var_entry = tk.Entry(root)
            data_var_entry.grid(row=1, column=1, padx=10, pady=10)

            # 创建按钮，点击按钮时获取输入框的值
            merge_button = tk.Button(root, text="Merge", command=get_values)
            merge_button.grid(row=2, column=0, columnspan=2, pady=10)

            # 启动主循环
            root.mainloop()

        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')

    def Show_N_list(self):
        plt.close('all')
        if self.file_path != "":
            self.showMinimized()
            self.ui.progressBar_6.setValue(0)
            self.list_gMean = []
            self.list_integPSD = []
            data = np.load(self.file_path)
            logGArray = data['conductance_array']

            logG_low = self.ui.doubleSpinBox_2.value()
            logG_high = self.ui.doubleSpinBox_16.value()
            windowsize = self.ui.spinBox_11.value()
            frequence = self.ui.spinBox_12.value()
            N_CacuTime = self.ui.spinBox_13.value()

            CUT_num = N_CacuTime
            self.list_minN = []
            list_exponentIdx = []
            list_minNIdx = []
            keyPara = {"le_CondHigh": logG_high, "le_CondLow": logG_low, "le_Frequence": frequence,
                       "le_WindowSize": windowsize}

            for i in range(CUT_num):
                result = nf.getminN(logGArray, keyPara, CUT_num=CUT_num - 1, CUT_time=i)  # 保存计算结果
                if result == 0:
                    self.showNormal()
                    return 0
                else:
                    minN, gMean, integPSD, exponentIdx, minNIdx = result  # 直接使用保存的结果
                    self.list_minN.append(minN)
                    self.list_gMean.append(gMean)  # his坐标
                    self.list_integPSD.append(integPSD)
                    list_exponentIdx.append(exponentIdx)  # 过程坐标
                    list_minNIdx.append(minNIdx)

            # Create subplots for ax1 and ax2
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Set labels for ax1
            ax1.set_ylabel("corrcoef")
            ax1.set_xlabel("n")

            # Plot data on ax1
            for i in range(CUT_num):
                ax1.plot(list_exponentIdx[i], list_minNIdx[i])

            # Set labels for ax2
            ax2.set_xlabel("time")
            ax2.set_ylabel("N")

            # Plot data on ax2
            ax2.scatter(list(range(1,CUT_num+1)),self.list_minN)
            # Save the figure to a file, specify the filename
            folder_name = "png_images"


            self.list_minN_png_path = os.path.join(folder_name, 'list_minN.png')
            plt.savefig(self.list_minN_png_path)
            # Display the plots

            self.initUI(self.ui.label_57, self.list_minN_png_path)
            self.ui.progressBar_6.setValue(100)
            cacu.signal_window('ok!')
            self.showNormal()
        else:
            self.ui.progressBar_6.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def Show_psd_list(self):
        plt.close('all')
        if self.list_gMean is not None:
            self.showMinimized()
            self.ui.progressBar_6.setValue(0)

            N_CacuTime = self.ui.spinBox_13.value()

            # 需要传参
            # range_y = (self.ui.doubleSpinBox_18.value(), self.ui.doubleSpinBox_19.value())
            # range_x = (self.ui.doubleSpinBox_15.value(), self.ui.doubleSpinBox_17.value())

            fig, ax = plt.subplots(1, N_CacuTime, figsize=(4 * N_CacuTime, 3.5))
            # 调整子图间距
            plt.subplots_adjust(hspace=0.4)
            # 调整布局
            plt.tight_layout()
            # If N_CacuTime is 1, convert ax to a list for consistent indexing
            if N_CacuTime == 1:
                ax = [ax]

            # Iterate over each axis and plot
            for i in range(N_CacuTime):
                # Plot gMean vs integPSD
                ax[i].plot(self.list_gMean[i], self.list_integPSD[i])

                # Plot 2D histogram
                h = ax[i].hist2d(np.log(self.list_gMean[i]),
                                 np.log(self.list_integPSD[i] / self.list_gMean[i] ** self.list_minN[i]),
                                 bins=(self.ui.spinBox_15.value(), self.ui.spinBox_16.value()),
                                 # range=[range_x, range_y],  # Specify appropriate range_x and range_y
                                 cmap=self.ui.color_style_2d.currentText())

                # Add a colorbar to the plot
                fig.colorbar(h[3], ax=ax[i])
                # Set x-axis label
                ax[i].set_xlabel("log(gMean)")
                # Set plot title
                ax[i].set_title("n=" + str(self.list_minN[i]))
                # Set y-axis label
                ax[i].set_ylabel("log(integPSD / gMean^minN)")

            # Save the figure to a file, specify the filename
            folder_name = "png_images"
            self.integPSD_png_path = os.path.join(folder_name, 'integPSD.png')
            plt.savefig(self.integPSD_png_path)
            # Display the plots

            self.initUI(self.ui.label_57, self.integPSD_png_path)
            self.ui.progressBar_6.setValue(100)
            cacu.signal_window('ok!')
            self.showNormal()
        else:
            self.ui.progressBar_6.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def pre_single_cut(self):
        plt.close('all')
        if self.file_path != "":

            self.ui.progressBar_6.setValue(0)
            i = self.ui.spinBox_14.value() - 1
            data = np.load(self.file_path)
            logGArray = data['conductance_array']
            DisArrays = data['distance_array']

            logGLow = self.ui.doubleSpinBox_2.value()
            logGHigh = self.ui.doubleSpinBox_16.value()
            windowsize = self.ui.spinBox_11.value()

            logGArraySelect2 = np.where((logGArray >= logGLow) & (logGArray <= logGHigh), logGArray,  0)
            DisArraySelect2 = np.where((logGArray >= logGLow) & (logGArray <= logGHigh), DisArrays, -np.inf)

            if np.count_nonzero(logGArraySelect2[i]) > windowsize:
                GArray_0 = logGArraySelect2[i][logGArraySelect2[i] != 0][:windowsize]
                DArray_0 = DisArraySelect2[i][logGArraySelect2[i] != 0][:windowsize]
            else:
                GArray_0 = logGArraySelect2[i][logGArraySelect2[i] != 0]
                DArray_0 = DisArraySelect2[i][logGArraySelect2[i] != 0]

            plt.plot(DArray_0, GArray_0)
            plt.xlabel("DArray")
            plt.ylabel("GArray")

            folder_name = "png_images"
            self.single_cut_png_path = os.path.join(folder_name, 'single_cut.png')
            plt.savefig(self.single_cut_png_path)
            # Display the plots
            self.initUI(self.ui.label_57, self.single_cut_png_path)
            self.ui.progressBar_6.setValue(100)
            cacu.signal_window('ok!')

        else:
            self.ui.progressBar_6.setValue(100)
            cacu.signal_window('no data!  please input it!')

    def psd_save(self):
        plt.close('all')
        if self.file_path != "":
            self.ui.progressBar.setValue(0)
            file_path = self.file_path
            folder_path = os.path.dirname(file_path)  # 提取single trance所在路径

            psd_minN_folder = os.path.join(folder_path, 'psd_minN')
            os.makedirs(psd_minN_folder, exist_ok=True)
            if self.list_minN:
                save_path = os.path.join(psd_minN_folder, 'psd_minN.png')
                save_path = os.path.normpath(save_path)
                # 打开图像文件
                image = Image.open(self.list_minN_png_path)
                # 保存图像到指定路径
                image.save(save_path)

                # 定义保存路径
                save_path_Line_path = os.path.join(psd_minN_folder, 'psd_minN.txt')

                # 打开文件并写入 list_minN
                with open(save_path_Line_path, 'w') as f:
                    for item in self.list_minN:
                        f.write(f"{item}\n")  # 每个元素写入一行

            if self.integPSD_png_path:
                save_path = os.path.join(psd_minN_folder, 'integPSD.png')
                save_path = os.path.normpath(save_path)
                # 打开图像文件
                image = Image.open(self.integPSD_png_path)
                # 保存图像到指定路径
                image.save(save_path)
            if self.single_cut_png_path:
                save_path = os.path.join(psd_minN_folder, 'single_cut.png')
                save_path = os.path.normpath(save_path)
                # 打开图像文件
                image = Image.open(self.single_cut_png_path)
                # 保存图像到指定路径
                image.save(save_path)
            cacu.signal_window('ok!')

        else:
            self.ui.progressBar.setValue(100)
            cacu.signal_window('no data!  please input and calculate it!')


if __name__ == '__main__':
    freeze_support()  # 在 Windows 上支持多进程打包

    app = QApplication()
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("InJunction.IVCompatibility")
    # 创建主窗口并显示
    window = MainWindow()
    sys.exit(app.exec())
