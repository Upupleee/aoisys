###############################################################################
#GUIDING INDEX
#RYAN   #SEND_TO_VRS_DEBUG_HERE
#JIngshu  #coco
###############################################################################
import os
import re
import logging
import time
import tifffile
import numpy as np
import torch
import PIL.Image
from PIL import Image
import subprocess
import sys
from copy import deepcopy
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QThread, QObject, QThreadPool, QRunnable, pyqtSlot, QDateTime
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from TemplateMatching.EfficientLoFTR.coarse_matching_new_test import coarse_matching_first, coarse_matching_eloftr_pillow_sample_only, coarse_matching_all
from TemplateMatching.EfficientLoFTR.fine_matching_loftr_new_drew_test import fine_matching_first, fine_matching_pillow_on_template, fine_matching_all, fine_matching_pillow_on_template_drew
from .ModelManger import ModelManager
from DefectDetection.sahi.sahi.auto_model import AutoDetectionModel
from DefectDetection.sahi.sahi.predict import predict, get_sliced_prediction,get_batched_sliced_prediction
from DefectDetection.sahi.sahi.utils.cv import  (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    crop_object_predictions,
    cv2,
    get_video_reader,
    read_image_as_pil,
    visualize_object_predictions,
)
from DefectDetection.sahi.sahi.utils.file import Path, increment_path
from DefectDetection.batchinference.vrs_generation import convert_coco_to_vrs
import numpy as np
import json
import ultralytics
import datetime 

PIL.Image.MAX_IMAGE_PIXELS = None 
MATERIAL_ROOT = "material"

class TaskSignals(QObject):
    match_finished = pyqtSignal(str)
    detect_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

# 全局JSON文件路径（基础名，实际会拼接资料名）
BASE_TASK_INFO_JSON = "_match_info.json"

# --- 模拟耗时任务的 QRunnable ---
class MatchAndDetectRunnable(QRunnable):
        # 类变量：收集方法1的任务信息（多线程共享）
    match_info = {"part_id": "", "samples": []}
    coco_results = []
    vrs = []
    _file_read = False  # 方法2是否已读取过JSON文件
    
    def __init__(self, sample_path, template_list, detection_params, callback_done, ab_cycle_count, material_id):
        super().__init__()
        self.signals = TaskSignals()
        self.sample_path = sample_path
        self.template_list = template_list
        self.ab_cycle_count = ab_cycle_count
        self.params = detection_params
        # 模型管理器（单例）
        self.model_manager = ModelManager()
        self.callback_done = callback_done

        self.is_method1 = False  # 标记当前任务是否使用方法1
        self.material_id = material_id  # 资料ID

    def extract_number_from_path(self, path):
        pattern = r'_?(\d+)\.\w+$'
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return None
    
    def get_tif_filenames_from_list(self, image_path_list, number):
        if 1 <= number <= len(image_path_list):
        # 获取第 number 个路径（转换为0-based索引）
            path = image_path_list[number - 1].strip()
                
            if path and path.lower().endswith(('.tif', '.tiff')):
                # 提取文件名部分
                filename = path.split("/")[-1]
                
                # 去掉后缀名（兼容 .tif 和 .tiff）
                if filename.lower().endswith(".tiff"):
                    filename_without_ext = filename[:-5]  # 移除 .tiff
                else:
                    filename_without_ext = filename[:-4]  # 移除 .tif
        
        return filename_without_ext
    
    def process_images(self, template_files, sample_image_path):
        Image.MAX_IMAGE_PIXELS = None

        if len(template_files) < 2:
            raise ValueError("模板文件数量不足两张，请检查模板文件夹。")

        template_second_last = template_files[-2]
        template_last = template_files[-1]

        img_second_last = Image.open(template_second_last).convert('L')
        img_last = Image.open(template_last).convert('L')

        width_second_last = img_second_last.width
        width_last = img_last.width

        ratio = width_last / width_second_last

        sample_img = Image.open(sample_image_path)
        sample_width, sample_height = sample_img.width, sample_img.height

        crop_width = int(sample_width * ratio)

        if crop_width > sample_width:
            print("警告：计算得到的保留宽度超过了样本图片总宽度，直接保留整个图片。")
            crop_width = sample_width

        left = sample_width - crop_width
        upper = 0
        right = sample_width
        lower = sample_height
        self.default_columns = 1

        cropped_img = sample_img.crop((left, upper, right, lower))

        return cropped_img, img_last
    
    def linear_adjust_template(self, template_img, captured_img, max_size):

        w,h = captured_img.size
        # captured_img_crop = captured_img.crop((0, 9000, w, h))
        captured_img_crop = captured_img.crop((0, 9000, w, h))
        template_array = np.array(template_img).astype(np.float32)
        captured_array = np.array(captured_img_crop).astype(np.float32)
        
        temp_h, temp_w = template_array.shape
        # 计算缩放比例，最长边缩放到max_size以内
        if max(temp_h, temp_w) > max_size:
            temp_scale = max_size / temp_h
            new_w = int(temp_w * temp_scale)
            template_array = cv2.resize(template_array, (new_w, max_size), interpolation=cv2.INTER_AREA)

        capt_h, capt_w = captured_array.shape
        # 计算缩放比例，最长边缩放到max_size以内
        if max(capt_h, capt_w) > max_size:
            capt_scale = max_size / capt_h
            new_w = int(capt_w * capt_scale)
            captured_array = cv2.resize(captured_array, (new_w, max_size), interpolation=cv2.INTER_AREA)

        mu_template = template_array.mean()
        sigma_template = template_array.std()
        mu_captured = captured_array.mean()
        sigma_captured = captured_array.std()

        adjusted = (template_array - mu_template) / sigma_template * sigma_captured + mu_captured

        adjusted = np.clip(adjusted, 0, 255)

        adjusted = adjusted.astype(np.uint8)
        captured_array = captured_array.astype(np.uint8)

        return adjusted, captured_array, temp_scale
    
    

    def template_matching(self, match_info):

        fname = Path(self.sample_path).name
        entry = next((e for e in match_info['samples']
                      if e['file_name'] == fname), None)
        if entry is None:
            print(f"No match info for {fname}")
            return None
        bbox    = entry['bbox']
        H_list  = entry['H_matrices']
        number=self.extract_number_from_path(self.sample_path)
        if len(self.template_list) ==number:
            sample_img, template_img=self.process_images(template_files=self.template_list, sample_image_path=self.sample_path)
            
            print("最后一个采样路径:",{self.sample_path})
            print("列表路径:",{self.template_list[number-1]})
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')
            print("采样路径:",{self.sample_path})
            print("列表路径:",{self.template_list[number-1]})

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        print(f"文件名字：{img_name}")
        template_arry, sample_arry , temp_scale= self.linear_adjust_template(template_img, sample_img, self.params.get('max_scale_size'))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        object_region,corresponding_sample_region = coarse_matching_all(
            sample_arry, 
            template_arry,
            bbox
        )
        if object_region is None or corresponding_sample_region is None:

            print("Coarse matching failed.")
            return 
        print("Coarse Matching Completed.")

        # Step 2: Fine Matching
        print("Starting Fine Matching Stage...")
        # Provide the paths to the cropped images from coarse matching
        
        # TODO: adding template automatic shifting procedure
        file_name = self.extract_prefix(img_name)
        result_overlay_image = fine_matching_all(  # Cropped sample image
            corresponding_sample_region,
            object_region,
            H_list
        )

        return result_overlay_image, file_name, img_name

    def extract_prefix(self, filename, delimiter='_'):
        
        last_delimiter_index = filename.rfind(delimiter)
        
        if last_delimiter_index != -1:
            # 如果找到分隔符，返回分割符之前的所有内容
            return filename[:last_delimiter_index]
        else:
            # 如果没有找到分隔符，返回原文件名
            return filename

    def _read_match_info_json(self):
        """读取JSON文件（仅执行一次）"""
        json_file = f"{self.material_id}{self.params.get('mode')}{BASE_TASK_INFO_JSON}"
        json_dir = "./material"  # 替换为你的JSON文件目录
        
        # 构建JSON文件路径
        json_path = os.path.join(json_dir, self.material_id, self.params.get('mode'),json_file)
        
        # 检查文件是否存在
        if not os.path.exists(json_path):
            print(f"❌ Match info file not found: {json_path}")
            return
        
        # 读取JSON内容
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                MatchAndDetectRunnable.match_info = json.load(jf)  # 存储到类变量供全局使用
                MatchAndDetectRunnable._file_read = True  # 标记为已读取
                print(f"✅ Successfully read match info (once): {json_path}")
                # 可在此处添加JSON数据的验证逻辑
        except json.JSONDecodeError:
            print(f"❌ Failed to parse JSON: {json_path}")
        except Exception as e:
            print(f"❌ Error reading JSON: {str(e)}")

    def run(self):
        self.is_running = True
        try:
            vrs_dir = None
            # self.is_method1 = (self.params.get('mode') in ['A', 'B'] and self.ab_cycle_count<2)
            self.is_method1 = False
            # 初始化或更新资料ID
            if not MatchAndDetectRunnable.match_info["part_id"]:
                MatchAndDetectRunnable.match_info["part_id"] = self.material_id
            elif MatchAndDetectRunnable.match_info["part_id"] != self.material_id:
                # 资料切换时，清空历史任务
                MatchAndDetectRunnable.match_info["part_id"] = self.material_id
                MatchAndDetectRunnable.match_info["samples"] = []
            if self.is_method1:
                matching_result, file_name, img_name, bbox, H_list= self.template_matching_first(self.model_manager)
                sample_info = {
                    "file_name": Path(self.sample_path).name,
                    "bbox": {
                        "x_min": int(bbox['x_min']),
                        "y_min": int(bbox['y_min']),
                        "x_max": int(bbox['x_max']),
                        "y_max": int(bbox['y_max'])
                    },
                    "H_matrices": []
                }
                for h_item in H_list:
                    grid = h_item.get("grid", [])
                    H_mat = h_item.get("H", [])
                    sample_info["H_matrices"].append({
                        "grid": [int(grid[0]), int(grid[1])] if grid else grid,
                        "H": [[float(val) for val in row] for row in H_mat] if H_mat else H_mat
                    })
                MatchAndDetectRunnable.match_info["samples"].append(sample_info)
                # json_path = os.path.join('results', f"{Path(file_name).stem}_match_info.json")
                # with open(json_path, 'w', encoding='utf-8') as jf:
                #     json.dump(match_info, jf, indent=4, ensure_ascii=False)
                # print(f"Match info JSON saved: {json_path}")
            else:
                # if not MatchAndDetectRunnable._file_read:
                #     self._read_match_info_json()  # 读取文件
                #     MatchAndDetectRunnable._file_read = True  # 标记为已读取
                # json_path = os.path.join('results', f"{Path(file_name).stem}_match_info.json")
                # if not os.path.exists(json_path):
                #     print("Match info file not found:", json_path)
                #     return
                # with open(json_path, 'r', encoding='utf-8') as jf:
                #     match_info = json.load(jf)
                # matching_result, file_name, img_name = self.template_matching(match_info)
                print('before matching')
                matching_result, file_name, img_name, number, temp_scale= self.template_matching_drew(self.model_manager)
                print('after matching')

            if matching_result is not None:
                if self.params.get('save_image'):
                    matching_np = np.array(matching_result)
                    if matching_np.ndim == 2:
                        matching_cv = matching_np
                    else:
                        matching_cv = cv2.cvtColor(matching_np, cv2.COLOR_RGB2BGR)

                    base_name = Path(file_name).stem
                    results_dir = os.path.join('results', base_name)

                    os.makedirs(results_dir, exist_ok=True)
                    output_path = os.path.join(results_dir, f"{img_name}.tiff")
                    cv2.imwrite(output_path, matching_cv, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
                    print(f"Overlay saved: {output_path}")
                # 2. 缺陷检测（使用预加载的模型）
                
                detector = self.model_manager.get_model(ModelManager.MODEL_TYPE_DETECTION, flag=self.params.get('flag'), conf_threshold=self.params.get('conf_thres'))
                print("load model successful")
                vrs_dir, fn, coco_path, batch_prediction = self._defect_detection(matching_result, file_name, detector)
                print("model detect successful")
                # # 获取temp_slice_widths列表
                # temp_slice_widths = self.params.get('temp_slice_widths', [])
                # # 根据number计算x_off
                # if number == 1:
                #     x_off = 0
                # else:
                #     # 需要取的元素个数 = number - 1
                #     count = number - 1
                #     # 确保不超过列表长度
                #     count = min(count, len(temp_slice_widths))
                #     # 取最后count个元素并求和
                #     x_off = sum(temp_slice_widths[-count:])
                # # vrs_dir, fn, coco_path, batch_prediction,coco_dir = self._defect_detection_coco_debug(matching_result, file_name, detector)
            
                # # x_off = self.params.get('temp_slice_widths')[number-1]
                # print(f"数量:{number}\n偏移:{x_off}")
                # for cat_id, x, y, bw, bh, score in batch_prediction:
                #     MatchAndDetectRunnable.coco_results.append({
                #         "category_id": cat_id,
                #         "bbox": [x/temp_scale + x_off, y/temp_scale, bw, bh],
                #         "score": score
                #     })
                # 获取temp_slice_widths列表
                temp_slice_widths = self.params.get('temp_slice_widths', [])

                # 计算count为temp_slice_widths的长度
                count = len(temp_slice_widths)

                # 根据number和count的关系计算x_off
                if number == count:
                    x_off = 0
                else:
                    # 计算需要取的元素个数 = count - number（因number ≤ count，结果一定非负）
                    take_count = count - number
                    # 取最后take_count个元素并求和
                    x_off = sum(temp_slice_widths[-take_count:]) if take_count > 0 else 0
                print(f"数量:{number}\n偏移:{x_off}")
                for cat_id, x, y, bw, bh, score in batch_prediction:
                    MatchAndDetectRunnable.coco_results.append({
                        "category_id": cat_id,
                        "bbox": [x/temp_scale + x_off, y/temp_scale, bw, bh],
                        "score": score
                    })
                
                self.signals.detect_finished.emit(str(coco_path))
        except Exception as e:
            error_message = f"在匹配和检测线程中发生异常: {e}"
            self.signals.error_occurred.emit(error_message)
            logging.error(error_message)
        finally:
            self.is_running = False
        print(f"处理: {self.sample_path}\n参数: {self.params}\n当前模式: {self.params.get('mode', '未知')}")
        self.callback_done(self.sample_path, vrs_dir)


    def template_matching_first(self, model_manger):
        number = self.extract_number_from_path(self.sample_path)
        if not number:
            raise ValueError(f"无法从路径提取编号: {self.sample_path}")

        if len(self.template_list) == number:
            sample_img, template_img = self.process_images(
                template_files=self.template_list, 
                sample_image_path=self.sample_path
            )
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        template_arry, sample_arry , temp_scale= self.linear_adjust_template(template_img, sample_img, self.params.get('max_scale_size'))

        # 执行粗匹配和精匹配（复用原逻辑）
        object_region, corresponding_sample_region, cut_box_sample = coarse_matching_first(
            sample_arry, template_arry, enlarge=True, margin_percent=0, scale_factor=1.0
        )
        if  object_region is None or  corresponding_sample_region is None:
            raise ValueError("粗匹配失败")
        file_name = self.extract_prefix(img_name)
        result_overlay_image, H_matrices= fine_matching_first(
            corresponding_sample_region, object_region,
            overlap_sample=self.params.get('overlap_samp'),
            overlap_template=self.params.get('overlap_temp'),
            Auto_split=self.params.get('auto_split'),
            default_rows=self.params.get('rows'),
            default_columns=self.params.get('cols')
        )
        return result_overlay_image, file_name, img_name, cut_box_sample, H_matrices
    
    def _template_matching(self):
        number = self.extract_number_from_path(self.sample_path)
        if not number:
            raise ValueError(f"无法从路径提取编号: {self.sample_path}")

        if len(self.template_list) == number:
            sample_img, template_img = self.process_images(
                template_files=self.template_list, 
                sample_image_path=self.sample_path
            )
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')

        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        template_arry, sample_arry , temp_scale= self.linear_adjust_template(template_img, sample_img, self.params.get('max_scale_size'))

        # 执行粗匹配和精匹配（复用原逻辑）
        object_region, corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
            sample_arry, template_arry, enlarge=True, margin_percent=0, scale_factor=1.0
        )
        if  object_region is None or  corresponding_sample_region is None:
            raise ValueError("粗匹配失败")
        file_name = self.extract_prefix(img_name)
        result_overlay_image = fine_matching_pillow_on_template(
            corresponding_sample_region, object_region,
            overlap_sample=self.params.get('overlap_samp'),
            overlap_template=self.params.get('overlap_temp'),
            Auto_split=self.params.get('auto_split'),
            default_rows=self.params.get('rows'),
            default_columns=self.params.get('cols')
        )
        return result_overlay_image, file_name, img_name
    
    def template_matching_drew(self, model_manger):
        number = self.extract_number_from_path(self.sample_path)
        if not number:
            raise ValueError(f"无法从路径提取编号: {self.sample_path}")
        
        print(number, "MDR")
        print(self.template_list, "asdasdasd")
        if len(self.template_list) == number:
            sample_img, template_img = self.process_images(
                template_files=self.template_list, 
                sample_image_path=self.sample_path
            )
            print(self.sample_path)
        else:
            sample_img = Image.open(self.sample_path).convert('L')
            template_img = Image.open(self.template_list[number-1]).convert('L')
            print(self.sample_path)
            print(self.self.template_list[number-1])
        print(0, "MDR")
        img_name = self.get_tif_filenames_from_list(self.template_list, number)
        template_arry, sample_arry , temp_scale= self.linear_adjust_template(template_img, sample_img, self.params.get('max_scale_size'))
        # 执行粗匹配和精匹配（复用原逻辑）
        object_region, corresponding_sample_region = coarse_matching_eloftr_pillow_sample_only(
            sample_arry, template_arry, enlarge=True, margin_percent=0, scale_factor=1.0, model=model_manger
        )
        if  object_region is None or  corresponding_sample_region is None:
            raise ValueError("粗匹配失败")
        print(1, "MDR")
        print(img_name)
        file_name = self.extract_prefix(img_name)
        print(2, "MDR")
        result_overlay_image = fine_matching_pillow_on_template_drew(
            corresponding_sample_region, object_region,
            grid_size=self.params.get('grid_size'),
            max_sampling_per_grid=self.params.get('max_sampling_per_grid'),
            enhance_algorithms=self.params.get('enhance_algorithms'),
            enhance_param=self.params.get('enhance_param'),
            extractor_name=self.params.get('extractor_name'),
            model = model_manger

        )
        print(3, "MDR")
        return result_overlay_image, file_name, img_name, number,temp_scale
    
    def part_infer(self, part, detector, y_offset =0, x_offset =0):
        batch_prediction = get_batched_sliced_prediction(
                image=part,
                detection_model=detector,
                slice_height=self.params.get('slice_win'),
                slice_width=self.params.get('slice_win'),
                overlap_height_ratio=0.05,
                overlap_width_ratio=0.05,
                verbose=0
            )
        preds = []
        for pred in batch_prediction.object_prediction_lists:
            x, y, bw, bh = pred.bbox.to_xywh()
            preds.append((pred.category.id, x + x_offset, y + y_offset, bw, bh, float(pred.score.value)))
        return preds

    def _defect_detection(self, matching_image, file_name, detector):
        if self.params.get('part_infer')==False:
            # 批量切片推理（复用原逻辑，但使用预加载的detector）
            batch_prediction = self.part_infer(matching_image, detector, y_offset =0, x_offset =0)
        else:
            height, width = matching_image.shape[:2]
            mid_height = height//2
            matching_image_parts = [(matching_image[:mid_height, :, :], (0, 0)),
                                    (matching_image[mid_height:, :, :], (mid_height, 0))]
            batch_prediction = []
            for part, (y_offset, x_offset) in matching_image_parts:
                part_prediction = self.part_infer(part, detector, y_offset, x_offset)
                batch_prediction.extend(part_prediction)


        # 保存结果
        project = f"./prediction_visual/{self.extract_prefix(file_name)}"
        os.makedirs(project, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        coco_dir = os.path.join(project, 'coco')
        os.makedirs(coco_dir, exist_ok=True)
        coco_json = os.path.join(coco_dir, f"detection_{ts}.json")
        with open(coco_json, 'w') as f:
            json.dump(batch_prediction, f, indent=4)
        vrs_dir = os.path.join(project, 'vrs')
        os.makedirs(vrs_dir, exist_ok=True)
        # vrs_path = os.path.join(vrs_dir, f"detection_{ts}.vrs")
        # convert_coco_to_vrs(
        #     batch_prediction.to_coco_predictions(image_id=1),
        #     vrs_path, image_height=matching_image.shape[0]
        # )

        return  vrs_dir, self.extract_prefix(file_name), coco_json, batch_prediction
    

# --- 控制器，管理任务、队列、线程池 ---
class DetectionController(QObject):
    detection_round_finished = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.thread_pool = QThreadPool()
        self.file_queue = Queue()
        self.expected_image_count = 0
        self.finished_count = 0
        self.processing = False
        self.slice_paths = []
        self.detection_params = {}

    def reset(self, expected_count):
        self.file_queue = Queue()
        self.expected_image_count = expected_count
        self.finished_count = 0
        self.processing = False
        # self.slice_paths.clear()
        self.detection_params.clear()

    def enqueue_file(self, file_path):
        self.file_queue.put(file_path)
        self.try_process_next()

    def extract_number_from_path(self, path):
        pattern = r'_?(\d+)\.\w+$'
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return None

    def try_process_next(self):
        if self.processing:
            return
        if self.file_queue.empty():
            return
        self.processing = True
        sample_path = self.file_queue.get()
        is_last = self.extract_number_from_path(sample_path) == self.expected_image_count
        
        
        # __import__('ipdb').set_trace()       
        params = self.last_image_params if is_last else self.detection_params
        print(f"当前AB面数量：{self.parent.ab_cycle_count}")
        runnable = MatchAndDetectRunnable(sample_path, self.slice_paths, params, self.matching_finished, self.parent.ab_cycle_count, self.parent.current_material)
        self.thread_pool.start(runnable)
        self.current_round_is_part_infer = params.get('part_infer') 
        # self.current_round_is_method1 = (params.get('mode') in ['A','B'] and self.parent.ab_cycle_count<2)


    def matching_finished(self, file_name, vrs_dir):
        self.finished_count += 1
        print(f"\n✅ 检测完成: {file_name}  ({self.finished_count}/{self.expected_image_count})")
        self.processing = False

        if self.finished_count == self.expected_image_count:
            print("🎉 当前轮全部检测完成！")
            # if self.current_round_is_method1:
            #     self.write_method1_match_info(json_path)
            # if self.current_round_is_part_infer:
            self.write_coco_info(vrs_dir) #added jingshu coco debug
            self.write_vrs_info(vrs_dir)
            self.detection_round_finished.emit()
        else:
            self.try_process_next()

    def write_vrs_info(self, vrs_dir):
        if not MatchAndDetectRunnable.coco_results:
            print("[方法1] 警告：未收集到任务信息，不写入JSON")
            return
        
        try:
            # 确保目录存在
            os.makedirs(vrs_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            vrs_path = os.path.join(vrs_dir, f"detection_{ts}.vrs")
            convert_coco_to_vrs(MatchAndDetectRunnable.coco_results, vrs_path, self.detection_params.get('temp_height') )
            print(f"\n===== 缺陷信息已写入 {vrs_path}（资料: {self.parent.current_material}） =====")
            MatchAndDetectRunnable.coco_results.clear()
        except Exception as e:
            print(f"写入vrs失败: {e}")



#added by jingshu
#coco_file_debug
    def write_coco_info(self, vrs_dir):
        """
        将YOLO生成的COCO检测结果写入JSON文件，保存到vrs_dir的上一级目录
        """
        if not MatchAndDetectRunnable.coco_results:
            print("[方法1] 警告：未收集到COCO检测结果，不写入JSON")
            return

        try:
            # vrs_dir的上一级目录
            parent_dir = os.path.dirname(vrs_dir)
            os.makedirs(parent_dir, exist_ok=True)

            # 文件名加时间戳
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            coco_path = os.path.join(parent_dir, f"detection_{ts}.json")

            # 保存COCO格式数据
            with open(coco_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(MatchAndDetectRunnable.coco_results, f, ensure_ascii=False, indent=4)

            print(f"\n===== COCO检测结果已写入 {coco_path}（资料: {self.parent.current_material}） =====")

        except Exception as e:
            print(f"写入COCO JSON失败: {e}")



        
    def write_method1_match_info(self, json_file):
        if not MatchAndDetectRunnable.match_info["samples"]:
            print("[方法1] 警告：未收集到任务信息，不写入JSON")
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(MatchAndDetectRunnable.match_info, f, ensure_ascii=False, indent=4)
                print(f"\n===== 方法1任务信息已写入 {json_file}（共{len(MatchAndDetectRunnable.match_info['samples'])}条，资料: {self.parent.current_material}） =====")
            # 清空类变量（避免下一轮方法1任务重复写入）
            MatchAndDetectRunnable.match_info["samples"].clear()
        except Exception as e:
            print(f"[方法1] 写入JSON失败: {e}")

    def prepare_detection(self, mode):
        if mode == "A":
            image_path = self.parent.get_current_image_path("A")
            if not image_path or not os.path.exists(image_path):
                self.parent.show_message("警告", "请先设置有效的A面图片!")
                return
                
            # 获取A面所有切分后图片路径并输出
            self.slice_paths = self.parent.get_slice_image_paths("A")
            print("\n===== 开始A面检测 =====")
            print(f"A面切分后图片路径列表 ({len(self.slice_paths)} 个):")
            for path in self.slice_paths:
                print(path)
                
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.current_scan_id = f"{self.parent.current_material}_{current_time}"

        elif mode == "B":
            image_path = self.parent.get_current_image_path("B")
            if not image_path or not os.path.exists(image_path):
                self.parent.show_message("警告", "请先设置有效的B面图片!")
                return
                
            # 获取B面所有切分后图片路径并输出
            self.slice_paths = self.parent.get_slice_image_paths("B")
            print("\n===== 开始B面检测 =====")
            print(f"B面切分后图片路径列表 ({len(self.slice_paths)} 个):")
            for path in self.slice_paths:
                print(path)
                
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.current_scan_id = f"{self.parent.current_material}_{current_time}"

            
        elif mode == "AB":
            a_image_path = self.parent.get_current_image_path("A")
            b_image_path = self.parent.get_current_image_path("B")
            
            if not a_image_path or not os.path.exists(a_image_path):
                self.parent.show_message("警告", "请先设置有效的A面资料!")
                return
                
            if not b_image_path or not os.path.exists(b_image_path):
                self.parent.show_message("警告", "请先设置有效的B面资料!")
                return
                
            current_side = "A" if self.parent.ab_cycle_count % 2 == 0 else "B"
            image_path = a_image_path if current_side == "A" else b_image_path
            
            # 获取当前面所有切分后图片路径并输出
            self.slice_paths = self.parent.get_slice_image_paths(current_side)
                
            current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.current_scan_id = f"{self.parent.current_material}_{self.parent.current_ab_cycle}_{current_time}"

            # self.parent.ab_cycle_count += 1
            mode = current_side  # 用于传递当前轮真实模式

        ini = self.parent.inipath
        self.detection_params = {
            'rows': self.parent.read_ini_param(ini, 'row', int),
            'cols': self.parent.read_ini_param(ini, 'col', int),
            'auto_split': self.parent.read_ini_param(ini, 'Auto_split', bool),
            'overlap_temp': self.parent.read_ini_param(ini, 'overlap_temp', int),
            'overlap_samp': self.parent.read_ini_param(ini, 'overlap_samp', int),
            'save_image': self.parent.read_ini_param(ini, 'SaveImage', bool),
            'flag': self.parent.read_ini_param(ini, 'flag', int),
            'slice_win': self.parent.read_ini_param(ini, 'slice', int),
            'conf_thres': self.parent.read_ini_param(ini, 'thres', float),
            'mode': mode,
            'part_infer':self.parent.read_ini_param(ini, 'PartInfer', bool),
            'temp_slice_widths': self.parent.temp_slice_widths,
            'temp_height': self.parent.temp_height,
            'max_scale_size': 30000,
            'grid_size': 6,
            'max_sampling_per_grid': 80,
            'enhance_algorithms': 'none',
            'enhance_param': [50, 100, 'none'],
            'extractor_name': 'aliked',
            'base_name': 'final_over_layer',
            'device': 'cuda'
        }

        # 为最后一张图设置特殊参数
        self.last_image_params = self.detection_params.copy()
        self.last_image_params['rows'] = 3
        self.last_image_params['cols'] = 2

