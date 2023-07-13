# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils
from ultralytics import YOLO
import torch
import os


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV8ClassificationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "yolov8m-cls"
        self.cuda = torch.cuda.is_available()
        self.input_size = 640
        self.conf_thres = 0.25
        self.update = False
        self.model_weight_file = ""
        self.class_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'models',
            'imagenet_classes.txt'
        )

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.class_file = param_map["class_file"]
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["cuda"] = str(self.cuda)
        param_map["input_size"] = str(self.input_size)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["update"] = str(self.update)
        param_map["model_weight_file"] = str(self.model_weight_file)
        param_map["class_file"] = str(self.class_file)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV8Classification(dataprocess.CClassificationTask):

    def __init__(self, name, param):
        dataprocess.CClassificationTask.__init__(self, name)
        # Create parameters class
        if param is None:
            self.set_param_object(InferYoloV8ClassificationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.classes = None
        self.model = None
        self.half = False
        self.model_name = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Get parameters :
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        # Load model
        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda else torch.device("cpu")
            self.half = True if param.cuda else False
            self.read_class_names(param.class_file)

            if param.model_weight_file:
                self.model = YOLO(param.model_weight_file)
            else:
                self.model = YOLO(f'{param.model_name}.pt')
            param.update = False

        # Inference on whole image
        if self.is_whole_image_classification():
            # Run detection
            results = self.model.predict(
                src_image,
                save=False,
                imgsz=param.input_size,
                conf=param.conf_thres,
                half=self.half,
                device=self.device
            )

            # Get result output (classes, confidences)
            classes_names = results[0].names
            probs = results[0].probs
            classes_idx = probs.top1
            classe_name = [classes_names[classes_idx]]
            confidence = probs.top1conf.detach().cpu().numpy()

            # Display results in Ikomia application
            self.set_whole_image_results(classe_name, [str(confidence)])

        # Inference on ROIs
        else:
            input_objects = self.get_input_objects()
            for obj in input_objects:
                roi_img = self.get_object_sub_image(obj)
                if roi_img is None:
                    continue

                results = self.model.predict(
                    roi_img,
                    save=False,
                    imgsz=param.input_size,
                    conf=param.conf_thres,
                    half=self.half,
                    device=self.device
                )

                probs = results[0].probs
                classes_idx = probs.top1
                confidence = probs.top1conf.detach().cpu().numpy()
                self.add_object(obj, classes_idx, float(confidence))

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV8ClassificationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        self.info.name = "infer_yolo_v8_classification"
        self.info.short_description = "Inference with YOLOv8 image classification models"
        self.info.description = "This algorithm proposes inference for image classification " \
                                "with YOLOv8 models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "AGPL-3.0"
        # URL of documentation
        self.info.documentation_link = "https://docs.ultralytics.com/"
        # Code source repository
        self.info.repository = "https://github.com/ultralytics/ultralytics"
        # Keywords used for search
        self.info.keywords = "YOLO, classification, ultralytics, coco"

    def create(self, param=None):
        # Create process object
        return InferYoloV8Classification(self.info.name, param)
