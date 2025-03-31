#!/usr/bin/env python

import os
import cv2
import csv
import base64
import numpy as np
import onnxruntime
import os.path as osp
from pages.face.scrfd import SCRFD
from pages.face.arcface_onnx import ArcFaceONNX

onnxruntime.set_default_logger_severity(3)

class Facedetect():
    def __init__(self):
        self.detector = SCRFD('pages/face/det_10g.onnx')
        self.detector.prepare(0)
        self.rec = ArcFaceONNX('pages/face/w600k_r50.onnx')
        self.rec.prepare(0)
        self.data_dict = {}

    def write_to_csv(self, name, image_path, feature):
        file_path = "pages/face/data/data.csv"
        directory = os.path.dirname(file_path)
        
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_exists = os.path.isfile(file_path)
        
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header if file does not exist
            if not file_exists:
                writer.writerow(["name", "image_path", "feature"])
            
            # Write data
            writer.writerow([name, image_path, feature])

    def load_csv_to_dict(self):
        file_path = "pages/face/data/data.csv"
        if not os.path.exists(file_path):
            print(f"File '{file_path}' does not exist.")
            return {}

        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader, None)  # Read the header row
            
            if headers != ["name", "image_path", "feature"]:
                print("CSV format mismatch! Expected columns: name, image_path, feature.")
                return {}

            for row in reader:
                if len(row) != 3:
                    continue  # Skip invalid rows
                
                name, image_path, feature = row
                self.data_dict[feature] = {"name": name, "image_path": image_path}


    def feature(self, image):
        bboxes, kpss = self.detector.autodetect(image, max_num=1)
        kps = kpss[0]
        feature_point = self.rec.get(image, kps)
        feature_string = " ".join(map(str, feature_point))
        feature_bytes = feature_string.encode("utf-8")
        encoded_data = base64.b64encode(feature_bytes)
        feature_str = encoded_data.decode("utf-8")
        
        return feature_str

    def base64_to_array(self, encode_feature):
        decoded_data = base64.b64decode(encode_feature)
        decoded_string = decoded_data.decode("utf-8")
        data_list = list(map(float, decoded_string.split()))

        return np.array(data_list)

    def detect(self, image):
        if image is None:
            print("Image not found or unable to read")
        
        bboxes, kpss = self.detector.autodetect(image)

        if len(bboxes) != 0:
            for index in range(len(bboxes)):
                detect_feature = self.rec.get(image, kpss[index])
                for feature, details in self.data_dict.items():
                    feature_data = self.base64_to_array(feature)
                    sim = self.rec.compute_sim(detect_feature, feature_data)
                    if sim<0.28:
                       conclu = 'They are NOT the same person'

                       face_match = False
                    else:
                       conclu = 'They ARE the same person'
                       face_match = True
                       print(f"Feature: {feature}, Name: {details['name']}, Image Path: {details['image_path']}")

