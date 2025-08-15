# SPDX-License-Identifier: MIT

import cv2
import onnxruntime as ort
import argparse
import numpy as np
import box_utils

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-320 onnx model

model_path = "./filedepends/models/det_2.5g.onnx"
img_paths = ["./filedepends/pics/face100.HEIC.jpg",
             "./filedepends/pics/face101.HEIC.jpg",
             "./filedepends/pics/face102.HEIC.jpg",
             "./filedepends/pics/face103.HEIC.jpg",
             "./filedepends/pics/face104.HEIC.jpg",
             "./filedepends/pics/face105.HEIC.jpg"]

# scale current rectangle to box
def scale(box):
    return box
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# crop image
def cropImageForModel(orig_image):
    model_input_size = (640, 640)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model_input_size[0], model_input_size[1]), cv2.INTER_AREA)
    image_mean = np.array([127, 127, 127], dtype=np.float32) # define a array for 127, 127, 127
    image = image.astype(np.float32) # convert to float32
    image = (image - image_mean) / 128.0 # Normalized to [-1, 1], make sure is float32
    image = np.transpose(image, [2, 0, 1]) #hwc->chw, c means channel
    image = np.expand_dims(image, axis=0) # insert a new dimension at axis 0
    return image

# face detection method
def faceDetector(face_detector, orig_image, threshold = 0.3):
    image = cropImageForModel(orig_image)
    #for input in face_detector.get_inputs():
        #print("Input name:", input.name)
        #print("Shape:", input.shape)
        #print("Type:", input.type)

    input_name = face_detector.get_inputs()[0].name
    #for output in face_detector.get_outputs():
        #print("Output name:", output.name)
        #print("Shape:", output.shape)
        #print("Type:", output.type)
        
    outputs = face_detector.run(None, {input_name: image})
    confidences = outputs[0:3]   # 每个是 (?, 1)
    confidences = np.concatenate(confidences, axis=0)  # 把它们拼接成一个整体

    boxes       = outputs[3:6]   # 每个是 (?, 4)
    boxes       = np.concatenate(boxes, axis=0)  # 把它们拼接成一个整体

    landmarks   = outputs[6:9]   # 每个是 (?, 10)
    landmarks   = np.concatenate(landmarks, axis=0)  # 把它们拼接成一个整体

    boxes, labels, probs = box_utils.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main void

    #cv2.imshow('', orig_image)
def test_py_numpy():
    a = np.array([[1, 2, 3], [11, 12, 13], [21, 22, 23]])
    b = a.reshape(1, 1, 3, 3)
    c = np.expand_dims(a, axis=0)
    c = np.expand_dims(c, axis=0)
    print(a)
    print(b)
    print(c)

if __name__ == "__main__":
    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
    # ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
    face_detector = ort.InferenceSession(model_path)

    color = (255, 0, 0)

    for img_path in img_paths:
        orig_image = cv2.imread(img_path)
        boxes, labels, probs = faceDetector(face_detector, orig_image, 0.2)
        output_path = "./output/" + img_path.split("/")[-1]
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)
            cv2.imwrite(output_path, orig_image)
            print("save image to ", output_path)
        print("finished")