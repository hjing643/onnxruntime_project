
import cv2
import onnxruntime as ort
import numpy as np
import box_utils

def init_model(model_path):
    face_detector = ort.InferenceSession(model_path)
    return face_detector

def faceDetector(face_detector, orig_image, threshold):
    cv2.


if __name__ == "__main__":
    face_detector = init_model("./filedepends/models/det_2.5g.onnx")
    regtangle_color = (255, 0, 0)
    img_paths = ["./filedepends/pics/face100.HEIC.jpg",
             "./filedepends/pics/face101.HEIC.jpg",
             "./filedepends/pics/face102.HEIC.jpg",
             "./filedepends/pics/face103.HEIC.jpg",
             "./filedepends/pics/face104.HEIC.jpg",
             "./filedepends/pics/face105.HEIC.jpg"]

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