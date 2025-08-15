import cv2
import onnxruntime as ort
import numpy as np

def init_model(model_path):
    model = ort.InferenceSession(model_path)
    return model

def preprocess(path):
    image = cv2.imread(path) # BGR
    if image is None:
        print("image is None")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # GRAY
    image = cv2.resize(image, (28, 28)) # resize to 28x28

    image = image.astype(np.float32) # float32
    image = image / 255.0 # normalize to 0-1
    image = image.reshape(1, 1, 28, 28) # reshape to 1x1x28x28
    return image

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def predict(model, image):
    if image is None or model is None:
        print("image or model is None")
        return None
    input_name = model.get_inputs()[0].name
    output_result = model.run(None, {input_name: image})
    print("logic done:", output_result)
    probs = softmax(output_result[0])
    predicted_class = np.argmax(probs)
    print("概率分布:", probs)
    print("预测类别:", predicted_class)

    return output_result

def test_py_numpy():
    a = np.array([[1], [2], [3], [4]]) # 4x1, two dimension 
    a2 = np.array([1, 2, 3, 4]) # 4, one dimension

    c = np.expand_dims(a, axis=0) # 1x4x1, three dimension
    c = np.expand_dims(c, axis=0) # 1x1x4x1, four dimension
    print(a)
    print(c)

if __name__ == "__main__":
    #test_py_numpy()
    model = init_model("./filedepends/models/mnist-8.onnx")
    image = preprocess("./filedepends/pics/1.jpg")
    output_result = predict(model, image)
    print(output_result)
