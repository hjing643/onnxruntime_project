import numpy as np
import cv2
def test_create_numpy_image():
    dark_image = np.zeros((256, 256), dtype=np.uint8)
    cv2.imwrite("./output/dark_image.png", dark_image)

    white_image = np.ones((256, 256), dtype=np.uint8) * 255
    cv2.imwrite("./output/white_image.png", white_image)

    first1 = np.arange(256, dtype=np.uint8)
    gray_image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    cv2.imwrite("./output/gray_image.png", gray_image)

if __name__ == "__main__":
    test_create_numpy_image()