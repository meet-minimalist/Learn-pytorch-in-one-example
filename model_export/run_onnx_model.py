
import cv2
import torch
import numpy as np
import onnxruntime as ort
import sys
sys.path.append("../")

import config

# op_onnx_path = "./frozen_models/simplecnn_model_final.onnx"
op_onnx_path = "./frozen_models/resnet18_model_final.onnx"
# img_path = "./test_imgs/cat.3.jpg"
img_path = "./test_imgs/dog.3426.jpg"


def preprocess(img, normalize):
    h, w = img.shape[:2]
    if h > w:
        off_t = (h - w) // 2
        square_img = img[off_t : off_t + w, : , :]
    else:
        off_l = (w - h) // 2
        square_img = img[: , off_l : off_l + h , :]

    square_img = cv2.resize(square_img, (config.input_size[1], config.input_size[0]))    # [w x h]
    if normalize == 'resnet18':
        square_img = square_img / 255.
        square_img[:, :, 0] = (square_img[:, :, 0] - 0.485) / 0.229
        square_img[:, :, 1] = (square_img[:, :, 1] - 0.456) / 0.224
        square_img[:, :, 2] = (square_img[:, :, 2] - 0.406) / 0.225
    elif normalize == 'simplecnn':
        square_img = square_img / 255.
    square_img = np.transpose(square_img, (2, 0, 1))        # CHW from HWC
    square_img = np.expand_dims(square_img, axis=0)
    square_img = np.float32(square_img)
    return square_img




ort_session = ort.InferenceSession(op_onnx_path)

ori_img = cv2.imread(img_path)
img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
img_ip = preprocess(img, normalize='resnet18')

softmax_op = ort_session.run(None, {'input': img_ip})[0]
idx = np.argmax(softmax_op)

if idx == 0:
    print("Class: Cat, Probability: {:.4f}".format(softmax_op[0][idx]))
else:
    print("Class: Dog, Probability: {:.4f}".format(softmax_op[0][idx]))

cv2.imshow('img', ori_img)
cv2.waitKey()
cv2.destroyAllWindows()
