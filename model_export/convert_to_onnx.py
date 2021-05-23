

import onnx
import torch
from onnxsim import simplify
import sys
sys.path.append("../")

import config
# from models.SimpleCNN import ConvModel
from models.ResNetModel import ResNetModel


# ckpt_path = "../summaries/2021_05_22_17_45_42/ckpt/model_eps_58_test_loss_0.1504.pth"		# simple cnn
# op_onnx_model_path = "./frozen_models/simplecnn_model_final.onnx"

ckpt_path = "../summaries/2021_05_22_23_11_17/ckpt/model_eps_58_test_loss_0.2679.pth"		# resnet18-full train
op_onnx_model_path = "./frozen_models/resnet18_model_final.onnx"
input_names = [ "input" ]
output_names = [ "output" ]


with torch.no_grad():   # Disabling the gradient calculations will reduce the calculation overhead.
    # model = ConvModel(num_classes=config.num_classes, inference=True)
    model = ResNetModel(num_classes=config.num_classes, freeze_backend=False, inference=True)

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model'])      # Only restoring model variables only 
model.eval()


dummy_input = torch.randn(1, 3, config.input_size[0], config.input_size[1])

# Saving onnx model
torch.onnx.export(model, dummy_input, op_onnx_model_path, input_names=input_names, output_names=output_names)

# Load onnx model to simplify it.
onnx_model = onnx.load(op_onnx_model_path)

# Simplify the model by removing redundant nodes.
model_simp, check = simplify(onnx_model)

print("Status: ", check)
onnx.save(model_simp, op_onnx_model_path)
