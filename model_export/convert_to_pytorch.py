
import torch
import sys
sys.path.append("../")

import config
# from models.SimpleCNN import ConvModel
from models.ResNetModel import ResNetModel


# ckpt_path = "../summaries/2021_05_22_17_45_42/ckpt/model_eps_58_test_loss_0.1504.pth"		# simple cnn
# op_model_path = "./frozen_models/simplecnn_model_final.pt"

ckpt_path = "../summaries/2021_05_22_23_11_17/ckpt/model_eps_58_test_loss_0.2679.pth"		# resnet18-full train
op_model_path = "./frozen_models/resnet18_model_final.pt"


with torch.no_grad():   # Disabling the gradient calculations will reduce the calculation overhead.
    # model = ConvModel(num_classes=config.num_classes, inference=True)
    model = ResNetModel(num_classes=config.num_classes, freeze_backend=False, inference=True)

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model'])      # Only restoring model variables only 
model.eval()

torch.save(model, op_model_path)

