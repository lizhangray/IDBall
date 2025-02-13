

import torch
from evaluator import Eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 或者你指定的设备
config = {
    "model_path": "model/checkpoint_DualDehaze+OTS_240502_xu_22.pth",  # 替换为你的 checkpoint 文件路径
    "save_path": "model/extracted/OTS-025.pth", # 替换为你想要保存提取模型的路径
    "scale": 4
}

myEvaluator = Eval(device=device,scale=4)

# 加载模型
myEvaluator.loadmodel(config["model_path"])

if isinstance(myEvaluator.model, torch.nn.DataParallel):
    model_state_dict = myEvaluator.model.module.state_dict() # 如果是 DataParallel，需要访问 .module
else:
    model_state_dict = myEvaluator.model.state_dict()

# 创建包含 "model" 键的 checkpoint 字典
checkpoint_to_save = {"model": model_state_dict}

# 保存提取的 model state_dict 到新的 checkpoint 文件
torch.save(checkpoint_to_save, config["save_path"])
print(f"已将提取的 model state_dict 保存到: {config['save_path']}")

print("完成!")

