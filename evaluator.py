import os

import torch
import torchvision
from tqdm import tqdm
from torch import clamp
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from piqa import SSIM, PSNR


import networks
import myutils


class Eval:
    def __init__(self, device, scale):
        self.device = device
        self.model = networks.CombineModel(self.device, scale)
        # self.model = torch.nn.DataParallel(self.model)

    def setup(self, batch, epoch):
        self.haze = batch[0].to(self.device)
        self.dehaze_GT = batch[1].to(self.device)
        self.dehaze_GT_HR = batch[2].to(self.device)
        self.epoch = epoch

    def loadmodel(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])

    def eval(self, dataset, task):
        eval_dataloader = DataLoader(dataset=dataset, batch_size=1)

        criterion_SSIM = SSIM()
        criterion_PSNR = PSNR()
        psnr_list = []
        ssim_list = []
        device = torch.device("cuda")

        output_path = "output/" + task + "/"

        if not os.path.exists("output_path"):
            os.makedirs(output_path, exist_ok=True)

        for index, data in tqdm(
            enumerate(eval_dataloader), total=len(eval_dataloader), desc="evaling"
        ):
            inputs, labels, labels_name = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_name = labels_name[0]
            with torch.no_grad():
                self.model.eval()
                """
                Both the <pad2affine> and <auto_pad_to_multiple> methods are implemented to address the requirement of 
                ensuring input images possess dimensions that are multiples of a specific value, typically 32, for 
                optimal processing within the neural network. Specifically, <pad2affine> represents a preliminary padding approach, 
                which, it is important to note, contains inherent limitations and potential bugs. This method was 
                utilized in the research paper.

                In contrast, <auto_pad_to_multiple> is presented as a more refined and robust padding method. 
                Employing <auto_pad_to_multiple> may potentially lead to improved performance metrics due to its enhanced 
                handling of padding.

                It is crucial to acknowledge that while <pad2affine> exhibits certain deficiencies in its padding mechanism, 
                the conclusions presented in the research paper remain robust and valid.
                """
                # inputs = myutils.auto_pad_to_multiple(inputs, mod=32, padding_mode='reflect')
                inputs = myutils.pad2affine(inputs, mod=32)
                D = self.model(inputs)
                centerCorp_HR = torchvision.transforms.CenterCrop(
                    (labels.shape[-2], labels.shape[-1])
                )
                D = centerCorp_HR(D)
                D = clamp(D, min=0, max=1)
                labels = clamp(labels, min=0, max=1)

                psnr_list.extend([criterion_PSNR(D, labels)])
                ssim_list.extend([criterion_SSIM(D.to("cpu"), labels.to("cpu"))])

                save_image(D, output_path + labels_name)

        avr_psnr = sum(psnr_list) / len(psnr_list)
        avr_ssim = sum(ssim_list) / len(ssim_list)
        print("psnr:{0},ssim:{1}".format(avr_psnr, avr_ssim))
