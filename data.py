
import os
from glob import glob

import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor


class EvalData_haze_SOTS_05(data.Dataset):
    def __init__(self, input_dir, gt_dir):
        super().__init__()

        fpaths = glob(os.path.join(input_dir, '*.jpg'))

        input_names = []
        gt_names = []
        for path in fpaths:
            input_names.append(path.replace("\\", "/").split('/')[-1])
            gt = path.replace("\\", "/").split('/')[-1].split('_')[0]
            gt_names.append(str(gt) + '.png')

        self.input_names = input_names
        self.gt_names = gt_names
        self.input_dir = input_dir
        self.gt_dir = gt_dir


    def generate_scale_label(self, input_img):
        f_scale = 0.5
        width, height = input_img.size
        input_img = input_img.resize((int(width * f_scale), (int(height * f_scale))), resample=(Image.BICUBIC))
        return input_img

    def get_images(self, index):

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]


        input_img = Image.open(os.path.join(self.input_dir, input_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')

        input_img = self.generate_scale_label(input_img)


        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])


        input_tensor = transform_input(input_img)
        gt_tensor = transform_gt(gt_img)


        return input_tensor ,gt_tensor ,input_name# ,haze1, gt1,haze2, gt2

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)



class EvalData_haze_SOTS_025(data.Dataset):
    def __init__(self, input_dir, gt_dir):
        super().__init__()

        fpaths = glob(os.path.join(input_dir, '*.jpg'))

        input_names = []
        gt_names = []
        for path in fpaths:
            input_names.append(path.replace("\\", "/").split('/')[-1])
            gt = path.replace("\\", "/").split('/')[-1].split('_')[0]
            gt_names.append(str(gt) + '.png')

        self.input_names = input_names
        self.gt_names = gt_names
        self.input_dir = input_dir
        self.gt_dir = gt_dir


    def generate_scale_label(self, input_img):
        f_scale = 0.25
        width, height = input_img.size
        input_img = input_img.resize((int(width * f_scale), (int(height * f_scale))), resample=(Image.BICUBIC))
        return input_img

    def get_images(self, index):

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]


        input_img = Image.open(os.path.join(self.input_dir, input_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')

        input_img = self.generate_scale_label(input_img)


        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])


        input_tensor = transform_input(input_img)
        gt_tensor = transform_gt(gt_img)


        return input_tensor ,gt_tensor ,input_name# ,haze1, gt1,haze2, gt2

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


class EvalData_lowlight_05(data.Dataset):
    def __init__(self, input_dir, gt_dir):
        super().__init__()

        fpaths = glob(os.path.join(input_dir, '*.png')) # Assuming input images are png, like the original lowlight code

        input_names = []
        gt_names = []
        for path in fpaths:
            input_names.append(path.replace("\\", "/").split('/')[-1])
            gt_names.append(path.replace("\\", "/").split('/')[-1]) # Assuming gt names are the same as input names for low-light if in different directories

        self.input_names = input_names
        self.gt_names = gt_names
        self.input_dir = input_dir
        self.gt_dir = gt_dir


    def generate_scale_label(self, input_img):
        f_scale = 0.5
        width, height = input_img.size
        input_img = input_img.resize((int(width * f_scale), (int(height * f_scale))), resample=(Image.BICUBIC))
        return input_img

    def get_images(self, index):

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]


        input_img = Image.open(os.path.join(self.input_dir, input_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')

        input_img = self.generate_scale_label(input_img)


        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])


        input_tensor = transform_input(input_img)
        gt_tensor = transform_gt(gt_img)


        return input_tensor ,gt_tensor ,input_name# ,haze1, gt1,haze2, gt2

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


class EvalData_lowlight_025(data.Dataset):
    def __init__(self, input_dir, gt_dir):
        super().__init__()

        fpaths = glob(os.path.join(input_dir, '*.png')) # Assuming input images are png, like the original lowlight code

        input_names = []
        gt_names = []
        for path in fpaths:
            input_names.append(path.replace("\\", "/").split('/')[-1])
            gt_names.append(path.replace("\\", "/").split('/')[-1]) # Assuming gt names are the same as input names for low-light if in different directories

        self.input_names = input_names
        self.gt_names = gt_names
        self.input_dir = input_dir
        self.gt_dir = gt_dir


    def generate_scale_label(self, input_img):
        f_scale = 0.25
        width, height = input_img.size
        input_img = input_img.resize((int(width * f_scale), (int(height * f_scale))), resample=(Image.BICUBIC))
        return input_img

    def get_images(self, index):

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]


        input_img = Image.open(os.path.join(self.input_dir, input_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')

        input_img = self.generate_scale_label(input_img)


        transform_input = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])


        input_tensor = transform_input(input_img)
        gt_tensor = transform_gt(gt_img)


        return input_tensor ,gt_tensor ,input_name# ,haze1, gt1,haze2, gt2

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
    

class EvalData_GoPro_025(data.Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir 
        self.subdir_names = sorted(os.listdir(root_dir))
        self.blur_image_paths = [] 
        self.sharp_image_paths = [] 
        self.image_names = [] 

        for subdir_name in self.subdir_names:
            blur_dir = os.path.join(root_dir, subdir_name, 'blur')
            sharp_dir = os.path.join(root_dir, subdir_name, 'sharp')      

            blur_fpaths = sorted(glob(os.path.join(blur_dir, '*.*'))) 
            sharp_fpaths = sorted(glob(os.path.join(sharp_dir, '*.*'))) 
            
            blur_filenames = [os.path.basename(fp) for fp in blur_fpaths]
            sharp_filenames = [os.path.basename(fp) for fp in sharp_fpaths]

            
            common_filenames = sorted(list(set(blur_filenames) & set(sharp_filenames))) 

            for filename in common_filenames:
                self.blur_image_paths.append(os.path.join(blur_dir, filename)) 
                self.sharp_image_paths.append(os.path.join(sharp_dir, filename)) 
                self.image_names.append(subdir_name+"_"+filename) 


    def generate_scale_label(self, input_img):
        f_scale = 0.25 
        width, height = input_img.size 
        input_img = input_img.resize((int(width * f_scale), (int(height * f_scale))), resample=Image.BICUBIC) 
        return input_img 

    def get_images(self, index):
        blur_path = self.blur_image_paths[index]
        sharp_path = self.sharp_image_paths[index] 
        image_name = self.image_names[index] 

        blur_img = Image.open(blur_path).convert('RGB') 
        sharp_img = Image.open(sharp_path).convert('RGB') 

        blur_img_scaled = self.generate_scale_label(blur_img) 

        transform_input = Compose([ToTensor()]) 
        transform_gt = Compose([ToTensor()])   

        blur_tensor = transform_input(blur_img_scaled) 
        sharp_tensor = transform_gt(sharp_img) 

        return blur_tensor, sharp_tensor, image_name 


    def __getitem__(self, index):
        res = self.get_images(index) 
        return res 

    def __len__(self):
        return len(self.blur_image_paths) 