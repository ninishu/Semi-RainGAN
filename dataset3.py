import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


def make_dataset(root, is_train):
    if is_train:

        input_t = open(os.path.join(root, 'train_input.txt'))
        ground_t = open(os.path.join(root, 'train_gt.txt'))
        depth_t = open(os.path.join(root, 'train_depth.txt'))

        inputs = [img_name.strip('\n') for img_name in input_t]
        gts = [img_name.strip('\n') for img_name in ground_t]
        depths = [img_name.strip('\n') for img_name in depth_t]

        input_t.close()
        ground_t.close()
        depth_t.close()

        return [[inputs[i], gts[i], depths[i]] for i in range(len(inputs))]

    else:

        input_t = open(os.path.join(root, 'test_input.txt'))
        ground_t = open(os.path.join(root, 'test_gt.txt'))
        depth_t = open(os.path.join(root, 'test_depth.txt'))

        inputs = [img_name.strip('\n') for img_name in
                 input_t]
        gts = [img_name.strip('\n') for img_name in
              ground_t]
        depths = [img_name.strip('\n') for img_name in
                 depth_t]

        input_t.close()
        ground_t.close()
        depth_t.close()

        return [[inputs[i], gts[i], depths[i]] for i in range(len(inputs))]


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, is_train=True):
        self.root = root
        self.imgs = make_dataset(root, is_train)
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        img_path, gt_path, depth_path = self.imgs[index]
        # print(img_path)
        # print(gt_path)
        # print(depth_path)
        img = Image.open(img_path)
        target = Image.open(gt_path)
        depth = Image.open(depth_path)
        if self.is_train:
            if self.transform is not None:
                img1, target1, depth1 = self.transform(img, target, depth)
            else:
                img1, target1, depth1 = img, target, depth

            img = np.array(img1)
            target = np.array(target1)

            # mask处理
            try:
                mask = (img - target).sum(axis=2)

                mask[mask <= 30] = 0
                mask[mask > 30] = 1
                mask = mask.astype(np.float32)

            except:
                print(img_path)
                return
            mask = torch.from_numpy(mask)

            img = transform(img1)
            target = transform(target1)
            depth = transform(depth1)
            return img, target, depth, mask, img_path, gt_path
        else:
            img = self.transform(img)
            target = self.transform(target)
            return img, target, img_path, gt_path

    def __len__(self):
        return len(self.imgs)
