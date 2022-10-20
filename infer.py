import os
import time
import cv2
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from nets import SemiRainGAN
from config import test_path
from misc import check_mkdir
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'DGNLNet2022-09-22'
args = {
    'snapshot': 'latest',
    'depth_snapshot': ''
}
transform = transforms.Compose([
    # transforms.Resize([256, 512]),
    transforms.ToTensor()])
root = os.path.join(test_path, 'test/')

to_pil = transforms.ToPILImage()


def heatmap(img):
    if len(img.shape) == 3:
        b, h, w = img.shape
        heat = np.zeros((b, 3, h, w)).astype('uint8')
        for i in range(b):
            heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, :, :], cv2.COLORMAP_JET), (2, 0, 1))
    else:
        b, c, h, w = img.shape
        heat = np.zeros((b, 3, h, w)).astype('uint8')
        for i in range(b):
            heat[i, :, :, :] = np.transpose(cv2.applyColorMap(img[i, 0, :, :], cv2.COLORMAP_JET), (2, 0, 1))
    return heat


def main():
    net = SemiRainGAN().cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                       map_location=lambda storage, loc: storage.cuda(0)))
    net.eval()
    avg_time = 0

    with torch.no_grad():

        img_list = [img_name for img_name in os.listdir(root)]

        for idx, img_name in enumerate(img_list):
            check_mkdir(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['snapshot'])))
            if len(args['depth_snapshot']) > 0:
                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['depth_snapshot'])))

            img = Image.open(os.path.join(root, img_name)).convert('RGB')
            w, h = img.size
            img_var = Variable(transform(img).unsqueeze(0)).cuda()

            start_time = time.time()

            # res, dps = net(img_var)
            # res = net(img_var)
            # if w > h:
            res, attention, depth = net(img_var)
            torch.cuda.synchronize()

            avg_time = avg_time + time.time() - start_time

            print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_list), avg_time / (idx + 1)))

            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))
            result.save(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (
                    exp_name, args['snapshot']), img_name)
            )

            attention = transforms.Resize((h, w))(attention.data.squeeze(0).cpu())
            attention = attention.cpu().data
            attention = attention * 255
            attention = np.clip(attention.numpy(), 0, 255).astype('uint8')
            attention = heatmap(attention)

            attention = np.transpose(attention[0], (1, 2, 0))

            # cv2.imshow('image',attention)

            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (
                exp_name, args['snapshot']), 'a-' + img_name), attention)
        # cv2.imwrite(os.path.join(ckpt_path, exp_name,'att_%s.png'% (idx)), attention)
        # attention.save(
        #     os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (
        #         exp_name, args['snapshot']), 'a-' + img_name)
        # )


if __name__ == '__main__':
    main()
