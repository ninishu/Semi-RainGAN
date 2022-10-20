import datetime
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import triple_transforms3
from nets import SemiRainGAN
from config import train_labeled_path, test_path, train_unlabeled_path, val_path
from dataset3 import ImageFolder
from misc import AvgMeter, check_mkdir
from loss import TVLoss, DCLoss, perceptual_loss
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

to_pil = transforms.ToPILImage()
cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'SemiRainGAN' + str(datetime.datetime.now().strftime("%Y-%m-%d"))
args = {
    'iter_num': 200000,
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'resume_snapshot': '',
    'val_freq': 50000000,
    'img_size_h': 256,
    'img_size_w': 512,
    'crop_size': 256,
    'snapshot_epochs': 2,
    'epoch': 200,
    'dc_patch_size': 20,
    'lambda_out': 10,
    'lambda_attention': 10,
    'lambda_TV': 1e-5,
    'lambda_DC': 2e-7

}

triple_transform = triple_transforms3.Compose([
    triple_transforms3.Resize((int(args['img_size_h']), int(args['img_size_w']))),
    # triple_transforms3.RandomCrop(args['crop_size'])
    # triple_transforms.RandomHorizontallyFlip()
])
test_transform = transforms.Compose([
    # transforms.Resize([256, 512]),
    transforms.ToTensor()])
to_tensor = transforms.ToTensor()
train_set_labeled = ImageFolder(train_labeled_path, transform=None, is_train=True)
train_loader_labeled = DataLoader(train_set_labeled, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)
# train_set_unlabeled = ImageFolder(train_unlabeled_path, transform=triple_transform, is_train=True)
# train_loader_unlabeled = DataLoader(train_set_unlabeled, batch_size=args['train_batch_size'], num_workers=0,
#                                     shuffle=True)

test1_set = ImageFolder(test_path, transform=test_transform, is_train=False)
test1_loader = DataLoader(test1_set, batch_size=1)

L1_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()
TVLoss = TVLoss()


log_path = os.path.join(ckpt_path, exp_name, '1.txt')
val_path = os.path.join(ckpt_path, exp_name, val_path)


def main():
    net = SemiRainGAN().cuda().train()
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ])

    if len(args['resume_snapshot']) > 0:
        print('training resumes from \'%s\'' % args['resume_snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_semi.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    check_mkdir(val_path)
    check_mkdir(os.path.join(val_path, "log"))
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    sup_loss_record = AvgMeter()
    out_loss_record = AvgMeter()
    attention_loss_record = AvgMeter()
    depth_loss_record = AvgMeter()

    result_psnr = 0
    result_ssim = 0
    for epoch in range(args['epoch']):

        # Supervised phase
        for i, data in enumerate(train_loader_labeled):
            # if curr_iter % 5 == 0:
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, gts, depth, mask, img_path, gt_path = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            depth = Variable(depth).cuda()
            mask = Variable(mask).cuda()
            optimizer.zero_grad()

            # result_psnr, result_ssim = validate(net, epoch, optimizer, result_psnr, result_ssim)
            out, attention, depth_pred = net(inputs)

            out_loss = L1_loss(out, gts)
            attention_loss = MSE_loss(attention[:, 0, :, :], mask)
            depth_loss = L1_loss(depth_pred, depth)

            loss = out_loss + attention_loss + depth_loss

            loss.backward()

            optimizer.step()

            sup_loss_record.update(loss.data, batch_size)
            out_loss_record.update(out_loss.data, batch_size)
            attention_loss_record.update(attention_loss.data, batch_size)
            depth_loss_record.update(depth_loss.data, batch_size)

            curr_iter += 1

            log = '[epoch %d], [iter %d], [supervised loss %.5f], [lr %.13f], [out_loss %.5f], [attention_loss %.5f],[depth_loss %.5f]' % \
                  (epoch, i, sup_loss_record.avg, optimizer.param_groups[1]['lr'],
                   out_loss_record.avg, attention_loss_record.avg, depth_loss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')

            # if (curr_iter + 1) % args['val_freq'] == 0:
            #     validate(net, curr_iter, optimizer)

        if (epoch + 1) % args['snapshot_epochs'] == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d_semi.pth' % (epoch + 1))))
            torch.save(optimizer.state_dict(),
                       os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (epoch + 1))))
        ###validating
        result_psnr, result_ssim = validate(net, epoch, optimizer, result_psnr, result_ssim)


def validate(net, epoch, optimizer, result_psnr, result_ssim):
    print('validating...')
    net.eval()
    Sum_ssim = 0
    Sum_psnr = 0
    with torch.no_grad():
        for i, data in enumerate(test1_loader):
            imgs, gts, image_path, gt_path = data
            img_var = Variable(imgs).cuda()
            image_path = image_path[0]
            gt_path = gt_path[0]
            img_name = image_path.split('/')[-1]
            img = Image.open(image_path).convert('RGB')
            w, h = img.size
            res, attention, depth = net(img_var)
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            result = to_tensor(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
            save_image(result, val_path + img_name)
            outs = cv2.imread(val_path + img_name)
            gts = cv2.imread(gt_path)
            psnrnum = compare_psnr(outs, gts, data_range=255)
            ssimnum = compare_ssim(outs, gts, data_range=255, multichannel=True)
            open(os.path.join(val_path, "log", "val.txt"), 'a').write(
                'i:%d,out:%s,gt:%s:psnr: %s, ssim:%s' % (
                i, str(val_path + img_name), str(gt_path), str(psnrnum), str(ssimnum)) + '\n')
            Sum_ssim += ssimnum
            Sum_psnr += psnrnum

    final_ssim = Sum_ssim / len(test1_loader)
    final_psnr = Sum_psnr / len(test1_loader)
    open(os.path.join(val_path, "log", "total.txt"), 'a').write(
        'epoch:%d,psnr: %s, ssim:%s' % (epoch, str(final_psnr), str(final_ssim)) + '\n')
    print('psnr: %s, ssim:%s ' % (str(final_psnr), str(final_ssim)))
    if final_psnr > result_psnr and final_ssim > result_ssim:
        print('saving...')
        result_psnr = final_psnr
        result_ssim = final_ssim
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'latest.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'latest_optim.pth'))

    net.train()
    return result_psnr, result_ssim


if __name__ == '__main__':
    main()
