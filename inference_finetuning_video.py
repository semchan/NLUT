import torch
from tqdm import tqdm
from pathlib import Path
from nlut_models import *
import torch.utils.data as data
from PIL import Image
from utils.losses import *
from parameter_finetuning import *
from torch.utils import data
import torch.nn as nn
from torchvision.utils import save_image
import time
import numpy as np
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


print(f'now device is {device}')


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def adjust_learning_rate(optimizer, iteration_count, opt):
    """Imitating the original implementation"""
    # lr = opt.lr / (1.0 + opt.lr_decay * iteration_count)
    lr = opt.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def p_transform():
    transform_list = [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def train_transform2():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def finetuning_train(opt, original=None, example=None):
    content_tf = train_transform()
    style_tf = train_transform()

    if original != None:
        content_images = content_tf(Image.open(
            original).convert('RGB')).unsqueeze(0).to(device)
        content_images = content_images.repeat(opt.batch_size, 1, 1, 1)
    else:
        content_dataset = FlatFolderDataset(opt.content_dir, content_tf)
        content_iter = iter(data.DataLoader(
            content_dataset, batch_size=opt.batch_size,
            sampler=InfiniteSamplerWrapper(content_dataset),
            num_workers=opt.n_threads))
    if example != None:
        style_images = style_tf(Image.open(
            example).convert('RGB')).unsqueeze(0).to(device)
        style_images = style_images.repeat(opt.batch_size, 1, 1, 1)
    else:
        style_dataset = FlatFolderDataset(opt.style_dir, style_tf)
        style_iter = iter(data.DataLoader(
            style_dataset, batch_size=opt.batch_size,
            sampler=InfiniteSamplerWrapper(style_dataset),
            num_workers=opt.n_threads))
    if opt.batch_size == 1:
        # content_images = content_images
        # style_images = style_images
        content_images = torch.cat([content_images, content_images], dim=0)
        style_images = torch.cat([style_images, style_images], dim=0)

    model = NLUTNet(opt.model, dim=opt.dim).to(device)
    print('Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters()) / 1000000.0))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("--------no checkpoint found---------")
    model.train()
    TVMN_temp = TVMN(opt.dim).to(device)

    # optimizer = torch.optim.Adam(model.module.parameters(), lr=opt.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    log_c = []
    log_s = []
    # log_mse = []
    Time = time.time()

    losses = AverageMeter()
    c_losses = AverageMeter()
    s_losses = AverageMeter()
    # mse_losses = AverageMeter()
    tv_losses = AverageMeter()
    mn_losses = AverageMeter()

    # -----------------------training------------------------
    for i in range(opt.start_iter, opt.max_iter):
        adjust_learning_rate(optimizer, iteration_count=i, opt=opt)
        if original == None:
            content_images = next(content_iter).to(device)
        if example == None:
            style_images = next(style_iter).to(device)

        stylized, st_out, others = model(
            content_images, content_images, style_images, TVMN=TVMN_temp)
        tvmn = others.get("tvmn")
        mn_cons = opt.lambda_smooth * \
            (tvmn[0]+10*tvmn[2]) + opt.lambda_mn*tvmn[1]

        loss_c, loss_s = model.encoder(content_images, style_images, stylized)
        loss_c = loss_c.mean()
        loss_s = loss_s.mean()

        # loss_mse = mseloss(content_images, stylized)
        loss_style = opt.content_weight*loss_c + opt.style_weight * \
            loss_s + opt.mn_cons_weight*mn_cons  # +tv_cons

        # optimizer update
        optimizer.zero_grad()
        loss_style.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()

        # update loss log
        log_c.append(loss_c.item())
        log_s.append(loss_s.item())
        # log_mse.append(loss_mse.item())

        losses.update(loss_style.item())
        c_losses.update(loss_c.item())
        s_losses.update(loss_s.item())
        # mse_losses.update(loss_mse.item())
        mn_losses.update(mn_cons.item())

        # save image
        if i % opt.print_interval == 0 or (i + 1) == opt.max_iter:

            if opt.batch_size == 1:
                content_image, style_image, stylized = content_images[
                    :1], style_images[:1], stylized[:1]
                output_name = os.path.join(opt.save_dir, "%06d.jpg" % i)
                output_images = torch.cat(
                    (content_image.cpu(), style_image.cpu(), stylized.cpu()), 0)
                save_image(stylized.cpu(), output_name, nrow=opt.batch_size)
            else:
                output_name = os.path.join(opt.save_dir, "%06d.jpg" % i)
                output_images = torch.cat(
                    (content_images.cpu(), style_images.cpu(), stylized.cpu()), 0)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print("iter %d   time/iter: %.2f  lr: %.6f loss_mn: %.4f loss_c: %.4f   loss_s: %.4f losses: %.4f " % (i,
                                                                                                                   (time.time(
                                                                                                                   )-Time)/opt.print_interval,
                                                                                                                   current_lr,
                                                                                                                   mn_losses.avg,
                                                                                                                   c_losses.avg, s_losses.avg,
                                                                                                                   losses.avg
                                                                                                                   ))
            log_c = []
            log_s = []
            Time = time.time()

        if (i + 1) % opt.save_model_interval == 0 or (i + 1) == opt.max_iter:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))

            state = {'iter': i, 'state_dict': state_dict,
                     'optimizer': optimizer.state_dict()}
            torch.save(state, opt.resume)
            torch.save(state, "./"+opt.save_dir+"/" +
                       str(i)+"_finetuning_style_lut.pth")


def get_lut(opt, original, example):

    # opt = setting.opt
    model = NLUTNet(opt.model, dim=opt.dim).to(device)
    print('Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters()) / 1000000.0))

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])

        else:
            print("--------no checkpoint found---------")

    model.train()
    TVMN_temp = TVMN(opt.dim).to(device)

    content_tf2 = train_transform2()
    content_images = content_tf2(Image.open(
        original).convert('RGB')).unsqueeze(0).to(device)
    style_images = content_tf2(Image.open(
        example).convert('RGB')).unsqueeze(0).to(device)

    content_images = content_images.repeat(2, 1, 1, 1)
    style_images = style_images.repeat(2, 1, 1, 1)

    stylized, st_out, others = model(
        content_images, content_images, style_images, TVMN=TVMN_temp)
    # save_image(stylized, "output_name.png", nrow=opt.batch_size)

    LUT = others.get("LUT")
    return LUT[:1]


def draw_video(target_mask, original_path, reference_mask, corrected_mask, LUT):
    sigmod_infer = nn.Sigmoid()
    cap_target_src = cv2.VideoCapture(target_mask)
    src_true, original = cap_target_src.read()

    example = reference_mask

    # 一些变换 toPIL&nb

    # LUT = infer(original_path, example)
    content_tf = p_transform()
    TrilinearInterpo = TrilinearInterpolation()

    Path(corrected_mask).parent.mkdir(parents=True, exist_ok=True)
    cap_target = cv2.VideoCapture(target_mask)
    cap_reference = content_tf(Image.open(
        reference_mask).convert('RGB')).unsqueeze(0).to(device)

    width = int(cap_target.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_target.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap_target.get(cv2.CAP_PROP_FOURCC))
    fps = cap_target.get(cv2.CAP_PROP_FPS) if fourcc else 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cap_corrected = cv2.VideoWriter(
        corrected_mask, fourcc, fps, (width, height))

    # frame_count = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = (cap_target.get(cv2.CAP_PROP_FRAME_COUNT))

    all_time = 0
    frame = 0

    try:
        with tqdm(desc=f"Frames", total=frame_count) as pbar:
            while all(cap.isOpened() for cap in (cap_target, cap_corrected)):
                ret_target, target = cap_target.read()

                if not ret_target:
                    break

                target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
                target = Image.fromarray(target)
                target = content_tf(target).unsqueeze(0).to(device)

                start_time = time.time()

                img_res = TrilinearInterpo(LUT, target)
                img_out = img_res+target

                # img_out = sigmod_infer(img_out)

                img_out = torch.squeeze(img_out, dim=0)
                img_out = torch.permute(img_out, (1, 2, 0))

                # 结束时间
                end_time = time.time()
                all_time = all_time+(end_time-start_time)

                corrected = img_out.detach().cpu().numpy()*255
                corrected = np.uint8(np.clip(corrected, 0, 255))

                corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

                cap_corrected.write(corrected)
                pbar.update(1)
                frame = frame + 1
            print(f'all fps: {fps/all_time}')
            average_time = 1000.0*all_time/frame  # ms
            print(f'average time: {average_time}')
    finally:
        cap_target.release()
        cap_corrected.release()


if __name__ == '__main__':
    opt = parser.parse_args()

    original = opt.content_path
    example = opt.style_path
    src_video = opt.src_video
    dst_video = opt.dst_video

    finetuning_train(opt, original, example)
    lut = get_lut(opt, original, example)
    draw_video(src_video, original, example, dst_video, lut)
    print('save to: {}'.format(dst_video))
    