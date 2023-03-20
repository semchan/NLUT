import torch
from nlut_models import *
from PIL import Image
from utils.losses import *
from parameter_finetuning import *
import torch.nn as nn
from torchvision.utils import save_image
import time
import os

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


def adjust_learning_rate(optimizer, iteration_count, opt):
    """Imitating the original implementation"""
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


def finetuning_train(opt, original, example):
    content_tf2 = train_transform2()
    content_images = content_tf2(Image.open(
        original).convert('RGB')).unsqueeze(0).to(device)
    style_images = content_tf2(Image.open(
        example).convert('RGB')).unsqueeze(0).to(device)

    content_images = content_images.repeat(opt.batch_size, 1, 1, 1)
    style_images = style_images.repeat(opt.batch_size, 1, 1, 1)

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
    mn_losses = AverageMeter()

    # -----------------------training------------------------
    for i in range(opt.start_iter, opt.max_iter):
        adjust_learning_rate(optimizer, iteration_count=i, opt=opt)
        if opt.batch_size == 1:
            content_images = torch.cat([content_images, content_images], dim=0)
            style_images = torch.cat([style_images, style_images], dim=0)

        stylized, st_out, others = model(
            content_images, content_images, style_images, TVMN=TVMN_temp)
        tvmn = others.get("tvmn")
        LUT = others.get("LUT")
        mn_cons = (opt.lambda_smooth *
                   (tvmn[0]+10*tvmn[2]) + opt.lambda_mn*tvmn[1])*opt.mn_cons_weight

        loss_c, loss_s = model.encoder(content_images, style_images, stylized)
        loss_c = loss_c.mean()
        loss_s = loss_s.mean()

        # loss_mse = mseloss(content_images, stylized)
        loss_style = opt.content_weight*loss_c + \
            opt.style_weight*loss_s + mn_cons  # +tv_cons

        # optimizer update
        optimizer.zero_grad()
        loss_style.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()

        # update loss log
        log_c.append(loss_c.item())
        log_s.append(loss_s.item())

        losses.update(loss_style.item())
        c_losses.update(loss_c.item())
        s_losses.update(loss_s.item())
        mn_losses.update(mn_cons.item())

        # save image
        if i % opt.print_interval == 0 or (i + 1) == opt.max_iter:

            if opt.batch_size == 1:
                content_images, style_images, stylized = content_images[
                    :1], style_images[:1], stylized[:1]
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
            # state_dict = model.module.state_dict()
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))

            state = {'iter': i, 'state_dict': state_dict,
                     'optimizer': optimizer.state_dict()}
            torch.save(state, opt.resume)
            torch.save(state, "./"+opt.save_dir+"/" +
                       str(i)+"_finetuning_style_lut.pth")
    return LUT[:1]


def get_lut(opt, original, example):
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

    LUT = others.get("LUT")
    return LUT[:1]


def draw_img(original, dst, LUT):
    content_tf2 = p_transform()
    target = content_tf2(Image.open(original).convert(
        'RGB')).unsqueeze(0).to(device)

    TrilinearInterpo = TrilinearInterpolation()
    img_res = TrilinearInterpo(LUT, target)
    img_out = img_res+target

    save_image(img_out, dst, nrow=1)


if __name__ == '__main__':
    opt = parser.parse_args()

    original = opt.content_path
    example = opt.style_path
    dst = opt.output_path

    lut = finetuning_train(opt, original, example)
    lut = get_lut(opt, original, example)
    draw_img(original, dst, lut)

    print('save to: {}'.format(dst))
