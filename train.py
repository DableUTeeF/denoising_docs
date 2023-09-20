import os
from data import Data
from models import Model
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
import tensorflow as tf
from torch import nn
import torch
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from matplotlib import pyplot as plt


def collate_fn(batch):
    min_len = 2048
    skips = []
    for i, b in enumerate(batch):
        if b[0].size(2) < 256:
            skips.append(i)
        else:
            min_len = min(min_len, b[0].size(2))
    size = np.random.randint(256, min(min_len, 800))
    # size = int(size/32)*32
    x = []
    y = []
    for i, b in enumerate(batch):
        if i in skips:
            continue
        x.append(b[0][..., :size])
        y.append(b[1][..., :size])
    return torch.stack(x), torch.stack(y)


def collate_fn_val(batch):
    min_len = 2048
    for i, b in enumerate(batch):
        min_len = min(min_len, b[0].size(2))
    # min_len = int(min_len/32)*32
    x = []
    y = []
    for i, b in enumerate(batch):
        x.append(b[0][..., :min_len])
        y.append(b[1][..., :min_len])
    return torch.stack(x), torch.stack(y)


if __name__ == '__main__':
    out_path = '/media/palm/Data/doc_denoise/denoising-dirty-documents/outputs_inter'
    os.makedirs(os.path.join(out_path, f'cp'), exist_ok=True)
    src_path = '/home/palm/PycharmProjects/mmmmocr/imgs/samples'
    batch_size = 8
    n_epochs = 30
    device = 'cuda'
    m = torch.tensor(IMAGENET_DEFAULT_MEAN)
    s = torch.tensor(IMAGENET_DEFAULT_STD)

    images = """19_left_00.jpg
19_left_01.jpg
19_left_02.jpg
19_left_03.jpg
19_left_04.jpg
19_left_05.jpg
19_left_06.jpg
19_left_07.jpg
19_left_08.jpg
19_left_09.jpg
19_left_10.jpg
19_right_00.jpg
19_right_01.jpg
19_right_02.jpg
19_right_03.jpg
19_right_04.jpg
19_right_05.jpg
19_right_06.jpg
19_right_07.jpg
19_right_08.jpg
19_right_09.jpg
19_right_10.jpg
19_right_11.jpg
19_right_12.jpg
19_right_13.jpg
19_right_14.jpg""".split('\n')

    train_data = Data('/media/palm/Data/doc_denoise/denoising-dirty-documents/train')
    val_data = Data('/media/palm/Data/doc_denoise/denoising-dirty-documents/val')
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn
                              )
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=collate_fn_val
                            )

    model = Model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    schedule = CosineLRScheduler(optimizer,
                                 t_initial=n_epochs,
                                 cycle_mul=1,
                                 lr_min=1e-8,
                                 cycle_decay=0.1,
                                 warmup_lr_init=1e-5,
                                 warmup_t=3,
                                 cycle_limit=1,
                                 t_in_epochs=True,
                                 noise_range_t=None,
                                 )
    min_losses = torch.inf
    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        progbar = tf.keras.utils.Progbar(len(train_loader))
        model.train()
        for idx, (x, cls) in enumerate(train_loader):
            x = x.to(device)
            cls = cls.to(device)
            logit = model(x)
            optimizer.zero_grad()
            loss = criterion(logit, cls)
            loss.backward()
            optimizer.step()
            printlog = [('loss1', loss.cpu().detach().numpy()),
                        ]
            progbar.update(idx + 1, printlog)
        schedule.step(epoch + 1)
        model.eval()
        progbar = tf.keras.utils.Progbar(len(val_loader))
        losses = []
        with torch.no_grad():
            for idx, (x, cls) in enumerate(val_loader):
                x = x.to(device)
                cls = cls.to(device)
                logit = model(x)
                loss1 = criterion(logit, cls)
                losses.append(loss1.cpu().numpy())
                printlog = [('loss1', loss1.cpu().detach().numpy()),
                            ]
                progbar.update(idx + 1, printlog)
        if np.mean(losses) < min_losses:
            min_losses = np.mean(losses)
        torch.save(model.state_dict(), os.path.join(out_path, f'cp/{epoch:02d}.pth'))
        os.makedirs(os.path.join(out_path, f'image/{epoch:02d}'), exist_ok=True)
        with torch.no_grad():
            for impath in images:
                image = Image.open(os.path.join(src_path, impath)).convert('RGB')
                image = val_data.aug(image).unsqueeze(0)
                # size = int(image.size(3)/32)*32
                # image = image[..., :size]
                logit = model(image.to(device)).cpu()[0]
                im = logit.permute(1, 2, 0).mul_(s).add_(m).numpy()
                im = np.clip(im, 0, 1)
                plt.imsave(os.path.join(out_path, f'image/{epoch:02d}', impath), im)

