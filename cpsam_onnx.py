import numpy as np
import torch
from torch import nn
import cv2
import onnx
import onnxruntime
import matplotlib as mpl
import matplotlib.pyplot as plt
from cellpose import utils, io, models, plot, vit_sam
from cellpose.vit_sam import Transformer
import fill_voids
import os
import time
import argparse
from pathlib import Path
from natsort import natsorted

import logging
models_logger = logging.getLogger(__name__)

_CPSAM_MODEL_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = Path.home().joinpath(".cellpose", "models")
MODEL_DIR = Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT

def cache_CPSAM_model_path():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cached_file = os.fspath(MODEL_DIR.joinpath('cpsam'))
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(_CPSAM_MODEL_URL, cached_file))
        utils.download_url_to_file(_CPSAM_MODEL_URL, cached_file, progress=True)
    return cached_file

class CPSAMONNX(nn.Module):

    def __init__(self, pretrained_model="cpsam", device=None, use_bfloat16=False):
        super(CPSAMONNX, self).__init__()
        self.device = device
        self.diam_mean = 30
        if pretrained_model != "cpsam" and os.path.exists(pretrained_model):
            self.pretrained_model = pretrained_model
        else:
            pretrained_model = "cpsam"
            self.pretrained_model = os.path.join(MODEL_DIR, pretrained_model)
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.net = Transformer(dtype=dtype).to(self.device)
        if not os.path.exists(self.pretrained_model):
            cache_CPSAM_model_path()
        models_logger.info(f">>>> loading model {self.pretrained_model}")
        self.net.load_model(self.pretrained_model, device=self.device)

    def forward(self, img, img_size, channels, diameter, cellprob_threshold, niter):
        print("--- forward", img.shape, img_size, channels, diameter, cellprob_threshold, niter)
        img = torch.permute(img, (0, 2, 3, 1))
        print(img.shape)
        img_selected_channels = torch.zeros_like(img)
        img_selected_channels[:, :, :, :channels.shape[0]] = img[:, :, :, channels]
        img = img_selected_channels.clone()
        print(img.shape)
        img = img.squeeze()
        img = torch.permute(img, (2, 0, 1))
        percentiles = torch.zeros((2), dtype=torch.long, device=self.device)
        percentiles[0] = 1
        percentiles[1] = 99
        img = set_img_normalized(img, percentiles)
        img = torch.permute(img, (1, 2, 0))
        img = img[np.newaxis, ...]
        print(img.shape)

        print("--- _run_net begin")
        bsize = 256
        tile_overlap = 0.1
        yf, styles = self.run_net(self.net, img, img_size, bsize, tile_overlap, diameter)
        yf = torch.nn.functional.interpolate(
            yf,
            size=(img_size[1], img_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        yf = torch.permute(yf, (0, 2, 3, 1))
        cellprob = yf[..., 2]
        cellprob = cellprob[0]
        dP = yf[..., :2]
        dP = torch.permute(dP, (3, 0, 1, 2))
        dP = dP[:, 0]
        styles = styles.squeeze()
        print(dP.shape)
        print(cellprob.shape)
        print(styles.shape)
        print("--- _run_net end")

        mask, flow_errors = after_run_net(cellprob, dP, img_size, cellprob_threshold, niter, self.device)

        rgb_of_flows = torch.zeros((*dP.shape[1:], 3), dtype=torch.uint8, device=self.device)
        rgb_of_flows = dx_to_circ(dP, percentiles, rgb_of_flows)
        return mask, flow_errors, rgb_of_flows

    def run_net(self, net, imgi, img_size, bsize, tile_overlap, diameter):
        nout = net.nout
        Lz, Ly0, Lx0, nchan = imgi.shape 
        img_size_resize = img_size * self.diam_mean // diameter
        print(nout, img_size, img_size_resize)
        div = 16
        Lpad = div * torch.ceil(img_size_resize / div) - img_size_resize
        Lpad = Lpad.long()
        print(Lpad)
        ypad1 = div // 2 + Lpad[0] // 2
        ypad2 = div // 2 + Lpad[0] - Lpad[0] // 2
        xpad1 = div // 2 + Lpad[1] // 2
        xpad2 = div // 2 + Lpad[1] - Lpad[1] // 2
        pads = (ypad1, ypad2, xpad1, xpad2)
        print(pads)
        img_size_pad = img_size_resize.clone()
        img_size_pad[1] += ypad1 + ypad2
        img_size_pad[0] += xpad1 + xpad2
        img_size_pad = img_size_pad.long()
        print(img_size_pad)

        nyx = torch.ceil((1. + 2 * tile_overlap) * img_size_pad / bsize)
        nyx = nyx.long()
        nyx[img_size_pad <= bsize] = 1 
        print(nyx)

        lyx = img_size.clone()
        lyx[bsize <= img_size_pad] = bsize
        lyx = lyx.long()
        print(lyx)
        
        imgb = torch.permute(imgi, (0, 3, 1, 2))
        print(imgi.shape)
        print(imgb.shape)
        imgb = torch.nn.functional.interpolate(
            imgb,
            size=(img_size_resize[1], img_size_resize[0]),
            mode="bilinear",
            align_corners=False,
        )
        print(imgb.shape)
        imgb = torch.nn.functional.pad(imgb, pads)
        print(imgb.shape)

        tile_overlap = min(0.5, max(0.05, tile_overlap))
        ystart = torch.linspace(0, img_size_pad[0] - lyx[0], nyx[0], dtype=torch.long, device=self.device)
        xstart = torch.linspace(0, img_size_pad[1] - lyx[1], nyx[1], dtype=torch.long, device=self.device)
        ystart = ystart.long()
        xstart = xstart.long()
        print(ystart)
        print(xstart)

        IMG = torch.zeros((ystart.shape[0], xstart.shape[0], nchan, lyx[0], lyx[1]), dtype=net.dtype, device=self.device)
        print(IMG.shape)
        ysub = torch.zeros((ystart.shape[0] * xstart.shape[0], 2), dtype=torch.long, device=self.device)
        xsub = torch.zeros((ystart.shape[0] * xstart.shape[0], 2), dtype=torch.long, device=self.device)
        IMG= set_imgb_to_IMG(imgb, ystart, xstart, lyx, ysub, xsub, IMG)
        print(ysub)
        print(xsub)

        IMGa = torch.reshape(IMG, (nyx[0] * nyx[1], nchan, lyx[0], lyx[1]))
        print(IMGa.shape)
        ya = torch.zeros((nyx[0] * nyx[1], nout, lyx[0], lyx[1]), dtype=net.dtype, device=self.device)
        stylea = torch.zeros((nyx[0] * nyx[1], 256), dtype=net.dtype, device=self.device)        
        print(ya.shape)
        print(stylea.shape)
        ya, stylea = net_forward(net, IMGa)
        Navg = torch.zeros((img_size_pad[1], img_size_pad[0]), dtype=net.dtype, device=self.device)
        yfi = torch.zeros((ya.shape[1], img_size_pad[1], img_size_pad[0]), dtype=net.dtype, device=self.device)
        print(yfi.shape)

        sig = 7.5
        xm = torch.arange(bsize, dtype=net.dtype, device=self.device)
        xm = torch.abs(xm - torch.mean(xm))
        mask = 1 / (1 + torch.exp((xm - (bsize / 2 - 20)) / sig))
        mask = mask * mask[:, None]
        mask = mask[bsize // 2 - ya.shape[-2] // 2:bsize // 2 + ya.shape[-2] // 2 + ya.shape[-2] % 2,
                    bsize // 2 - ya.shape[-1] // 2:bsize // 2 + ya.shape[-1] // 2 + ya.shape[-1] % 2]
        yfi, Navg = set_yfi_Navg(ya, mask, ysub, xsub, yfi, Navg)

        yf = torch.zeros((Lz, nout, img_size_pad[1], img_size_pad[0]), dtype=torch.float32, device=self.device)
        styles = torch.zeros((Lz, 256), dtype=torch.float32, device=self.device)
        print(yf.shape)
        print(styles.shape)
        yfi /= Navg
        yf[0] = yfi
        stylei = torch.sum(stylea, dim=0)
        stylei /= torch.sum(stylei**2)**0.5
        styles[0] = stylei
        yf = yf[:, :, ypad1 : img_size_pad[1] - ypad2, xpad1 : img_size_pad[0] - xpad2]
        print(yf.shape)
        return yf, styles

def after_run_net(cellprob, dP, img_size, cellprob_threshold, niter, device):
    print("--- follow_flows begin")
    print(torch.max(cellprob))
    print(torch.min(cellprob))
    cellprob_bool = cellprob > cellprob_threshold.to(device)
    inds = get_inds(cellprob_bool)
    print(inds.shape)
    print(inds)
    p_final = follow_flows(dP * cellprob_bool / 5., inds, img_size, niter, device)
    p_final = p_final.long()
    print(p_final.shape)
    print(p_final)
    print("--- follow_flows end")

    print("--- get_masks_torch begin")
    max_size_fraction = 0.4
    mask = get_masks_torch(p_final, inds, dP.shape[1:], img_size, max_size_fraction, device)
    mask = torch.reshape(mask, (img_size[0], img_size[1]))
    del p_final
    print("--- get_masks_torch end")
    
    print("--- get_flow_errors begin")
    flow_errors = get_flow_errors(mask, dP, device)
    print("--- get_flow_errors end")
    return mask, flow_errors

def follow_flows(dP, inds, img_size, niter, device):
    ndim = img_size.shape[0]
    pt = torch.zeros((*[1]*ndim, inds.shape[1], ndim), dtype=torch.float32, device=device)
    print(pt.shape)
    im = torch.zeros((1, ndim, img_size[0], img_size[1]), dtype=torch.float32, device=device)
    print(im.shape)
    for n in range(ndim):
        pt[0, 0, :, ndim - n - 1] = inds[n]
        im[0, ndim - n - 1] = dP[n]
    img_size_minus_1 = img_size.clone()
    img_size_minus_1 -= 1
    img_size_minus_1 = img_size_minus_1.long()
    for k in range(ndim):
        im[:, k] *= 2. / img_size_minus_1[k]
        pt[..., k] /= img_size_minus_1[k]
    pt *= 2 
    pt -= 1
    pt = set_pt(im, niter, ndim, pt)
    pt += 1 
    pt *= 0.5
    for k in range(ndim):
        pt[..., k] *= img_size_minus_1[k]
    return pt[..., [1, 0]].squeeze().T

def get_masks_torch(pt, inds, shape0, img_size, max_size_fraction, device):
    print(pt.shape)
    print(inds.shape)
    print(shape0)
    ndim = len(shape0)
    
    rpad = 20
    pt += rpad
    pt = torch.clamp(pt, min=0)
    for i in range(pt.shape[0]):
        max_size = shape0[i]+rpad-1
        if type(max_size) is not int:
            max_size = max_size.to(device)
        pt[i] = torch.clamp(pt[i], max=max_size)
    t = torch.empty(shape0[0] + 2*rpad, shape0[1] + 2*rpad, device=device)
    shape = t.size()
    print(shape)
    img_size_pad = img_size + 2*rpad
    img_size_pad = img_size_pad.long()
    print(img_size_pad)

    output, counts = torch.unique(pt.t(), return_counts=True, dim=0)
    h1 = torch.zeros(shape, dtype=torch.long, device=device)
    pt0 = output.t()[0]
    pt1 = output.t()[1]
    pt_tuple = (pt0, pt1)
    h1[pt_tuple] = counts.long()

    hmax1 = max_pool_nd(h1.unsqueeze(0), img_size_pad, kernel_size=5)
    hmax1 = hmax1.squeeze()

    seeds1_tuple = torch.nonzero((h1 - hmax1 > -1e-6) * (h1 > 10), as_tuple=True)
    del hmax1
    npts = h1[seeds1_tuple]
    isort1 = torch.argsort(npts)
    seeds1_0 = seeds1_tuple[0]
    seeds1_0 = seeds1_0[isort1]
    seeds1_1 = seeds1_tuple[1]
    seeds1_1 = seeds1_1[isort1]
    seeds1 = torch.stack((seeds1_0, seeds1_1), dim=1)

    n_seeds = seeds1.shape[0]
    h_slc = torch.zeros((n_seeds, *[11]*ndim), device=device)
    h_slc = set_h_slc(h1, seeds1, h_slc)
    del h1

    seed_masks = torch.zeros((n_seeds, *[11]*ndim), device=device)
    seed_masks[:,5,5] = 1
    seed_masks_size = img_size.clone()
    seed_masks_size[0] = seed_masks.shape[1]
    seed_masks_size[1] = seed_masks.shape[2]
    seed_masks_size = seed_masks_size.long()
    for iter in range(5):
        seed_masks = max_pool_nd(seed_masks, seed_masks_size, kernel_size=3)
        seed_masks *= h_slc > 2
    del h_slc

    M1 = torch.zeros(shape, dtype=torch.long, device=device)
    M1 = set_M1(seeds1, seed_masks, M1)
    del seed_masks
    pt0 = pt[0]
    pt1 = pt[1]
    pt_tuple = (pt0, pt1)
    M1 = M1[pt_tuple]

    M0 = torch.zeros(shape0, dtype=torch.long, device=device)
    inds0 = inds[0]
    inds1 = inds[1]
    inds_tuple = (inds0, inds1)
    M0[inds_tuple] = M1
    uniq, counts = torch.unique(M0, return_counts=True)
    big = shape0[0] * shape0[1] * max_size_fraction
    bigc = uniq[counts > big]
    bigc = bigc[bigc > 0]
    M0 = set_labels_zero(bigc, M0)
    print(M0.shape)
    print(M0[M0 > 0])
    return M0

def get_flow_errors(mask, flows, device):
    print(mask.shape)
    print(flows.shape)
    dP_masks = masks_to_flows(mask, device)
    print(dP_masks.shape)
    print(dP_masks[dP_masks > 0])
    flow_errors = torch.zeros((torch.max(mask)), dtype=torch.float32, device=device)
    for i in range(dP_masks.shape[0]):
        error = (dP_masks[i] - flows[i] / 5.)**2
        m = torch.zeros((torch.max(mask)), dtype=torch.float32, device=device)
        m = set_flow_errors(error, mask, m)
        flow_errors += m
    return flow_errors

def masks_to_flows(masks, device):
    Ly0, Lx0 = masks.shape
    Ly, Lx = Ly0 + 2, Lx0 + 2
    masks = get_masks_if_max_0(masks)
    masks_padded = torch.nn.functional.pad(masks, (1, 1, 1, 1))
    shape = masks_padded.shape
    yx = torch.nonzero(masks_padded).t()
    neighbors = torch.zeros((2, 9, yx[0].shape[0]), dtype=torch.long, device=device)
    yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
    for i in range(9):
        neighbors[0, i] = yx[0] + yxi[0][i]
        neighbors[1, i] = yx[1] + yxi[1][i]
    isneighbor = torch.ones((9, yx[0].shape[0]), dtype=torch.bool, device=device)
    m0 = masks_padded[neighbors[0, 0], neighbors[1, 0]]
    for i in range(1, 9):
        isneighbor[i] = masks_padded[neighbors[0, i], neighbors[1, i]] == m0
    del m0, masks_padded

    labels_num = torch.max(masks)
    centers = torch.zeros((labels_num, 2), dtype=torch.long, device=device)
    ext = torch.zeros((labels_num), dtype=torch.long, device=device)
    centers, ext = set_find_objects(masks, centers, ext)
    meds_p = centers + 1
    T = torch.zeros(shape, dtype=torch.float32, device=device)
    mu = set_extend_centers(neighbors, isneighbor, meds_p, ext, T)
    del neighbors, isneighbor, meds_p

    mu /= (1e-60 + torch.sum(mu**2, dim=0)**0.5)
    mu0 = torch.zeros((2, Ly0, Lx0), dtype=torch.float32, device=device)
    mu0[:, yx[0] - 1, yx[1] - 1] = mu
    return mu0

def normalize99(img, percentiles):
    input = torch.flatten(img)
    in_sorted, in_argsort = torch.sort(input, dim=0)
    positions = percentiles * (input.shape[0]-1) / 100
    floored = torch.floor(positions)
    ceiled = floored + 1
    ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
    weight_ceiled = positions-floored
    weight_floored = 1.0 - weight_ceiled
    d0 = in_sorted[floored.long()] * weight_floored
    d1 = in_sorted[ceiled.long()] * weight_ceiled
    d0 += d1
    x01 = d0[0]
    x99 = d0[1]
    print(x01, x99)
    d2 = torch.max(d0 - torch.min(d0))
    img = torch.where(d2 > 0, (img - x01) / (x99 - x01), img)
    return img

@torch.jit.script
def set_img_normalized(img, percentiles):
    nchan = img.shape[0]
    for c in range(nchan):
        img[c] = normalize99(img[c], percentiles)
    return img

@torch.jit.script
def set_imgb_to_IMG(imgb, ystart, xstart, lyx, ysub, xsub, IMG):
    for j in range(ystart.shape[0]):
        for i in range(xstart.shape[0]):
            y0 = ystart[j]
            y1 = ystart[j] + lyx[0]
            x0 = xstart[i]
            x1 = xstart[i] + lyx[1]
            ysub[j * xstart.shape[0] + i, 0] = y0
            ysub[j * xstart.shape[0] + i, 1] = y1
            xsub[j * xstart.shape[0] + i, 0] = x0
            xsub[j * xstart.shape[0] + i, 1] = x1
            IMG[j, i] = imgb[0, :, y0:y1, x0:x1]
    return IMG

def net_forward(net, x):
    print("--- net_forward")
    print(x.shape)
    net.eval()
    with torch.no_grad():
        y, style = net(x)[:2]
    print(y.shape)
    print(style.shape)
    return y, style

@torch.jit.script
def set_yfi_Navg(ya, mask, ysub, xsub, yfi, Navg):
    for j in range(ysub.shape[0]):
        yfi[:, ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += ya[j] * mask
        Navg[ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += mask   
    return yfi, Navg

@torch.jit.script
def get_inds(cellprob_bool):
    inds = torch.nonzero(cellprob_bool)
    inds = torch.transpose(inds, 0, 1)
    if inds.shape[1] == 0:
        cellprob_bool[0:2, 0:2] = True
        inds = torch.nonzero(cellprob_bool)
        inds = torch.transpose(inds, 0, 1)
    return inds

@torch.jit.script
def get_masks_if_max_0(masks):
    if torch.max(masks) == 0:
        masks[0:2, 0:2] = 1
    return masks

@torch.jit.script
def set_pt(im, niter, ndim: int, pt):
    for t in torch.arange(niter[0]):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
        for k in range(ndim):
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1., 1.)
    return pt

def max_pool_nd(h, img_size, kernel_size=5):
    hmax = max_pool1d(h, img_size, kernel_size=kernel_size, axis=1)
    hmax2 = max_pool1d(hmax, img_size, kernel_size=kernel_size, axis=2)
    del hmax
    return hmax2

def max_pool1d(h, img_size, kernel_size=5, axis=1):
    out = h.clone()
    nd = img_size[axis - 1]    
    k0 = kernel_size // 2
    for d in range(-k0, k0+1):
        if axis==1:
            mv = out[:, max(-d,0):torch.min(nd-d,nd)]
            hv = h[:, max(d,0):torch.min(nd+d,nd)]
            out[:, max(-d,0):torch.min(nd-d,nd)] = torch.maximum(mv, hv)
        elif axis==2:
            mv = out[:, :, max(-d,0):torch.min(nd-d,nd)]
            hv = h[:,  :, max(d,0):torch.min(nd+d,nd)]
            out[:, :, max(-d,0):torch.min(nd-d,nd)] = torch.maximum(mv, hv)
    return out

@torch.jit.script
def set_h_slc(h1, seeds1, h_slc):
    for k in range(seeds1.shape[0]):
        h_slc[k] = h1[seeds1[k][0]-5:seeds1[k][0]+6, seeds1[k][1]-5:seeds1[k][1]+6]
    return h_slc

@torch.jit.script
def set_M1(seeds1, seed_masks, M1):
    for k in range(seed_masks.shape[0]):
        a = torch.nonzero(seed_masks[k])
        a0 = a.t()[0]
        a1 = a.t()[1]
        a0 += seeds1[k][0] - 5
        a1 += seeds1[k][1] - 5
        a = [a0, a1]
        M1[a0, a1] = 1 + k
    return M1

@torch.jit.script
def arrange_labels(M0):
    uniq, inverse_indices = torch.unique(M0, return_inverse=True)
    return inverse_indices

@torch.jit.script
def set_labels_zero(bigc, M0):
    for i in range(bigc.shape[0]):
        M0[M0 == bigc[i]] = 0
    M0 = arrange_labels(M0)
    return M0

@torch.jit.script
def find_objects(masks, slices):
    labels_num = torch.max(masks)
    for i in range(1, int(labels_num) + 1):
        mask_i = masks == i
        yxi = torch.nonzero(mask_i).t()
        ymin = torch.min(yxi[0])
        ymax = torch.max(yxi[0])
        xmin = torch.min(yxi[1])
        xmax = torch.max(yxi[1])
        slices[i - 1, 0] = ymin
        slices[i - 1, 1] = ymax
        slices[i - 1, 2] = xmin
        slices[i - 1, 3] = xmax
    return slices
    
@torch.jit.script
def set_find_objects(masks, centers, ext):
    labels_num = torch.max(masks)
    for i in range(1, int(labels_num) + 1):
        mask_i = masks == i
        yxi = torch.nonzero(mask_i).t()
        ymin = torch.min(yxi[0])
        ymax = torch.max(yxi[0])
        xmin = torch.min(yxi[1])
        xmax = torch.max(yxi[1])
        yxi = torch.nonzero(masks[ymin:ymax + 1, xmin:xmax + 1] == i).t()
        ymed = torch.mean(yxi[0].float())
        xmed = torch.mean(yxi[1].float())
        imin = torch.argmin(((yxi[1] - xmed)**2 + (yxi[0] - ymed)**2))
        ymed = yxi[0][imin] + ymin
        xmed = yxi[1][imin] + xmin
        centers[i - 1, 0] = ymed
        centers[i - 1, 1] = xmed
        ext[i - 1] = (ymax + 1 - ymin) + (xmax + 1 - xmin) + 2
    return centers, ext

@torch.jit.script
def set_extend_centers(neighbors, isneighbor, meds, ext, T):
    if ext.numel() == 0:
        niter = 0
    else:
        niter = 2 * torch.max(ext)
        niter = int(niter)
    meds0 = meds.t()[0]
    meds1 = meds.t()[1]
    for i in range(niter):
        T[meds0, meds1] += 1
        Tneigh = T[neighbors[0], neighbors[1]]
        Tneigh *= isneighbor
        T[neighbors[0, 0], neighbors[1, 0]] = torch.mean(Tneigh, dim=0)
    grads = T[neighbors[0, [2, 1, 4, 3]], neighbors[1, [2, 1, 4, 3]]]
    dy = grads[0] - grads[1]
    dx = grads[2] - grads[3]
    del grads
    mu_torch = torch.stack((dy, dx), dim=-2)
    return mu_torch

@torch.jit.script
def set_flow_errors(error, mask, m):
    labels_num = torch.max(mask)
    for i in range(1, int(labels_num) + 1):
        mask_i = mask == i
        yxi = torch.nonzero(mask_i)
        error_i = error * mask_i
        m[i - 1] = torch.sum(error_i[error_i > 0]) / yxi.shape[0]
    return m

@torch.jit.script
def dx_to_circ(dP, percentiles, rgb):
    mag = 255 * torch.clamp(normalize99(torch.sqrt(torch.sum(dP**2, dim=0)), percentiles), min=0, max=1)
    angles = torch.atan2(dP[1], dP[0]) + torch.pi
    a = 2
    mag /= a
    rgb[..., 0] = torch.clamp(mag * (torch.cos(angles) + 1), min=0, max=255)
    rgb[..., 1] = torch.clamp(mag * (torch.cos(angles + 2 * torch.pi / 3) + 1), min=0, max=255)
    rgb[..., 2] = torch.clamp(mag * (torch.cos(angles + 4 * torch.pi / 3) + 1), min=0, max=255)
    return rgb

def show(image_path, device):
    start = time.perf_counter()
    model = CPSAMONNX(device=device)
    print(f"load time: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    img = imread(image_path)
    img_original = img
    img_resized, img_size, channels, diameter, cellprob_threshold, niter = get_inputs(img, device=device)
    mask, flow_errors, rgb_of_flows = model.forward(img_resized, img_size, channels, diameter, cellprob_threshold, niter)
    print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    flow_threshold = 0.8
    min_size = 15
    mask = post_process(mask, flow_errors, flow_threshold, min_size)
    print(f"post_process time: {(time.perf_counter() - start) * 1000:.2f} ms")
    show_mask(img_original, img_size, mask, rgb_of_flows)

def export_onnx(pretrained_model, image_path, device):
    model = CPSAMONNX(device=device, pretrained_model=pretrained_model)
    img = imread(image_path)
    img_resized, img_size, channels, diameter, cellprob_threshold, niter = get_inputs(img, niter_default=20, device=device)
    torch.onnx.export(
        model,
        (
            img_resized,
            img_size,
            channels,
            diameter,
            cellprob_threshold,
            niter,
        ),
        "cpsam.onnx",
        verbose=False,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["img",  "img_size", "channels", "diameter", "cellprob_threshold", "niter"],
        output_names=["mask", "flow_errors", "rgb_of_flows"],
    )

def import_onnx(image_path, device):
    start = time.perf_counter()
    session, input_names, output_names = get_session("cpsam.onnx", device)
    img = imread(image_path)
    img_original = img
    img_resized, img_size, channels, diameter, cellprob_threshold, niter = get_inputs(img, device=device)
    inputs = [
        img_resized.cpu().numpy(), 
        img_size.cpu().numpy(), 
        channels.cpu().numpy(),
        diameter.cpu().numpy(), 
        cellprob_threshold.cpu().numpy(), 
        niter.cpu().numpy(), 
    ]
    print(f"load time: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    mask, flow_errors, rgb_of_flows = session.run(
        output_names, 
        {
        input_names[i]: inputs[i] for i in range(len(input_names))
        }
    )
    print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    mask = torch.from_numpy(mask)
    flow_errors = torch.from_numpy(flow_errors)
    rgb_of_flows = torch.from_numpy(rgb_of_flows)
    flow_threshold = 0.8
    min_size = 15
    mask = post_process(mask, flow_errors, flow_threshold, min_size)
    print(f"post_process time: {(time.perf_counter() - start) * 1000:.2f} ms")
    show_mask(img_original, img_size, mask, rgb_of_flows)

def get_session(onnx_path, device):
    print(onnxruntime.get_available_providers())
    if device.type == "cpu":
        providers=["CPUExecutionProvider"]
    else:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
            })
        ]
    session = onnxruntime.InferenceSession(
        onnx_path, 
        providers=providers
    )
    model_inputs = session.get_inputs()
    input_names = [
        model_inputs[i].name for i in range(len(model_inputs))
    ]
    input_shapes = [
        model_inputs[i].shape for i in range(len(model_inputs))
    ]
    model_outputs = session.get_outputs()
    output_names = [
        model_outputs[i].name for i in range(len(model_outputs))
    ]
    output_shapes = [
        model_outputs[i].shape for i in range(len(model_outputs))
    ]
    print(input_names)
    print(input_shapes)
    print(output_names)
    print(output_shapes)
    return session, input_names, output_names

def imread(image_path):
    img = cv2.imread(image_path)
    img = img[..., [2, 1, 0]]
    return img

def get_inputs(img, niter_default=200, device=torch.device("cpu")):
    img = cv2.resize(img, (512, 512))
    print(img.shape)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :].astype(np.float32)
    img = torch.from_numpy(img).to(device)
    print(img.shape)
    img_size = torch.tensor([img.shape[2], img.shape[3]], dtype=torch.long)
    print(img_size)
    channels = torch.tensor([1, 0], dtype=torch.long)
    print("channels", channels)
    diameter = torch.tensor([40], dtype=torch.long)
    print("diameter", diameter)
    cellprob_threshold = torch.tensor([0.0], dtype=torch.float32)
    print("cellprob_threshold", cellprob_threshold)
    niter = torch.tensor([niter_default], dtype=torch.long)
    print("niter", niter)
    return img, img_size, channels, diameter, cellprob_threshold, niter

def show_mask(img_original, img_size, mask, rgb_of_flows):
    show_original = True;
    if show_original:
        mask = torch.reshape(mask, (1, 1, mask.shape[0], mask.shape[1]))
        mask = mask.float()
        mask = torch.nn.functional.interpolate(
            mask,
            size=(img_original.shape[0], img_original.shape[1])
        )
        mask = mask.long()
        mask = mask.squeeze()
        rgb_of_flows = torch.permute(rgb_of_flows, (2, 0, 1))
        rgb_of_flows = rgb_of_flows.unsqueeze(0)
        rgb_of_flows = rgb_of_flows.float()
        rgb_of_flows = torch.nn.functional.interpolate(
            rgb_of_flows,
            size=(img_original.shape[0], img_original.shape[1])
        )
        rgb_of_flows = rgb_of_flows.long()
        rgb_of_flows = rgb_of_flows.squeeze()
        rgb_of_flows = torch.permute(rgb_of_flows, (1, 2, 0))
    mask = mask.detach().cpu().numpy()
    save_mask(mask)
    rgb_of_flows = rgb_of_flows.detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite("rgb_of_flows.jpg", rgb_of_flows)
    mpl.rcParams['toolbar'] = 'None'
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    if show_original:
        axes[0].imshow(img_original)
    else:
        img_resized = cv2.resize(img_original, (int(img_size[0]), int(img_size[1])))
        axes[0].imshow(img_resized)
    axes[1].imshow(rgb_of_flows)
    outlines_pred = utils.outlines_list(mask)
    for o in outlines_pred:
        axes[0].plot(o[:,0], o[:,1], color=[1,1,0.3], lw=0.75, ls="--")
    axes[0].axis('off')
    axes[1].axis('off')
    fig.tight_layout()
    fig.canvas.manager.set_window_title('Cellpose')
    plt.gcf().set_facecolor((41/255.0, 44/255.0, 47/255.0))
    plt.show()

def save_mask(mask):
    import colorsys
    mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    labels_num = mask.max()
    RGB_tuples = [colorsys.hsv_to_rgb(x*1.0/labels_num, 0.5, 0.5) for x in range(labels_num)]
    for i in range(1, int(labels_num) + 1):
        msk = mask == i
        mask_image[msk, :] = tuple([255*x for x in RGB_tuples[i - 1]])
    cv2.imwrite("mask.png", mask_image)

def post_process(mask, flow_errors, flow_threshold, min_size):
    print("--- post_process begin")
    print_mask(flow_errors, print_more=True)
    print_mask(mask)
    mask = remove_bad_flow_masks(mask, flow_errors, flow_threshold)
    labels_num = torch.max(mask)
    slices = torch.zeros((labels_num, 4), dtype=torch.long)
    slices = find_objects(mask, slices)
    mask = fill_holes_and_remove_small_masks(mask, min_size, slices)
    print_mask(mask)
    print("--- post_process end")
    return mask

def remove_bad_flow_masks(mask, flow_errors, flow_threshold):
    badi = torch.nonzero(flow_errors > flow_threshold).T[0]
    badi = 1 + badi
    print(badi.shape)
    print(badi)
    mask = set_labels_zero(badi, mask)
    return mask

def fill_holes_and_remove_small_masks(masks, min_size, slices):
    j = 0
    for i, slc in enumerate(slices):
        msk = masks[slc[0]:slc[1] + 1, slc[2]:slc[3] + 1] == (i + 1)
        npix = torch.sum(msk)
        if npix < min_size:
            masks[slc[0]:slc[1] + 1, slc[2]:slc[3] + 1][msk] = 0
        elif npix > 0:
            msk = msk.detach().cpu().numpy()
            msk = fill_voids.fill(msk)
            msk = torch.from_numpy(msk)
            masks[slc[0]:slc[1] + 1, slc[2]:slc[3] + 1][msk] = (j + 1)
            j += 1
    return masks

def print_mask(mask, print_more=False):
    print(mask.dtype)
    if mask.numel() != 0:
        print(mask.max())
    print(mask.shape)
    print(mask)
    if mask.shape == mask[mask > 0].shape:
        return
    print(mask[mask > 0].shape)
    if print_more:
        torch.set_printoptions(edgeitems=100)
    print(mask[mask > 0])
    if print_more:
        torch.set_printoptions(edgeitems=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,default="show",required=False,help="show/export/import")
    parser.add_argument("--pretrained_model",type=str,default="cpsam",required=False,help="model path")
    parser.add_argument("--image",type=str,default="../demo_images/img00.png",required=False,help="image path")
    parser.add_argument("--device",type=str,default="cpu",required=False,help="cpu or cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.mode == "show":
        show(args.image, device)
    elif args.mode == "export":
        export_onnx(args.pretrained_model, args.image, device)
    elif args.mode == "import":
        import_onnx(args.image, device)





























