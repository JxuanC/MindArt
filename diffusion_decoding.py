import h5py
import json
from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys
sys.path.append("../utils/")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

prompt = ["A image showing"]

def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgidxs",
        default=['all'],
        nargs="*",
        type=str,
        help="img idx"
    )
    parser.add_argument(
        "--promptids",
        default=[0],
        nargs="*",
        type=int,
        help="prompt id"
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default='sub-3'
    )

    # Set parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidxs = opt.imgidxs
    gpu = opt.gpu
    promptids = opt.promptids
    subject=opt.subject
    
    captdir = f'../../mindart/decoded/DIR/{subject}/captions/test_preds.json'
    with open(captdir, 'r') as file:
        pre_caption = json.load(file)
    scores_latent = np.load(f'../../mindart/decoded/DIR/{subject}/prediction.npy', allow_pickle=True).item()

    # Load Stable Diffusion Model
    config = './stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt = './stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    model = load_model_from_config(config, f"{ckpt}", gpu)

    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    n_iter = 200
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    batch_size = n_samples
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    outdir = f'../../mindart/decoded/DIR/{subject}/reconstruction/'
    os.makedirs(outdir, exist_ok=True)


    precision = 'autocast'
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    
    if(imgidxs[0] == 'all'):
        imgidxs = list(scores_latent.keys())
    if(promptids[0] == -1):
        promptids = range(len(prompt))
    for promptid in promptids:
        sample_path = os.path.join(outdir, f"prompt_{promptid}")
        os.makedirs(sample_path, exist_ok=True)
        for imgidx in imgidxs:
            imgarr = torch.Tensor(scores_latent[imgidx].reshape(4,40,40)).unsqueeze(0).to('cuda')
            
            # Generate image from Z
            precision_scope = autocast if precision == "autocast" else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        x_samples = model.decode_first_stage(imgarr)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            im = Image.fromarray(x_sample.astype(np.uint8)).resize((512,512))
            im = np.array(im)


            init_image = load_img_from_arr(im).to('cuda')
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

            # Load c (Semantics)
            caption = prompt[promptid] + [caption['pre_cap'] for caption in pre_caption if caption['image_id'].split('.')[0] == imgidx][0]
            c = model.get_learned_conditioning(caption)

            # Generate image from Z (image) + C (semantics)
            base_count = 0
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        for n in trange(n_iter, desc="Sampling"):
                            uc = model.get_learned_conditioning(batch_size * [""])

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{imgidx}_{base_count:03}.png"))    
                            base_count += 1


if __name__ == "__main__":
    main()
