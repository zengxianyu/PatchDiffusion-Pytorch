"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from PIL import Image

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
#import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from patch_diffusion import dist_util, logger
from patch_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

#def save_images(images, figure_path, figdims='4,4', scale='5'):
#    figdims = [int(d) for d in figdims.split(',')]
#    scale = float(scale)
#
#    if figdims is None:
#        m = len(images)//10 + 1
#        n = 10
#    else:
#        m, n = figdims
#
#    plt.figure(figsize=(scale*n, scale*m))
#
#    for i in range(len(images[:m*n])):
#        plt.subplot(m, n, i+1)
#        plt.imshow(images[i])
#        plt.axis('off')
#
#    plt.tight_layout()
#    plt.savefig(figure_path)
#    print(f"saved image samples at {figure_path}")

def save_output(out_path, arr0, label_arr=None, save_npz=True):
    Image.fromarray(arr0).save(out_path+".png")
    if save_npz:
        if label_arr is not None:
            np.savez(out_path+".npz", arr0[None,...], label_arr[None,...])
        else:
            np.savez(out_path+".npz", arr0[None,...])

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    
    model_names = args.model_path.split(",")
    models = []

    for model_name in model_names:
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(model_name, map_location="cpu")
        )

        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        models.append(model)


    if model.classifier_free and model.num_classes and args.guidance_scale != 1.0:
        model_fns = [diffusion.make_classifier_free_fn(model, args.guidance_scale) for model in models]

        def denoised_fn(x0):
            s = th.quantile(th.abs(x0).reshape([x0.shape[0], -1]), 0.995, dim=-1, interpolation='nearest')
            s = th.maximum(s, th.ones_like(s))
            s = s[:, None, None, None]
            x0 = x0.clamp(-s, s) / s
            return x0    
    else:
        model_fns = models
        denoised_fn = None

    logger.log("sampling...")
    N_sample = 0
    all_images = []
    all_labels = []
    while N_sample < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fns,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
            device=dist_util.dev()
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()

        out_path = args.save_dir
        logger.log(f"saving to {out_path}")
        for i in range(len(sample)):
            fname = str(N_sample+i).zfill(5)
            out_path_i = os.path.join(out_path, fname)
            save_output(out_path_i, sample[i])
            logger.log(f"saving to {out_path_i}")
        N_sample += sample.shape[0]


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        guidance_scale=1.5,
        save_dir="",
        figdims="4,4",
        figscale="5"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
