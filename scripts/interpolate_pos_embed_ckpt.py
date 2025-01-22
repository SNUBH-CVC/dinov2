import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf


def parse_args():
    parser = argparse.ArgumentParser(description="Interpolate position embeddings")
    parser.add_argument("--ckpt_path", type=str, default="./pretrain/dinov2_vitb14_pretrain.pth")
    parser.add_argument("--output_path", type=str, default="./pretrain/dinov2_vitb14_pretrain_embedding_interpolated.pth")
    parser.add_argument("--out_dim", type=int, default=256)
    return parser.parse_args()


def interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    # Interpolate the position embeddings without changing the first row (class token)
    patch_pos_embed = nnf.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)


def main():
    args = parse_args()
    model_weights = torch.load(args.ckpt_path)
    pos_embed = model_weights["pos_embed"]
    print("Original position embedding shape:", pos_embed.shape)

    width = height = args.out_dim // 14
    interpolated_pos_embed = interpolate_pos_encoding(pos_embed, width, height)
    pos_embed_param = nn.Parameter()
    pos_embed_param.data = interpolated_pos_embed
    model_weights["pos_embed"] = pos_embed_param
    print("Interpolated position embedding shape:", pos_embed_param.shape)
    torch.save(model_weights, args.output_path)


if __name__ == "__main__":
    main()