import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


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


model_path = "./pretrain/dinov2_vitb14_pretrain.pth"
model = torch.load(model_path, map_location="cpu")
input_tensor = model["pos_embed"]
pos_embed_interp = interpolate_pos_encoding(input_tensor, 16, 16)
pos_embed = nn.Parameter(torch.zeros(1, 256))
pos_embed.data = pos_embed_interp
model["pos_embed"] = pos_embed
torch.save(model, "./pretrain/dinov2_vitb14_pretrain_embedding_interpolated.pth")
