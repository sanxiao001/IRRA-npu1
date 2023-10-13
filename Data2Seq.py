
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from transformers.models.clip import CLIPTokenizer

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x


def zero_padding(text_tensor, tar_dim, device=None):
    padding_size = tar_dim - text_tensor.shape[1]
    zero_tensor = torch.zeros((text_tensor.shape[0], padding_size), device=device)
    padded_tensor = torch.cat([text_tensor, zero_tensor], dim=1)
    return padded_tensor

class Data2Seq(nn.Module):

    def __init__(self,modality,dim):
        super().__init__()
        self.modality = modality
        self.embed_dim = dim
        if self.modality == 'image' or self.modality == 'infrared':
            self.embed = PatchEmbed(embed_dim=self.embed_dim)
        elif self.modality == 'text':
            self.embed = CLIPTokenizer.from_pretrained("/home/ma-user/work/IRRA-npu1/openai / clip-vit-large-patch14")


    def forward(self,data):
        if self.modality in ['image', 'infrared', 'x-ray', 'video', 'graph', 'hyper', 'time-series', 'imu','text' ]:
            embeddings = self.embed(data)
        elif self.modality =='text':
            embeddings = self.embed(data)
            embeddings = zero_padding(text_tensor=embeddings, tar_dim = self.embed_dim)

        return embeddings