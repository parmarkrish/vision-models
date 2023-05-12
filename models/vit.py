import torch
import torch.nn as nn
from transformer import Transformer

class ViT(nn.Module):
    def __init__(self, patch_size, num_layers, hidden_dim, num_heads, mlp_size, num_classes):
        super().__init__()
        # 3 x 224 x 224 -> hidden_size * 224//patch_size * 224//patch_size
        # 3 x 224 x 224 -> 768 x (14 x 14) -> (14 x 14) x 768 (permute in forward)
        self.patchify_lin_proj = nn.Sequential(nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size),
                                               nn.Flatten(2))
        
        img_size = 224
        num_patches = (img_size // patch_size)**2
        self.cls_token = nn.Parameter(torch.rand(hidden_dim))
        self.pos_emb = nn.Parameter(torch.rand(num_patches, hidden_dim))

        self.transformer = Transformer(num_layers, hidden_dim, num_heads, mlp_size)
        self.mlp_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # (N, C, H, W) -> (N, D, S) -> (N, S, D)
        patch_emb = self.patchify_lin_proj(x).transpose(-1, -2)
        patch_pos_emb = patch_emb + self.pos_emb
        # TODO: fix below code
        patch_pos_emb = torch.vstack(self.cls_token, patch_pos_emb)
        out = self.transformer(patch_pos_emb)[:, 0, :] # (N, D)
        out = self.mlp_head(out)
        return out




