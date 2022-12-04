# TGAA_ViT群

from typing import Sequence, Union
import torch
import torch.nn as nn

from networks.modules.transformerblock_TGAA import TransformerBlock

class TGAA_ViTs(nn.Module):

    def __init__(
            self,
            hidden_size,
            mlp_dim,
            num_heads,
            dropout_rate,
    ) -> None:
        self.vith = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
        self.vitw = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
        self.vitd = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
        self.vithw = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
        self.vithd = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
        self.vitwd = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)

    def forward(self, x):
        # TGAA ViT的组织方式，不是简单的串联
        xph = self.patch_embedding_h(x)  # b h h_s 计算复杂度从 (hwd)^2 降低为 h^2， 空间从(hwd)降为h
        xpw = self.patch_embedding_w(x)  # b h h_s
        xpd = self.patch_embedding_d(x)  # b h h_s

        h_att = self.vit1(xph)  # b h h_s
        w_att = self.vit2(xpw)  # b h h_s
        d_att = self.vit3(xpd)  # b h h_s

        hw_att = self.vithw_1(torch.mul(h_att, w_att))
        hd_att = self.vithd_1(torch.mul(h_att, d_att))
        wd_att = self.vitwd_1(torch.mul(w_att, d_att))
        return hw_att * hd_att * wd_att
