'''
Author: xiaoniu
Date: 2026-01-16 22:27:49
LastEditors: xiaoniu
LastEditTime: 2026-01-16 22:30:28
Description: To be continued.
'''
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
import torch.nn as nn
from transformer_layers import DownSample3D, FuserLayer, UpSample3D
from patch_embed import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
)
class Upsampler(nn.Module):
    '''
    Super Resolution for 2D data
    input(B,C_in,H,W) -> output(B,C_out,scale*H,scale*W)
    residual_to_LR: whether to add bicubic upsampled input to output
    '''
    
class SwinTransformer3D(nn.Module):

    def __init__(
        self,
        img_size=(721, 1440),
        surface_channels=4,
        surface_mask_channels=3,
        upper_air_channels=13,
        upper_air_variables=5,
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        use_upper_air=True,
        output_surface_channels=None,
        periodic_lon=True,
    ):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        self.use_upper_air = use_upper_air
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.surface_channels = surface_channels
        self.surface_mask_channels = surface_mask_channels
        self.upper_air_channels = upper_air_channels
        self.upper_air_variables = upper_air_variables

        
        if output_surface_channels is None:
            self.output_surface_channels = surface_channels
        else:
            self.output_surface_channels = output_surface_channels
        self.pl_upper = 0 if not use_upper_air else math.ceil(upper_air_channels / patch_size[0])
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size[1:],
            in_chans=surface_channels + surface_mask_channels,  # add
            embed_dim=embed_dim,
        )
        if self.use_upper_air:
            self.patchembed3d = PatchEmbed3D(
                img_size=(upper_air_channels, img_size[0], img_size[1]),
                patch_size=patch_size,
                in_chans=upper_air_variables,
                embed_dim=embed_dim,
            )
        else:
            self.patchembed3d = None
        patched_inp_shape = (
            self.pl_upper+1,    # self.pl_upper = math.ceil(13 / 2) = 7. 7+1=8
            math.ceil(img_size[0] / patch_size[1]),     # math.ceil(721 / 4) = 181
            math.ceil(img_size[1] / patch_size[2]),     # math.ceil(1440 / 4) = 360
        )

        self.layer1 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
            periodic_lon=periodic_lon,
        )

        patched_inp_shape_downsample = (
            self.pl_upper+1 ,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )
        self.downsample = DownSample3D(
            in_dim=embed_dim,
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
        )
        self.layer2 = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
            periodic_lon=periodic_lon,
        )
        self.layer3 = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
            periodic_lon=periodic_lon,
        )
        self.upsample = UpSample3D(
            embed_dim * 2, embed_dim, patched_inp_shape_downsample, patched_inp_shape
        )
        self.layer4 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
            periodic_lon=periodic_lon,
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D(
            img_size, patch_size[1:], 2 * embed_dim, self.output_surface_channels
        )
        if self.use_upper_air:
            self.patchrecovery3d = PatchRecovery3D(
                (self.upper_air_channels, img_size[0], img_size[1]), patch_size, 2 * embed_dim, self.upper_air_variables
            )
        else:
            self.patchrecovery3d = None

    def prepare_input(self, surface, surface_mask, upper_air=None):
        """Prepares the input to the model in the required shape.
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        if surface_mask.dim() == 3:
            surface_mask = surface_mask.unsqueeze(0).repeat(surface.shape[0], 1, 1, 1)
        if not self.use_upper_air:
            return torch.concat([surface, surface_mask], dim=1)
        assert upper_air is not None, "upper_air =True 时，upper_air 不能为空"
        upper_air = upper_air.reshape(
            upper_air.shape[0], -1, upper_air.shape[3], upper_air.shape[4]
        )
        return torch.concat([surface, surface_mask, upper_air], dim=1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """
        surface = x[:, :(self.surface_channels + self.surface_mask_channels), :, :]
        surface = self.patchembed2d(surface)
        if self.use_upper_air:
            upper_air = x[:, (self.surface_channels + self.surface_mask_channels):, :, :].reshape(x.shape[0], self.upper_air_variables, self.upper_air_channels, x.shape[2], x.shape[3])
            upper_air = self.patchembed3d(upper_air)
            x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        else:
            x = surface.unsqueeze(2)
        B, C, Pl, Lat, Lon = x.shape    #经过(2,4,4)的patch后，变成(B, C, 8, 181, 360)
        x = x.reshape(B, C, -1).transpose(1, 2) #(B, C, Pl*Lat*Lon) -> (B, Pl*Lat*Lon, C)

        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_surface = self.patchrecovery2d(output_surface)
        if not self.use_upper_air:
            return output_surface
        output_upper_air = output[:, :, 1:, :, :]
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air

if __name__ == "__main__":
    lat,lon = 64,128
    B = 1
    surface_channels = 7
    surface_mask_channels = 3
    upper_air_channels = 14
    upper_air_variables = 6

    surface = torch.randn(B,surface_channels,lat,lon)
    surface_mask = torch.randn(B,surface_mask_channels,lat,lon)
    upper_air = torch.randn(B,upper_air_variables,upper_air_channels,lat,lon)
    
    model = SwinTransformer3D(img_size=(lat,lon), 
                  surface_channels=surface_channels, 
                  surface_mask_channels=surface_mask_channels, 
                  upper_air_channels=upper_air_channels, 
                  upper_air_variables=upper_air_variables,
                  patch_size=(2,4,4),
                  embed_dim=48,
                  num_heads=(3, 6, 12, 6),
                  window_size=(2, 6, 12),
                  periodic_lon=False,
    )
    x = model.prepare_input(surface,surface_mask,upper_air)
    out_surface, out_upper_air = model(x)
    print("x.shape: ", x.shape)
    print("out_surface.shape: ", out_surface.shape)
    print("out_upper_air.shape: ", out_upper_air.shape)