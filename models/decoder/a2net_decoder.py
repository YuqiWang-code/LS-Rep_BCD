"""A2Net decoder components with zero-parameter SCTC calibration.

Components
----------
- NeighborFeatureAggregation: multi-scale feature aggregation.
- TemporalFusionModule: optional symmetric cross-temporal calibration followed
  by the original A2Net absolute-difference temporal fusion.
- Decoder: FPN-style decoder with supervised attention.

SCTC is part of the deploy graph. It has no learnable parameters and is not
removed at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_calibration import MultiScaleTemporalCalibration


class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super().__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        return self.relu(c_fuse + self.conv_identity(c))


class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super().__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d

        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_aggregation_s2 = FeatureFusionModule(
            self.mid_d * 2,
            self.in_d[1],
            self.out_d,
        )

        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_aggregation_s3 = FeatureFusionModule(
            self.mid_d * 3,
            self.in_d[2],
            self.out_d,
        )

        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_aggregation_s4 = FeatureFusionModule(
            self.mid_d * 3,
            self.in_d[3],
            self.out_d,
        )

        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_aggregation_s5 = FeatureFusionModule(
            self.mid_d * 2,
            self.in_d[4],
            self.out_d,
        )

    def forward(self, c2, c3, c4, c5):
        c2_s2 = self.conv_scale2_c2(c2)
        c3_s2 = self.conv_scale2_c3(c3)
        c3_s2 = F.interpolate(
            c3_s2,
            size=c2_s2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s2 = self.conv_aggregation_s2(torch.cat([c2_s2, c3_s2], dim=1), c2)

        c2_s3 = self.conv_scale3_c2(c2)
        c3_s3 = self.conv_scale3_c3(c3)
        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = F.interpolate(
            c4_s3,
            size=c3_s3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s3 = self.conv_aggregation_s3(
            torch.cat([c2_s3, c3_s3, c4_s3], dim=1),
            c3,
        )

        c3_s4 = self.conv_scale4_c3(c3)
        c4_s4 = self.conv_scale4_c4(c4)
        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = F.interpolate(
            c5_s4,
            size=c4_s4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        s4 = self.conv_aggregation_s4(
            torch.cat([c3_s4, c4_s4, c5_s4], dim=1),
            c4,
        )

        c4_s5 = self.conv_scale5_c4(c4)
        c5_s5 = self.conv_scale5_c5(c5)
        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], dim=1), c5)

        return s2, s3, s4, s5


class TemporalFeatureFusionModule(nn.Module):
    """Original A2Net difference encoder for one feature scale."""

    def __init__(self, in_d, out_d):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)

        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(
                self.in_d,
                self.in_d,
                kernel_size=3,
                stride=1,
                padding=7,
                dilation=7,
            ),
            nn.BatchNorm2d(self.in_d),
        )

        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(
                self.in_d,
                self.in_d,
                kernel_size=3,
                stride=1,
                padding=5,
                dilation=5,
            ),
            nn.BatchNorm2d(self.in_d),
        )

        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(
                self.in_d,
                self.in_d,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
            ),
            nn.BatchNorm2d(self.in_d),
        )

        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(
                self.in_d,
                self.out_d,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm2d(self.out_d),
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

    def forward(self, x1, x2):
        # The temporal calibration, when enabled, has already happened before
        # this point. The original A2Net temporal difference is unchanged.
        x = torch.abs(x1 - x2)

        x_branch1 = self.conv_branch1(x)

        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)

        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)

        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)

        return self.relu(self.conv_branch5(x) + x_branch4)


class TemporalFusionModule(nn.Module):
    """Four-scale temporal fusion with optional SCTC before differencing."""

    def __init__(
        self,
        in_d=32,
        out_d=32,
        calibration_mode: str = "none",
    ):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.calibration_mode = str(calibration_mode).lower()

        self.temporal_calibration = MultiScaleTemporalCalibration(
            mode=self.calibration_mode,
            num_scales=4,
        )

        self.tffm_x2 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x3 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x4 = TemporalFeatureFusionModule(self.in_d, self.out_d)
        self.tffm_x5 = TemporalFeatureFusionModule(self.in_d, self.out_d)

    def forward(
        self,
        x1_2,
        x1_3,
        x1_4,
        x1_5,
        x2_2,
        x2_3,
        x2_4,
        x2_5,
    ):
        temporal1, temporal2 = self.temporal_calibration(
            (x1_2, x1_3, x1_4, x1_5),
            (x2_2, x2_3, x2_4, x2_5),
        )

        c2 = self.tffm_x2(temporal1[0], temporal2[0])
        c3 = self.tffm_x3(temporal1[1], temporal2[1])
        c4 = self.tffm_x4(temporal1[2], temporal2[2])
        c5 = self.tffm_x5(temporal1[3], temporal2[3])

        return c2, c3, c4, c5


class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super().__init__()
        self.mid_d = mid_d
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)
        return x_out, mask


class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super().__init__()
        self.mid_d = mid_d

        self.sam_p5 = SupervisedAttentionModule(self.mid_d)
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.sam_p3 = SupervisedAttentionModule(self.mid_d)

        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(
            d4
            + F.interpolate(
                p5,
                size=d4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(
            d3
            + F.interpolate(
                p4,
                size=d3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(
            d2
            + F.interpolate(
                p3,
                size=d2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5


__all__ = [
    "FeatureFusionModule",
    "NeighborFeatureAggregation",
    "TemporalFeatureFusionModule",
    "TemporalFusionModule",
    "SupervisedAttentionModule",
    "Decoder",
]
