import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) as used in ESRGAN."""

    def __init__(self, nf=32, gc=16, bias=True):
        """Initializes RDB block.

        Args:
            nf (int): Number of input/output features.
            gc (int): Growth channel count (growth rate).
            bias (bool): Enable bias in convolutions.
        """
        super().__init__()
        # gc: growth channel, i.e. intermediate channel width
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirical residual scaling factor as per ESRGAN paper
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB)."""

    def __init__(self, nf=32, gc=16):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Residual scaling
        return out * 0.2 + x

class GeneratorGAN1(nn.Module):
    """GAN-1 Generator (ERA5 25km -> ERA5-Land 10km)."""

    def __init__(self, in_nc=18, out_nc=1, nf=32, nb=4, gc=16):
        """Initializes GAN-1 Generator.

        Args:
            in_nc (int): Number of input channels (typically 18 raw bands).
            out_nc (int): Number of output channels (1 for precipitation).
            nf (int): Number of intermediate feature channels.
            nb (int): Number of RRDB blocks.
            gc (int): Growth channel count inside RDB blocks.
        """
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # Stack RRDB blocks
        self.rrdb_blocks = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_trunk = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling to intermediate 10km grid (size: 230 x 360)
        self.upsample = nn.Upsample(size=(230, 360), mode='bilinear', align_corners=False)
        
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        out = self.rrdb_blocks(fea)
        out = self.conv_trunk(out) + fea
        
        # Scale to 230 x 360
        out = self.upsample(out)
        
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out

class GeneratorGAN2(nn.Module):
    """GAN-2 Generator (Intermediate 10km -> High-res 5km)."""

    def __init__(self, in_nc=4, out_nc=1, nf=32, nb=4, gc=16):
        """Initializes GAN-2 Generator.

        Args:
            in_nc (int): Input channels (1 GAN-1 output + 3 physics/DEM channels).
            out_nc (int): Number of output channels (1 for precipitation).
            nf (int): Intermediate feature channels.
            nb (int): Number of RRDB blocks.
            gc (int): Growth channel count inside RDB blocks.
        """
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # Stack RRDB blocks
        self.rrdb_blocks = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_trunk = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling by 2x to final 5km target (size: 460 x 720)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        out = self.rrdb_blocks(fea)
        out = self.conv_trunk(out) + fea
        
        # Upsample by 2x to target
        out = self.upsample(out)
        
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out

class PatchGANDiscriminator(nn.Module):
    """Fully convolutional PatchGAN discriminator (Markovian discriminator)."""

    def __init__(self, in_nc=1, ndf=32):
        """Initializes the PatchGAN Discriminator.

        Args:
            in_nc (int): Number of input channels (typically 1 for precipitation).
            ndf (int): Base number of discriminator feature channels.
        """
        super().__init__()
        
        self.model = nn.Sequential(
            # input is (in_nc) x H x W
            nn.Conv2d(in_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (ndf) x H/2 x W/2
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (ndf*2) x H/4 x W/4
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (ndf*4) x H/8 x W/8
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (ndf*8) x H/8-something x W/8-something
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
            # Output represents patch probability score (no sigmoid - handled in logits loss)
        )

    def forward(self, x):
        return self.model(x)
