import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_unet import ResNetEncoder, DoubleConv

def debug_decoder():
    print("Debugging decoder shapes...")
    
    # Get encoder outputs
    encoder = ResNetEncoder(12)
    x = torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        e1, e2, e3, e4 = encoder(x)
    
    print("Encoder outputs:")
    print(f"  e1: {e1.shape}")  # [2, 64, 16, 16]
    print(f"  e2: {e2.shape}")  # [2, 128, 8, 8]
    print(f"  e3: {e3.shape}")  # [2, 256, 4, 4]
    print(f"  e4: {e4.shape}")  # [2, 512, 2, 2]
    
    # Decoder step by step
    print("\\nDecoder steps:")
    
    # Decoder 4: e4 (512, 2,2) -> upsample to 4,4 then concat with e3 (256, 4,4)
    up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
    d4_up = up4(e4)
    print(f"  up4(e4): {d4_up.shape}")  # Should be [2, 512, 4, 4]
    
    # Concat with e3 skip connection
    d4_cat = torch.cat([d4_up, e3], dim=1)
    print(f"  cat up4+e3: {d4_cat.shape}")  # Should be [2, 512+256=768, 4, 4]
    
    # Apply conv block
    conv4 = DoubleConv(768, 256)
    d4 = conv4(d4_cat)
    print(f"  after conv4: {d4.shape}")   # Should be [2, 256, 4, 4]
    
    # Decoder 3: d4 (256, 4,4) -> upsample to 8,8 then concat with e2 (128, 8,8)
    up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
    d3_up = up3(d4)
    print(f"  up3(d4): {d3_up.shape}")      # Should be [2, 256, 8, 8]
    
    d3_cat = torch.cat([d3_up, e2], dim=1)
    print(f"  cat up3+e2: {d3_cat.shape}")  # Should be [2, 256+128=384, 8, 8]
    
    conv3 = DoubleConv(384, 128)
    d3 = conv3(d3_cat)
    print(f"  after conv3: {d3.shape}")     # Should be [2, 128, 8, 8]
    
    # Decoder 2: d3 (128, 8,8) -> upsample to 16,16 then concat with e1 (64, 16,16)
    up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
    d2_up = up2(d3)
    print(f"  up2(d3): {d2_up.shape}")        # Should be [2, 128, 16, 16]
    
    d2_cat = torch.cat([d2_up, e1], dim=1)
    print(f"  cat up2+e1: {d2_cat.shape}")    # Should be [2, 128+64=192, 16, 16]
    
    conv2 = DoubleConv(192, 64)
    d2 = conv2(d2_cat)
    print(f"  after conv2: {d2.shape}")       # Should be [2, 64, 16, 16]
    
    # Decoder 1: d2 (64, 16,16) -> upsample to 32,32 then concat with ??? (we don't have e0)
    # Actually, looking at standard UNet, we upsample and that's it for final layer
    up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
    d1_up = up1(d2)
    print(f"  up1(d2): {d1_up.shape}")        # Should be [2, 64, 32, 32]
    
    # Final conv to get to 1 channel
    final_conv = nn.Conv2d(64, 1, 1)
    d1 = final_conv(d1_up)
    print(f"  final conv: {d1.shape}")        # Should be [2, 1, 32, 32]
    
    # Need to upsample to 64x64
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    output = upsample(d1)
    print(f"  final upsample: {output.shape}")  # Should be [2, 1, 64, 64]

if __name__ == "__main__":
    debug_decoder()