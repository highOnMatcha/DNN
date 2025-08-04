"""
Discriminator architectures for Pokemon sprite generation.

This package provides discriminator models for adversarial training
including PatchGAN discriminators for Pix2Pix and CycleGAN.
"""

from .cyclegan import CycleGANDiscriminator
from .pix2pix import Pix2PixDiscriminator

__all__ = [
    "CycleGANDiscriminator",
    "Pix2PixDiscriminator",
]
