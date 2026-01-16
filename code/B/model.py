# B/model.py
# -----------------------------------------------------------------------------
# Task B (Deep Learning): Convolutional Neural Network (CNN) for BreastMNIST.
#
# This file defines a small but configurable CNN where "capacity" controls the
# number of channels (width) of the network:
#   - small:  16 base channels
#   - base:   32 base channels
#   - large:  64 base channels
#
# Channel scaling:
#   c1 = cap
#   c2 = 2*cap
#   c3 = 4*cap
#
# This directly implements the coursework "capacity axis" for Task B:
# increasing capacity -> more parameters/compute and potentially stronger
# representation, but also higher overfitting risk under small data budgets.
#
# Input:
#   - x: torch.Tensor of shape (B, in_channels, 28, 28)
# Output:
#   - logits: torch.Tensor of shape (B, num_classes)
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


class SimpleCNN_B(nn.Module):
    """
    A compact CNN with adjustable width ("capacity") for BreastMNIST-style inputs.

    Architectural pattern:
      (Conv -> BatchNorm -> ReLU -> Pool) x 2
      Conv -> ReLU -> Global pooling -> Linear classifier

    Design motivations:
      - BatchNorm helps stabilise optimisation and can reduce sensitivity to
        learning rate / initialization.
      - MaxPooling reduces spatial resolution and compute while building some
        translation robustness.
      - AdaptiveAvgPool2d((1,1)) implements "global average pooling", which:
          * removes dependence on fixed spatial resolution at the classifier head
          * reduces parameters compared to a large fully-connected layer
          * often improves generalisation for CNNs on small images
    """

    def __init__(self, num_classes=2, in_channels=1, capacity="base"):
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes (BreastMNIST is binary -> default 2).
        in_channels : int
            Number of input image channels (grayscale -> default 1).
        capacity : str
            Controls network width (number of channels). Must be one of:
            {"small", "base", "large"}.
        """
        super().__init__()

        # Map capacity string to the base channel width.
        # This is the main "capacity axis" knob for Task B.
        cap = {"small": 16, "base": 32, "large": 64}[capacity]

        # Channel sizes for the three convolution stages.
        # Increasing these increases representational capacity and compute.
        c1, c2, c3 = cap, cap * 2, cap * 4

        # ---------------------------------------------------------------------
        # Feature extractor (self.net)
        #
        # Input expected: (B, in_channels, 28, 28)
        #
        # Block 1:
        #   Conv(in_channels -> c1, kernel=3, pad=1): (B, c1, 28, 28)
        #   BatchNorm(c1):                            (B, c1, 28, 28)
        #   ReLU:                                     (B, c1, 28, 28)
        #   MaxPool(2):                               (B, c1, 14, 14)
        #
        # Block 2:
        #   Conv(c1 -> c2, kernel=3, pad=1):          (B, c2, 14, 14)
        #   BatchNorm(c2):                            (B, c2, 14, 14)
        #   ReLU:                                     (B, c2, 14, 14)
        #   MaxPool(2):                               (B, c2, 7, 7)
        #
        # Block 3:
        #   Conv(c2 -> c3, kernel=3, pad=1):          (B, c3, 7, 7)
        #   ReLU:                                     (B, c3, 7, 7)
        #   AdaptiveAvgPool((1,1)):                   (B, c3, 1, 1)
        #
        # After global pooling, each channel becomes a single scalar feature.
        # ---------------------------------------------------------------------
        self.net = nn.Sequential(
            # ----- Block 1 -----
            nn.Conv2d(in_channels, c1, 3, padding=1),  # preserve 28x28
            nn.BatchNorm2d(c1),                        # stabilise training / scale features
            nn.ReLU(inplace=True),                     # non-linearity (inplace saves memory)
            nn.MaxPool2d(2),                           # 28x28 -> 14x14

            # ----- Block 2 -----
            nn.Conv2d(c1, c2, 3, padding=1),           # preserve 14x14
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 14x14 -> 7x7

            # ----- Block 3 -----
            nn.Conv2d(c2, c3, 3, padding=1),           # preserve 7x7
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),              # global average pooling -> 1x1
        )

        # Final linear classifier:
        # Input features after pooling: c3 (because tensor is (B,c3,1,1))
        # Output: num_classes logits
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_channels, 28, 28)
            Input batch of images.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Unnormalised class scores (logits). Use CrossEntropyLoss for training.
        """
        # Extract convolutional features: (B, in_channels, 28, 28) -> (B, c3, 1, 1)
        x = self.net(x)

        # Flatten spatial dims while keeping batch dimension:
        # (B, c3, 1, 1) -> (B, c3)
        x = torch.flatten(x, 1)

        # Map feature vector to class logits: (B, c3) -> (B, num_classes)
        return self.fc(x)
