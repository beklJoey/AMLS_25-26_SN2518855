# A/model.py
# -----------------------------------------------------------------------------
# A simple CNN model for BreastMNIST (grayscale 1x28x28) binary classification.
#
# Although Task A in your report focuses on a classical ML pipeline (logistic
# regression on flattened / PCA features), this file defines a small CNN that
# can be used as a lightweight, course-aligned baseline architecture.
#
# Design goals:
#   - Keep the architecture minimal and easy to explain in a coursework report
#   - Use standard CNN building blocks:
#       Conv -> ReLU -> Pool  (repeated)
#       then a small MLP classifier head
#   - Provide a clear "capacity / regularisation knob" via dropout_p
#
# Input:
#   - x: torch.Tensor of shape (B, 1, 28, 28)
# Output:
#   - logits: torch.Tensor of shape (B, num_classes)
#     (raw scores before softmax; suitable for CrossEntropyLoss)
# -----------------------------------------------------------------------------

import torch.nn as nn


class SimpleCNN_A(nn.Module):
    """
    Small CNN mapping 1x28x28 images to `num_classes` logits.

    This model is intentionally compact:
      - Two convolution blocks for feature extraction
      - A small fully-connected head for classification

    Notes for reporting:
      - Convolutions introduce inductive biases (locality, weight sharing),
        which helps learning spatial features on images.
      - Pooling reduces spatial resolution and adds translation robustness.
      - Dropout acts as regularisation to reduce overfitting.
    """

    def __init__(self, num_classes=2, dropout_p=0.2):
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes. BreastMNIST is binary, so default is 2.
        dropout_p : float
            Dropout probability in the classifier head. This is a simple
            regularisation knob: higher p -> stronger regularisation.
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # Feature extractor: Conv -> ReLU -> Pool repeated twice
        #
        # Input shape:  (B, 1, 28, 28)
        # After Conv1:  (B, 16, 28, 28)  (padding=1 keeps spatial size)
        # After Pool1:  (B, 16, 14, 14)  (2x2 max pool halves H and W)
        # After Conv2:  (B, 32, 14, 14)
        # After Pool2:  (B, 32, 7, 7)
        #
        # Channel dimensions (1 -> 16 -> 32) control representational capacity:
        # more channels generally increases capacity (and compute).
        # ---------------------------------------------------------------------
        self.features = nn.Sequential(
            # First convolution:
            # - in_channels=1 because BreastMNIST images are grayscale
            # - out_channels=16 learns 16 feature maps
            # - kernel_size=3 gives a 3x3 receptive field (common default)
            # - padding=1 preserves 28x28 resolution
            nn.Conv2d(1, 16, 3, padding=1),

            # Non-linearity to allow learning non-linear decision boundaries
            nn.ReLU(),

            # Downsample spatial dimensions: 28x28 -> 14x14
            # Pooling helps reduce computation and adds some translation invariance
            nn.MaxPool2d(2),  # 14x14

            # Second convolution increases channel capacity: 16 -> 32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            # Further downsample: 14x14 -> 7x7
            nn.MaxPool2d(2),  # 7x7
        )

        # ---------------------------------------------------------------------
        # Classifier head: Flatten -> Dropout -> Linear -> ReLU -> Linear
        #
        # Flatten converts (B, 32, 7, 7) -> (B, 32*7*7) = (B, 1568)
        # Then:
        #   - Dropout reduces overfitting by randomly zeroing activations
        #   - Linear(1568 -> 64) learns a compact embedding
        #   - Final Linear(64 -> num_classes) outputs logits for classification
        #
        # NOTE:
        # - We return logits (not probabilities). For training, use CrossEntropyLoss
        #   which internally applies log-softmax.
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # Regularisation knob:
            # - dropout_p=0.2 means 20% of units are dropped during training
            # - disabled automatically in eval() mode
            nn.Dropout(dropout_p),

            # Fully-connected layer from flattened conv features to a hidden layer
            nn.Linear(32 * 7 * 7, 64),

            nn.ReLU(),

            # Output layer: logits for each class
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 28, 28)
            Batch of input images.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Unnormalised class scores. Apply softmax only for inference/visualisation
            if needed; for training use CrossEntropyLoss directly on logits.
        """
        # Extract spatial features using convolution + pooling blocks
        x = self.features(x)

        # Map features to class logits
        return self.classifier(x)

