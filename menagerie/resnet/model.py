"""ResNet [https://arxiv.org/abs/1512.03385] implemented in PyTorch."""
from typing import Dict, Tuple

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torchmetrics import Accuracy


class BasicBlock(nn.Module):
    """A basic block for the ResNet architecture.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolution.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolution.
        shortcut (nn.Sequential): The shortcut connection (if stride != 1 or in_planes != out_planes).
        expansion (int): The expansion factor for the number of channels in the last conv block.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """Initializes the BasicBlock.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride for the convolutional layers.
        """
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the block.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class Bottleneck(nn.Module):
    """A bottleneck block for the ResNet architecture, used in deeper networks (ResNet-50, 101, and 152).

    The bottleneck design reduces the number of parameters and computational complexity, allowing
    the network to go deeper without a significant increase in resource requirements.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer with a kernel size of 1x1.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolution.
        conv2 (nn.Conv2d): The second convolutional layer with a kernel size of 3x3.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolution.
        conv3 (nn.Conv2d): The third convolutional layer with a kernel size of 1x1.
        bn3 (nn.BatchNorm2d): Batch normalization after the third convolution.
        shortcut (nn.Sequential): The shortcut connection, used to match the dimensions if necessary.
        expansion (int): The expansion factor for the number of channels in the last conv block.
    """

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        """Initializes the Bottleneck block.

        Args:
            in_planes (int): Number of input channels.
            planes (int): Number of intermediate channels (the output channels are planes * expansion).
            stride (int): Stride for the second convolutional layer.
        """
        super().__init__()
        # The first convolution reduces the number of features (dimensionality reduction)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # The second convolution performs the main processing with a larger kernel size
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # The third convolution increases the number of features (dimensionality expansion)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut connection to match up dimensions for the residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the block.
        """
        # Applying the first convolution, batch normalization, and ReLU
        out = F.relu(self.bn1(self.conv1(x)))

        # Applying the second convolution, batch normalization, and ReLU
        out = F.relu(self.bn2(self.conv2(out)))

        # Applying the third convolution and batch normalization
        out = self.bn3(self.conv3(out))

        # Adding the shortcut connection and applying ReLU
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetModel(nn.Module):
    """The ResNet architecture.

    Attributes:
        layer1 (nn.Sequential): The first layer of blocks.
        layer2 (nn.Sequential): The second layer of blocks.
        layer3 (nn.Sequential): The third layer of blocks.
        layer4 (nn.Sequential): The fourth layer of blocks.
        linear (nn.Linear): Fully connected layer for classification.
    """

    def __init__(self, block, num_blocks, num_classes=10):
        """Initializes the ResNet.

        Args:
            block (BasicBlock): The type of block to use.
            num_blocks (list of int): Number of blocks in each layer.
            num_classes (int): Number of classes for classification.
        """
        super().__init__()
        self.in_planes = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Creating the four layers of blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Fully connected layer for classification
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int):
        """Creates a layer with the specified number of blocks.

        Args:
            block (BasicBlock): The type of block to use.
            planes (int): Number of output channels for the blocks.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the blocks in the layer.

        Returns:
            nn.Sequential: A Sequential container of blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # Append each block to the layer
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the ResNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # Initial convolution and batch normalization
        out = F.relu(self.bn1(self.conv1(x)))

        # Passing through the layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Apply global average pooling
        # Assuming out has shape [batch_size, channels, height, width]
        out = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)))

        # Flatten the output for the linear layer
        out = out.view(out.size(0), -1)

        # Final linear layer for classification
        out = self.linear(out)

        return out


class ResNet(pl.LightningModule):
    """The ResNet architecture implemented as a PyTorch Lightning Module.

    Attributes:
        model (nn.Module): The internal ResNet model.
        learning_rate (float): Learning rate for the optimizer.

    Methods:
        forward: Performs a forward pass through the network.
        training_step: Processes a single training batch.
        validation_step: Processes a single batch on the validation set.
        test_step: Processes a single batch on the test set.
        configure_optimizers: Configures the optimizers.
    """

    def __init__(self, block, num_blocks, num_classes=10, learning_rate=1e-3):
        """Initializes the ResNet module.

        Args:
            block (nn.Module): The type of block to use (BasicBlock or Bottleneck).
            num_blocks (list of int): Number of blocks in each layer.
            num_classes (int): Number of classes for classification.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.model = ResNetModel(block, num_blocks, num_classes)
        self.loss_function = F.cross_entropy
        self.learning_rate = learning_rate
        self.train_accuracy = Accuracy("multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy("multiclass", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor from the network.
        """
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        """Processes a single training batch.

        Args:
            batch (tuple): A tuple containing input data and labels.
            batch_idx (int): The index of the batch.

        Returns:
            Tensor: The loss for the batch.
        """
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.loss_function(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Processes a single batch on the validation set.

        Args:
            batch (tuple): A tuple containing input data and labels.
            batch_idx (int): The index of the batch.
        """
        x, y = batch["image"], batch["label"]
        logits = self.forward(x)
        loss = self.loss_function(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("val_loss", loss)

    def test_step(self, batch: Tuple, batch_idx: int):
        """Processes a single batch on the test set.

        Args:
            batch (tuple): A tuple containing input data and labels.
            batch_idx (int): The index of the batch.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_function(logits, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        """Configures the optimizers (and optionally learning rate schedulers).

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


def ResNet18(classes: int = 10) -> ResNet:
    """Constructs a ResNet-18 model.

    Returns:
        ResNet: The constructed ResNet-18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=classes)


def ResNet34(classes: int = 10) -> ResNet:
    """Constructs a ResNet-34 model.

    Returns:
        ResNet: The constructed ResNet-34 model.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=classes)


def ResNet50(classes: int = 10) -> ResNet:
    """Constructs a ResNet-50 model.

    Returns:
        ResNet: The constructed ResNet-50 model.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=classes)


def ResNet101(classes: int = 10) -> ResNet:
    """Constructs a ResNet-101 model.

    Returns:
        ResNet: The constructed ResNet-101 model.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=classes)


def ResNet152(classes: int = 10) -> ResNet:
    """Constructs a ResNet-152 model.

    Returns:
        ResNet: The constructed ResNet-152 model.
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=classes)
