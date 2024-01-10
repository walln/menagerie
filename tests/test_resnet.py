"""Tests for ResNet model."""
from menagerie.resnet.model import ResNet18


def test_resnet():
    """Test ResNet model."""
    model = ResNet18()
    assert model is not None
