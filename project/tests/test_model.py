import pytest
import torch

from src.models.model import create_model


@pytest.mark.parametrize("model_name", ["resnet18", "resnet34", "resnet50", "mobilenet_v2"])
def test_model_creation(model_name):
    model = create_model(model_name, num_classes=6, pretrained=False, device="cpu")
    assert model is not None

    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4, 6)


def test_model_output_range():
    model = create_model("resnet18", num_classes=6, pretrained=False, device="cpu")
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    probs = torch.softmax(output, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4))
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_model_gradient_flow():
    model = create_model("resnet18", num_classes=6, pretrained=False, device="cpu")
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1])

    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_labels)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
