import torch
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters


def depth_experiments(layers_quantity, layer_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[layer_type] * layers_quantity
    ).to(device)

    print(f"Model parameters: {count_parameters(model)}")

    history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))

    plot_training_history(history)


if __name__ == "__main__":
    depth_experiments(7, {"type": "linear", "size": 512})
