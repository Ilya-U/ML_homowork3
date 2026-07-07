import torch
from datasets import get_mnist_loaders, get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters


def width_experiments(layer_1, layer_2, layer_3, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if dataset == "cifar":
        train_loader, test_loader = get_cifar_loaders(batch_size=64)
        input_size = 3072
    else:
        train_loader, test_loader = get_mnist_loaders(batch_size=64)
        input_size = 784

    model = FullyConnectedModel(
        input_size=input_size,
        num_classes=10,
        layers=[
            {"type": "linear", "size": layer_1},
            {"type": "linear", "size": layer_2},
            {"type": "linear", "size": layer_3}
        ]
    ).to(device)

    print(f"Model parameters: {count_parameters(model)}")

    history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))

    plot_training_history(history)
    return history

def perform_width_epxeriments(type_):
    widths = [64, 256, 1024, 2048]
    for i in widths:
        layer_1 = i
        second_layer_width = int(layer_1 / 2)
        third_layer_width = int(layer_1 / 4)
        width_experiments(layer_1, second_layer_width, third_layer_width, type_)

def grid_search():
    params = [
        [256, 128, 64],
        [265, 256, 256],
        [64, 128, 256],
        [64, 32, 16 ],
        [64, 64, 64],
        [16, 32, 64],
        [1024, 512, 256],
        [1024, 1024, 1024],
        [256, 512, 1024]
    ]
    best_val_acc = 0.0
    best_params = []
    all_accs = []
    for i in params:
        layer_1, layer_2, layer_3 = i
        result = width_experiments(layer_1, layer_2, layer_3, "MNIST")
        accs = result["test_accs"]
        val_acc = sum(accs) / len(accs)
        if val_acc > best_val_acc:
            best_params = i
        all_accs.append(val_acc)

    return best_params, all_accs



if __name__ == "__main__":
    best_params, all_accs = grid_search()
