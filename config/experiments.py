from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class ExperimentConfig:
    name: str
    train_dataset: str
    test_dataset: str
    train_transform: Optional[str] = None
    test_transform: Optional[str] = None
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001

EXPERIMENTS = {
    "mnist_baseline": ExperimentConfig(
        name="MNIST Baseline",
        train_dataset="mnist",
        test_dataset="mnist",
        train_transform="identity",
        test_transform="identity"
    ),
    "colored_mnist": ExperimentConfig(
        name="Colored MNIST → MNIST",
        train_dataset="mnist",
        test_dataset="mnist",
        train_transform="colorize",
        test_transform="identity"
    ),
    "fashion_mnist_baseline": ExperimentConfig(
        name="Fashion-MNIST Baseline",
        train_dataset="fashion_mnist",
        test_dataset="fashion_mnist",
        train_transform="identity",
        test_transform="identity"
    ),
    "colored_fashion_mnist": ExperimentConfig(
        name="Colored Fashion-MNIST → Fashion-MNIST",
        train_dataset="fashion_mnist",
        test_dataset="fashion_mnist",
        train_transform="colorize",
        test_transform="identity"
    ),
    "emnist_baseline": ExperimentConfig(
        name="Extended MNIST Baseline",
        train_dataset="emnist",
        test_dataset="emnist",
        train_transform="identity",
        test_transform="identity"
    ),
    "colored_emnist": ExperimentConfig(
        name="Colored Extended MNIST → Extended MNIST",
        train_dataset="emnist",
        test_dataset="emnist",
        train_transform="colorize",
        test_transform="identity"
    )
}