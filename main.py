from config.experiments import EXPERIMENTS
from datasets.loader import DatasetLoader
from transforms.transforms import TRANSFORMS_REGISTRY, set_global_seed
from utils.logging import ExperimentLogger
from models.cnn import SimpleCNN, train_model, test_model, DATASET_NUM_CLASSES  # Added import
import torch
import time
from adaptation import ADAPTATION_REGISTRY


def run_experiment(config, logger):
    set_global_seed(2025)
    dataset_loader = DatasetLoader(seed=2025)
    
    train_transform = TRANSFORMS_REGISTRY.get(config.train_transform)
    test_transform = TRANSFORMS_REGISTRY.get(config.test_transform)
    
    train_loader, test_loader = dataset_loader.load_dataset(
        config.train_dataset,
        train_transform=train_transform,
        test_transform=test_transform
    )
    
    train_loader.dataset.save_samples(config.name)
    

    num_classes = DATASET_NUM_CLASSES[config.train_dataset]
    print(f"Creating model with {num_classes} output classes")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes).to(device)

    start_time = time.time()
    train_model(model, train_loader, device, config.epochs)
    train_time = time.time() - start_time

    accuracy, avg_confidence = test_model(
        model, 
        test_loader, 
        device,
        adaptation_method=getattr(config, 'adaptation_method', None) 
    )
    
    logger.log_metrics(
        experiment_name=config.name,
        accuracy=accuracy,
        avg_confidence=avg_confidence,
        adaptation_time=train_time
    )


def main():
    logger = ExperimentLogger("results.json")
    
    for config in EXPERIMENTS.values():
        print(f"\nRunning experiment: {config.name}")
        run_experiment(config, logger)


if __name__ == "__main__":
    main()