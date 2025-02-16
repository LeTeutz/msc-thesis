import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple
from adaptation.registry import ADAPTATION_REGISTRY

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

DATASET_NUM_CLASSES = {
    'mnist': 10,
    'fashion_mnist': 10,
    'emnist': 47 ,
    'mnist_100': 10
}

def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                device: torch.device,
                epochs: int = 5,
                learning_rate: float = 0.001) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


def test_model(model: nn.Module,
               test_loader: torch.utils.data.DataLoader,
               device: torch.device,
               adaptation_method: Optional[str] = None) -> Tuple[float, float]:
    print(f"\nDEBUG: test_model parameters:")
    print(f"adaptation_method: {adaptation_method} (type: {type(adaptation_method)})")
    print(f"Available adapters: {list(ADAPTATION_REGISTRY.keys())}")
    
    model.eval()
    correct = 0
    total = 0
    confidence_sum = 0.0
    
    adapter = None
    if adaptation_method is not None:
        print(f"DEBUG: Creating adapter for method: {adaptation_method}")
        adapter_cls = ADAPTATION_REGISTRY[adaptation_method]
        print(f"DEBUG: Using adapter class: {adapter_cls.__name__}")

        if adaptation_method == 't3a':
            num_classes = model.fc_layers[-1].out_features  
            adapter = adapter_cls(model, device, num_classes=num_classes)
        else:
            adapter = adapter_cls(model, device)
    else:
        print("DEBUG: No adaptation method specified")

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if adapter is not None:
                with torch.enable_grad(): 
                    outputs = adapter.adapt_and_predict(inputs)
            else:
                outputs = model(inputs)
            
            probabilities = F.softmax(outputs, dim=1)
            pred_prob, predicted = torch.max(probabilities, 1)
            correct += (predicted == targets).sum().item()
            confidence_sum += pred_prob.sum().item()
            total += targets.size(0)

    accuracy = correct / total
    avg_confidence = confidence_sum / total

    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Average Confidence: {avg_confidence:.4f}")

    return accuracy, avg_confidence