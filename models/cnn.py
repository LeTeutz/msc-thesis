import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple

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
    'emnist': 47  
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
               device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    confidence_sum = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
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