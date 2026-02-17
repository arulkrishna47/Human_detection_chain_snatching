import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snatch_dir = "dataset/chain_classifier/train/snatch"
    not_snatch_dir = "dataset/chain_classifier/train/not_snatch"

    snatch_count = len(os.listdir(snatch_dir))
    not_snatch_count = len(os.listdir(not_snatch_dir))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(
        root="dataset/chain_classifier/train",
        transform=transform
    )

    targets = np.array(train_data.targets)
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        sampler=sampler,
        num_workers=0
    )

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 32 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = CNN().to(device)

    pos_weight = torch.tensor([not_snatch_count / snatch_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("models/chain_snatch", exist_ok=True)

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(
            model.state_dict(),
            f"models/chain_snatch/epoch_{epoch+1}.pth"
        )

if __name__ == "__main__":
    main()
