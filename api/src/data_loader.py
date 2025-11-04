import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir="data_split", batch_size=32, num_workers=2):
    """
    Create train, validation, and test dataloaders with transformations.
    """

    # Image transformations for different phases
    train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets from folders
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_transforms)
    test_data = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"✅ Loaded data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"Classes: {train_data.classes}")

    return train_loader, val_loader, test_loader, train_data.classes


if __name__ == "__main__":
    # quick test
    train_loader, val_loader, test_loader, classes = get_data_loaders()
    for images, labels in train_loader:
        print("Batch shape:", images.shape)
        print("Labels:", labels)
        break
