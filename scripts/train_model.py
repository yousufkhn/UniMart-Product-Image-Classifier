# scripts/train_model.py (suggested)
import json
import sys
from pathlib import Path
# Ensure repo root is on sys.path so `src` imports work when running this script directly
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from api.src.data_loader import get_data_loaders
from api.src.model_builder import build_model
from api.src.model_utils import save_checkpoint, accuracy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)  # safe default for Windows
    p.add_argument("--data_dir", type=str, default="data_split")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, classes = get_data_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc = accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, path="checkpoints/best_model.pth")

            with open("checkpoints/best_model_classes.json", "w") as f:
                json.dump(classes, f)

    print("Training complete.")
    test_acc = accuracy(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()