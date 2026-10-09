import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import BurgersDataset
from model import FNO1d


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D Fourier Neural Operator")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--modes", type=int, default=16, help="Fourier modes retained")
    parser.add_argument("--width", type=int, default=64, help="Model hidden channel width")
    parser.add_argument("--samples", type=int, default=1200, help="Total synthetic samples")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Execution Hardware: {device}")

    # Dataset generation & train/test split
    dataset = BurgersDataset(num_samples=args.samples, nx=256)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Model, Optimizer, Scheduler, Loss
    model = FNO1d(modes=args.modes, width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    loss_history = []

    print("[+] Initiating FNO Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_set)
        loss_history.append(epoch_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {epoch_loss:.6f}")

    # Final Evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            test_loss += criterion(preds, y_batch).item() * x_batch.size(0)

    print(f"[+] Training Completed. Final Test MSE: {test_loss / len(test_set):.6f}")

    # Save Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fno_1d_burgers.pth")
    print("[+] Model Checkpoint saved to checkpoints/fno_1d_burgers.pth")


if __name__ == "__main__":
    main()
