from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .metrics import classification_metrics
from .utils import save_json


def evaluate(model, graphs, device, batch_size: int = 4):
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    labels, preds = [], []
    total_loss = 0.0
    total_items = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1).to(device)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * y.numel()
            total_items += y.numel()

            labels.extend(y.cpu().tolist())
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(labels, preds)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def train(
    model,
    train_graphs,
    test_graphs,
    device,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    output_dir: str | Path = "results/runs/default",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            labels = batch.y.view(-1).to(device)

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        train_loss = running_loss / max(count, 1)
        test_metrics = evaluate(model, test_graphs, device, batch_size)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "metrics": test_metrics,
                },
                output_dir / "best.pt",
            )

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    save_json(history[-1], output_dir / "final_metrics.json")
    return history
