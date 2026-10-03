from __future__ import annotations

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from src.vietnamese_fake_news.data import load_prepared_csv, split_dataset
from src.vietnamese_fake_news.graph import GraphBuilder
from src.vietnamese_fake_news.model import KAGNClassifier
from src.vietnamese_fake_news.trainer import evaluate
from src.vietnamese_fake_news.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--max-length", type=int, default=50)
    p.add_argument("--entity-count", type=int, default=5)
    p.add_argument("--phobert-model", default="vinai/phobert-base")
    p.add_argument("--bidirectional-edges", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    df = load_prepared_csv(args.data)
    _, test_df = split_dataset(df, args.test_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.phobert_model)
    node_encoder = AutoModel.from_pretrained(args.phobert_model)

    builder = GraphBuilder(
        tokenizer=tokenizer,
        encoder=node_encoder,
        max_length=args.max_length,
        entity_count=args.entity_count,
        entity_mode="first_tokens",
        bidirectional_edges=args.bidirectional_edges,
        device=device,
    )
    test_graphs = builder.build_dataframe(test_df)

    model = KAGNClassifier(
        text_model=args.phobert_model,
        max_length=args.max_length,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate(model, test_graphs, device, args.batch_size)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
