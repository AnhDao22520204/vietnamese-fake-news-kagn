from __future__ import annotations

import argparse

from transformers import AutoModel, AutoTokenizer

from src.vietnamese_fake_news.data import load_prepared_csv, split_dataset
from src.vietnamese_fake_news.entity import VietnameseNER
from src.vietnamese_fake_news.graph import GraphBuilder
from src.vietnamese_fake_news.model import KAGNClassifier
from src.vietnamese_fake_news.trainer import train
from src.vietnamese_fake_news.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train Vietnamese fake-news PhoBERT-GCN model.")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=50)
    p.add_argument("--entity-count", type=int, default=5)
    p.add_argument("--entity-mode", choices=["first_tokens", "ner"], default="first_tokens")
    p.add_argument("--bidirectional-edges", action="store_true")
    p.add_argument("--output-dir", default="results/runs/default")
    p.add_argument("--phobert-model", default="vinai/phobert-base")
    p.add_argument("--ner-model", default="NlpHUST/ner-vietnamese-electra-base")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    df = load_prepared_csv(args.data)
    train_df, test_df = split_dataset(df, args.test_size, args.seed)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.phobert_model)
    node_encoder = AutoModel.from_pretrained(args.phobert_model)
    ner = VietnameseNER(args.ner_model) if args.entity_mode == "ner" else None

    builder = GraphBuilder(
        tokenizer=tokenizer,
        encoder=node_encoder,
        max_length=args.max_length,
        entity_count=args.entity_count,
        entity_mode=args.entity_mode,
        ner=ner,
        bidirectional_edges=args.bidirectional_edges,
        device=device,
    )

    print("Building train graphs...")
    train_graphs = builder.build_dataframe(train_df)
    print("Building test graphs...")
    test_graphs = builder.build_dataframe(test_df)

    model = KAGNClassifier(
        text_model=args.phobert_model,
        max_length=args.max_length,
    ).to(device)

    train(
        model=model,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
