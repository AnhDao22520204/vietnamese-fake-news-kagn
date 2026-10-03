from __future__ import annotations

import torch
from torch_geometric.data import Data

from .entity import first_token_entities


class GraphBuilder:
    def __init__(
        self,
        tokenizer,
        encoder,
        max_length: int = 50,
        entity_count: int = 5,
        entity_mode: str = "first_tokens",
        ner=None,
        bidirectional_edges: bool = False,
        device: torch.device | None = None,
    ):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_length = max_length
        self.entity_count = entity_count
        self.entity_mode = entity_mode
        self.ner = ner
        self.bidirectional_edges = bidirectional_edges
        self.device = device or torch.device("cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

    def _entities(self, text: str):
        if self.entity_mode == "first_tokens":
            return first_token_entities(text, self.entity_count)
        if self.entity_mode == "ner":
            if self.ner is None:
                raise ValueError("entity_mode='ner' requires a VietnameseNER instance.")
            return self.ner.extract(text)
        raise ValueError(f"Unsupported entity mode: {self.entity_mode}")

    def build(self, post_text: str, label: int) -> Data:
        entities = self._entities(post_text)

        node_texts = [post_text] + [ent[0] for ent in entities]
        inputs = self.tokenizer(
            node_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()

        edges = [(0, i) for i in range(1, len(node_texts))]
        if self.bidirectional_edges:
            edges += [(i, 0) for i in range(1, len(node_texts))]

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(
            x=embeddings,
            edge_index=edge_index,
            y=torch.tensor([int(label)], dtype=torch.long),
        )
        data.text = post_text
        return data

    def build_dataframe(self, df):
        return [
            self.build(row["text"], int(row["label"]))
            for _, row in df.iterrows()
        ]
