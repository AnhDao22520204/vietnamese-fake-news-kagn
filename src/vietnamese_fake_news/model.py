from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import AutoModel, AutoTokenizer


class KAGNClassifier(nn.Module):
    """
    Clean modularization of the uploaded notebook architecture.

    Important:
    - The original notebook declares MultiheadAttention but does not use it.
    - This implementation preserves that fact rather than inventing a new attention path.
    """

    def __init__(
        self,
        num_classes: int = 2,
        text_model: str = "vinai/phobert-base",
        dropout: float = 0.5,
        max_length: int = 50,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.max_length = max_length

        # Preserved for alignment with original notebook.
        # It is intentionally not used in forward(), because the uploaded source
        # did not use it either.
        self.attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True,
        )

        self.gcn1 = GCNConv(768, 256)
        self.gcn2 = GCNConv(256, 128)
        self.dropout = dropout
        self.classifier = nn.Linear(128 + 768, num_classes)

    def forward(self, batch):
        device = next(self.parameters()).device

        # PyG Batch stores string attributes as a Python list.
        texts = list(batch.text)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        text_outputs = self.text_encoder(**inputs)
        text_cls = text_outputs.last_hidden_state[:, 0, :]

        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)

        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)

        # batch.ptr contains the start index of each graph.
        root_indices = batch.ptr[:-1].to(device)
        root_nodes = x[root_indices]

        combined = torch.cat([text_cls, root_nodes], dim=-1)
        return self.classifier(combined)
