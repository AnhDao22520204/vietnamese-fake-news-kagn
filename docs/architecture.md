# Architecture

## Source-notebook implementation

```text
Article / post text
        │
        ├──────────────→ PhoBERT text encoder
        │                    │
        │                    └── [CLS] = 768-d
        │
        └──→ Graph construction
               │
               ├── root node = full text
               ├── 5 placeholder entity nodes
               └── root → entity edges
                         │
                         ↓
              PhoBERT node [CLS] embeddings
                         │
                         ↓
                   GCNConv 768→256
                         │
                    ReLU + Dropout
                         │
                         ↓
                   GCNConv 256→128
                         │
                         ↓
                   root node 128-d

Text 768-d + Graph root 128-d
             ↓
        Concatenate
             ↓
          896-d
             ↓
       Linear(896, 2)
             ↓
         Real / Fake
```
