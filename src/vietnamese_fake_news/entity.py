from __future__ import annotations

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class VietnameseNER:
    """NER utility corresponding to the NER pipeline defined in the original notebook."""

    def __init__(self, model_id: str = "NlpHUST/ner-vietnamese-electra-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

    def extract(self, text: str):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        truncated = self.tokenizer.decode(
            inputs["input_ids"][0],
            skip_special_tokens=True,
        )
        results = self.pipeline(truncated)
        return [(r["word"], r["entity_group"]) for r in results]


def first_token_entities(text: str, count: int = 5):
    """
    Source-aligned fallback.

    The uploaded notebook's build_graph() does not call the NER pipeline.
    It takes the first five whitespace-separated tokens as entity placeholders.
    """
    return [(token, "ENTITY") for token in text.split()[:count]]
