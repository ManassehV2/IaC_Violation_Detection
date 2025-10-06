import torch

class SlidingWindowDataCollator:
    def __call__(self, features):
        batch = {"input_ids": [], "attention_mask": []}
        labels = []
        for f in features:
            batch["input_ids"].append(f["input_ids"])
            batch["attention_mask"].append(f["attention_mask"])
            if isinstance(f["labels"], list):
                labels.append(torch.tensor(f["labels"], dtype=torch.float32))
            else:
                labels.append(f["labels"])
        batch["labels"] = torch.stack(labels)
        batch["window_counts"] = [f["window_counts"] for f in features]
        return batch

class SimpleDataCollator:
    def __call__(self, features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float32),
        }