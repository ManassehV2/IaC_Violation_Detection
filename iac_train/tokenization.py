from transformers import AutoTokenizer
import torch
from .state import State
from .preprocessing import enhanced_terraform_preprocessing

SPECIAL_TOKENS = [
    "<RESOURCE>", "<DATA>", "<PROVIDER>", "<MODULE>",
    "<SECURITY_OPEN>", "<WILDCARD>", "<PUBLIC_ACCESS>",
    "<INTERNET_ACCESS>", "<NO_ENCRYPTION>", "<ADMIN_ACCESS>", "<ROOT_ACCESS>"
]

def init_tokenizer(backbone: str):
    if State.tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
        State.tokenizer = tokenizer
    return State.tokenizer

def create_sliding_windows(text, window_size, stride, max_windows):
    tokenizer = State.tokenizer
    processed_text = enhanced_terraform_preprocessing(text)
    tokens = tokenizer.tokenize(processed_text)

    if len(tokens) <= window_size:
        return [processed_text]

    windows = []
    start_idx = 0
    boundary_tokens = {'<RESOURCE>', '<DATA>', '<PROVIDER>', '<MODULE>'}

    for _ in range(max_windows):
        end_idx = min(start_idx + window_size, len(tokens))
        if end_idx < len(tokens):
            for i in range(min(30, len(tokens) - end_idx)):
                if tokens[end_idx + i] in boundary_tokens:
                    end_idx += i
                    break

        window_tokens = tokens[start_idx:end_idx]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        windows.append(window_text)

        start_idx += stride
        if end_idx >= len(tokens):
            break

    return windows[:max_windows]

def tokenize_with_windows(batch, window_size=384, stride=192, max_windows=6):
    tokenizer = State.tokenizer
    input_ids_list, attention_masks_list, labels_list, window_counts = [], [], [], []

    for code, label in zip(batch["tf_code"], batch["labels"]):
        windows = create_sliding_windows(code, window_size - 2, stride, max_windows)

        window_input_ids, window_attention_masks = [], []
        for window in windows:
            enc = tokenizer(window, truncation=True, padding="max_length",
                            max_length=window_size, return_tensors="pt")
            window_input_ids.append(enc["input_ids"][0])
            window_attention_masks.append(enc["attention_mask"][0])

        while len(window_input_ids) < max_windows:
            padding_ids = torch.zeros(window_size, dtype=torch.long)
            padding_mask = torch.zeros(window_size, dtype=torch.long)
            window_input_ids.append(padding_ids)
            window_attention_masks.append(padding_mask)

        input_ids_list.append(torch.stack(window_input_ids))
        attention_masks_list.append(torch.stack(window_attention_masks))
        labels_list.append(torch.tensor(label, dtype=torch.float32))
        window_counts.append(len(windows))

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks_list,
        "labels": labels_list,
        "window_counts": window_counts
    }

def tokenize_simple(example, max_length=512):
    tokenizer = State.tokenizer
    processed_code = enhanced_terraform_preprocessing(example["tf_code"])
    enc = tokenizer(processed_code, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors=None)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": example["labels"]
    }