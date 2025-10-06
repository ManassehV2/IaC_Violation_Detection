import torch
from torch import nn
import torch.nn.functional as F

class SmartFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = self.alpha * focal_weight * bce
        return focal_loss.mean()

def _init_classifier(hidden_size, num_labels):
    clf = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(hidden_size, 768),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(768, 384),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(384, num_labels)
    )
    for m in clf:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    return clf

class EnhancedSlidingWindowClassifier(nn.Module):
    def __init__(self, base_model, num_labels, max_windows=6):
        super().__init__()
        self.base = base_model
        self.max_windows = max_windows
        hidden_size = base_model.config.hidden_size
        self.window_attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1)
        self.attention_pooling = nn.Linear(hidden_size, 1)
        self.classifier = _init_classifier(hidden_size, num_labels)
        self.criterion = SmartFocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, input_ids, attention_mask, labels=None, window_counts=None):
        batch_size = len(input_ids) if isinstance(input_ids, list) else input_ids.size(0)
        if isinstance(input_ids, list):
            if isinstance(input_ids[0], torch.Tensor):
                device = input_ids[0].device
            elif isinstance(input_ids[0], list) and len(input_ids[0]) > 0:
                device = input_ids[0][0].device if isinstance(input_ids[0][0], torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pis, pms = [], []
            for i in range(len(input_ids)):
                pis.append(input_ids[i].to(device) if isinstance(input_ids[i], torch.Tensor) else torch.tensor(input_ids[i], device=device))
                pms.append(attention_mask[i].to(device) if isinstance(attention_mask[i], torch.Tensor) else torch.tensor(attention_mask[i], device=device))
            input_ids = torch.stack(pis)
            attention_mask = torch.stack(pms)
        else:
            device = input_ids.device

        window_reprs = []
        for i in range(batch_size):
            sample_reprs = []
            actual_windows = window_counts[i] if window_counts is not None else self.max_windows
            for j in range(min(actual_windows, self.max_windows)):
                win_input = input_ids[i, j:j+1]
                win_mask = attention_mask[i, j:j+1]
                if win_mask.sum() > 0:
                    outputs = self.base(input_ids=win_input, attention_mask=win_mask)
                    hidden = outputs.last_hidden_state
                    scores = self.attention_pooling(hidden).squeeze(-1)
                    scores = scores.masked_fill(win_mask == 0, -1e4)
                    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                    win_repr = torch.sum(hidden * weights, dim=1)
                    sample_reprs.append(win_repr)
            if sample_reprs:
                if len(sample_reprs) > 1:
                    stacked = torch.stack(sample_reprs, dim=1)  # [B=1, W, H]
                    att, _ = self.window_attention(
                        stacked.transpose(0, 1), stacked.transpose(0, 1), stacked.transpose(0, 1)
                    )
                    combined = att.mean(dim=0)
                else:
                    combined = sample_reprs[0]
                window_reprs.append(combined)
            else:
                window_reprs.append(torch.zeros(1, self.base.config.hidden_size, device=device))
        final_repr = torch.cat(window_reprs, dim=0)
        logits = self.classifier(final_repr)
        loss = self.criterion(logits, labels.float()) if labels is not None else None
        return {"loss": loss, "logits": logits}

class SlidingWindowsNoAttentionClassifier(nn.Module):
    def __init__(self, base_model, num_labels, max_windows=6):
        super().__init__()
        self.base = base_model
        self.max_windows = max_windows
        hidden_size = base_model.config.hidden_size
        self.attention_pooling = nn.Linear(hidden_size, 1)
        self.classifier = _init_classifier(hidden_size, num_labels)
        self.criterion = SmartFocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, input_ids, attention_mask, labels=None, window_counts=None):
        batch_size = len(input_ids) if isinstance(input_ids, list) else input_ids.size(0)
        if isinstance(input_ids, list):
            if isinstance(input_ids[0], torch.Tensor):
                device = input_ids[0].device
            elif isinstance(input_ids[0], list) and len(input_ids[0]) > 0:
                device = input_ids[0][0].device if isinstance(input_ids[0][0], torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pis, pms = [], []
            for i in range(len(input_ids)):
                pis.append(input_ids[i].to(device) if isinstance(input_ids[i], torch.Tensor) else torch.tensor(input_ids[i], device=device))
                pms.append(attention_mask[i].to(device) if isinstance(attention_mask[i], torch.Tensor) else torch.tensor(attention_mask[i], device=device))
            input_ids = torch.stack(pis)
            attention_mask = torch.stack(pms)
        else:
            device = input_ids.device

        window_reprs = []
        for i in range(batch_size):
            sample_reprs = []
            actual_windows = window_counts[i] if window_counts is not None else self.max_windows
            for j in range(min(actual_windows, self.max_windows)):
                win_input = input_ids[i, j:j+1]
                win_mask = attention_mask[i, j:j+1]
                if win_mask.sum() > 0:
                    outputs = self.base(input_ids=win_input, attention_mask=win_mask)
                    hidden = outputs.last_hidden_state
                    scores = self.attention_pooling(hidden).squeeze(-1)
                    scores = scores.masked_fill(win_mask == 0, -1e4)
                    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                    win_repr = torch.sum(hidden * weights, dim=1)
                    sample_reprs.append(win_repr)
            if sample_reprs:
                if len(sample_reprs) > 1:
                    stacked = torch.stack(sample_reprs, dim=1)
                    combined = torch.mean(stacked, dim=1)
                else:
                    combined = sample_reprs[0]
                window_reprs.append(combined)
            else:
                window_reprs.append(torch.zeros(1, self.base.config.hidden_size, device=device))
        final_repr = torch.cat(window_reprs, dim=0)
        logits = self.classifier(final_repr)
        loss = self.criterion(logits, labels.float()) if labels is not None else None
        return {"loss": loss, "logits": logits}

class NoSlidingWindowClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.classifier = _init_classifier(hidden_size, num_labels)
        self.criterion = SmartFocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        loss = self.criterion(logits, labels.float()) if labels is not None else None
        return {"loss": loss, "logits": logits}