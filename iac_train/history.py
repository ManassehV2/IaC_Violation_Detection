class TrainingHistoryTracker:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'learning_rate': [],
            'epochs': []
        }
        self.processed_steps = set()

    def update(self, trainer_state):
        if hasattr(trainer_state, 'log_history') and trainer_state.log_history:
            train_metrics, val_metrics = {}, {}
            for log in trainer_state.log_history:
                step = log.get('step', 0)
                if step in self.processed_steps:
                    continue
                if 'loss' in log and 'eval_loss' not in log:
                    epoch = log.get('epoch', 0)
                    train_metrics.setdefault(epoch, {})
                    train_metrics[epoch]['loss'] = log['loss']
                    if 'learning_rate' in log:
                        train_metrics[epoch]['learning_rate'] = log['learning_rate']
                if any(k.startswith('eval_') for k in log.keys()):
                    epoch = log.get('epoch', 0)
                    val_metrics.setdefault(epoch, {})
                    if 'eval_loss' in log: val_metrics[epoch]['loss'] = log['eval_loss']
                    if 'eval_f1_micro' in log: val_metrics[epoch]['f1'] = log['eval_f1_micro']
                    if 'eval_precision' in log: val_metrics[epoch]['precision'] = log['eval_precision']
                    if 'eval_recall' in log: val_metrics[epoch]['recall'] = log['eval_recall']
                self.processed_steps.add(step)
            for epoch in sorted(train_metrics.keys()):
                if epoch not in self.history['epochs']:
                    self.history['epochs'].append(epoch)
                    self.history['train_loss'].append(train_metrics[epoch].get('loss', 0))
                    self.history['learning_rate'].append(train_metrics[epoch].get('learning_rate', 0))
                    if epoch in val_metrics:
                        self.history['val_loss'].append(val_metrics[epoch].get('loss', 0))
                        self.history['val_f1'].append(val_metrics[epoch].get('f1', 0))
                        self.history['val_precision'].append(val_metrics[epoch].get('precision', 0))
                        self.history['val_recall'].append(val_metrics[epoch].get('recall', 0))
                        self.history['train_f1'].append(val_metrics[epoch].get('f1', 0))
                        self.history['train_precision'].append(val_metrics[epoch].get('precision', 0))
                        self.history['train_recall'].append(val_metrics[epoch].get('recall', 0))

    def get_final_history(self):
        lists = [v for v in self.history.values() if isinstance(v, list) and len(v) > 0]
        if not lists:
            return {k: [] for k in self.history.keys()}
        max_len = max(len(v) for v in lists)
        processed = {}
        for k, v in self.history.items():
            if isinstance(v, list):
                if len(v) < max_len:
                    v.extend([v[-1] if v else 0] * (max_len - len(v)))
                processed[k] = v[:max_len]
        return processed
