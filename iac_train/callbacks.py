from transformers import TrainerCallback

class LiveProgressCallback(TrainerCallback):
    def __init__(self):
        self.current_epoch = 0
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch) + 1
        print(f"\nStarting Epoch {self.current_epoch}/{args.num_train_epochs}")
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            parts = []
            if 'loss' in logs: parts.append(f"Loss: {logs['loss']:.4f}")
            if 'eval_f1_micro' in logs: parts.append(f"Val F1: {logs['eval_f1_micro']:.4f}")
            if 'eval_loss' in logs: parts.append(f"Val Loss: {logs['eval_loss']:.4f}")
            if parts:
                print("  " + " ".join(parts))
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if logs:
            msg = f"Epoch {self.current_epoch} complete"
            if 'eval_f1_micro' in logs:
                msg += f" | Val F1: {logs['eval_f1_micro']:.4f}"
            print(msg)

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=2, min_delta=0.005, monitor_metric="eval_f1_micro", fallback_metric="eval_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.fallback_metric = fallback_metric
        self.best_metric = None
        self.best_fallback = None
        self.wait_count = 0
        self.fallback_wait = 0
        self.stopped_epoch = 0
        self.metric_higher_better = monitor_metric in ["eval_f1_micro", "eval_f1_macro", "eval_precision", "eval_recall"]
        self.fallback_higher_better = fallback_metric in ["eval_f1_micro", "eval_f1_macro", "eval_precision", "eval_recall"]

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        current = logs.get(self.monitor_metric)
        fb = logs.get(self.fallback_metric)
        if current is None:
            print(f"[WARNING] Early stopping metric {self.monitor_metric} not found in logs")
            return

        if self.best_metric is None:
            self.best_metric = current
            self.best_fallback = fb
            print(f"[Early Stopping] Initial {self.monitor_metric}: {current:.4f}")
            return

        improved = False
        if self.metric_higher_better:
            if current > self.best_metric + self.min_delta:
                improved = True; self.best_metric = current
        else:
            if current < self.best_metric - self.min_delta:
                improved = True; self.best_metric = current

        if improved:
            self.wait_count = 0; self.fallback_wait = 0
            print(f"[Early Stopping] {self.monitor_metric} improved to {current:.4f} (best: {self.best_metric:.4f})")
        else:
            self.wait_count += 1
            print(f"[Early Stopping] No improvement in {self.monitor_metric}: {current:.4f} (best: {self.best_metric:.4f}) - Patience: {self.wait_count}/{self.patience}")

            if fb is not None and self.best_fallback is not None:
                worsened = False
                if self.fallback_higher_better:
                    if fb < self.best_fallback - self.min_delta:
                        worsened = True
                else:
                    if fb > self.best_fallback + self.min_delta:
                        worsened = True

                if worsened:
                    self.fallback_wait += 1
                    print(f"[Early Stopping] {self.fallback_metric} worsened: {fb:.4f} (was: {self.best_fallback:.4f}) - Count: {self.fallback_wait}")
                else:
                    self.fallback_wait = 0
                    self.best_fallback = fb

            should_stop, reason = False, ""
            if self.wait_count >= self.patience:
                should_stop, reason = True, f"{self.monitor_metric} did not improve for {self.patience} epochs"
            elif self.fallback_wait >= 2:
                should_stop, reason = True, f"{self.fallback_metric} worsened for 2 consecutive epochs"

            if should_stop:
                self.stopped_epoch = state.epoch
                control.should_training_stop = True
                print(f"\nEarly stopping triggered at epoch {int(state.epoch)}")
                print(f"   Reason: {reason}")
                print(f"   Best {self.monitor_metric}: {self.best_metric:.4f}\n")