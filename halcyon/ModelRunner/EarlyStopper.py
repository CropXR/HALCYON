class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    def should_stop(self, val_loss) -> bool:
        """Call after each epoch. Returns True if training should stop.

        An improvement is only counted if val_loss dropped by at least
        min_delta, to avoid reacting to noise.
        """
        improved = val_loss < self.best_loss - self.min_delta

        if improved:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.patience

    def hyperparams_to_log(self) -> dict:
        return {"patience": self.patience, "min_delta": self.min_delta}
