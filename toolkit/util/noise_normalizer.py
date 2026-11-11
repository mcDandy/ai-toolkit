"""
Running mean training loss per noise level, shared by everything that wants a
"noise-adjusted" loss (extensions/live_learning_monitor, extensions/training_insights).

Raw diffusion loss depends far more on the timestep a sample happened to draw
than on the image itself. rel_loss = loss / (the run's running mean at that
timestep): 1.0 is typical, >1 harder than usual, well below 1 easy or memorised.

One instance per logger (get_normalizer), so all consumers see the same numbers.
Exactly one consumer calls update() per sample (the per-file loss writer when it
is installed); everyone else uses relative() to read without updating.
"""
from typing import List, Optional

BUCKETS = 20
MIN_COUNT = 8
ALPHA = 0.01

_ATTR = '_shared_noise_normalizer'


class NoiseNormalizer:
    def __init__(self):
        self.mean: List[float] = [0.0] * BUCKETS
        self.count: List[int] = [0] * BUCKETS

    @staticmethod
    def bucket(timestep: float) -> int:
        return min(BUCKETS - 1, max(0, int(float(timestep) / 1000.0 * BUCKETS)))

    def relative(self, loss: Optional[float], timestep: Optional[float]) -> Optional[float]:
        """rel_loss against the current mean, without updating it. None until
        the bucket has enough history."""
        if timestep is None or loss is None or not (loss == loss):
            return None
        b = self.bucket(timestep)
        if self.count[b] < MIN_COUNT or self.mean[b] <= 0:
            return None
        return loss / self.mean[b]

    def update(self, loss: Optional[float], timestep: Optional[float]) -> Optional[float]:
        """Causal rel_loss (against the mean before this sample), then folds the
        sample in: a cumulative mean while warming up, then an EMA that follows
        the run as the loss level drops."""
        rel = self.relative(loss, timestep)
        if timestep is None or loss is None or not (loss == loss):
            return rel
        b = self.bucket(timestep)
        self.count[b] += 1
        self.mean[b] += max(ALPHA, 1.0 / self.count[b]) * (loss - self.mean[b])
        return rel


def get_normalizer(owner) -> NoiseNormalizer:
    """The normaliser shared by everything attached to `owner` (the trainer's logger)."""
    norm = getattr(owner, _ATTR, None)
    if norm is None:
        norm = NoiseNormalizer()
        setattr(owner, _ATTR, norm)
    return norm
