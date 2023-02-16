import dataclasses

import numpy as np


@dataclasses.dataclass
class Scores:
    scores: np.ndarray
    scores_on_corrupted: np.ndarray

    @property
    def max_scores(self):
        return np.max(self.scores, axis=-1)
        # in case not softmax already:
        # return np.max(tf.nn.softmax(self.scores, axis=-1), axis=-1)

    @property
    def max_scores_on_corrupted(self):
        return np.max(self.scores_on_corrupted, axis=-1)
        # in case not softmax already:
        # return np.max(tf.nn.softmax(self.scores_on_corrupted, axis=-1), axis=-1)

    @property
    def predicted_classes(self):
        return np.argmax(self.scores, axis=-1)

    @property
    def predicted_classes_on_corrupted(self):
        return np.argmax(self.scores_on_corrupted, axis=-1)
