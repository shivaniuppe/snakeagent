import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        score = self.training_env.get_attr('score')
        self.logger.record('Score', score[0])

        return True

