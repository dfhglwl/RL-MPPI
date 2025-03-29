import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, size=1e6):
        self.max_size = int(size)  # 修改属性名称
        self.buffer = deque(maxlen=self.max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):  # 添加魔术方法
        return len(self.buffer)

    @property
    def current_size(self):  # 重命名方法
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()