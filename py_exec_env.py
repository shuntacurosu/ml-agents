
import atexit
import dataclasses
import itertools
from multiprocessing import Process, Queue
from queue import Full
from typing import List
import gym
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np

from Logger import Logger
logger = Logger(__name__, loglevel=Logger.INFO)

@dataclasses.dataclass
class ActionType: CONTINUE = 0; DISCRETE = 1

NUM_AGENT = 4
COLS = 2
ROWS = 2

# 環境
FPS = 15

# ENV_NAME = 'Pendulum-v1'
# ACTION_TYPE = ActionType.CONTINUE
# OBS_SIZE = 3
# CONTINUOUS_SIZE = 1             # 行動が連続値の場合の行動数
# DISCRETE_BRANCHES = tuple()     # 行動が離散値の場合の行動数
# ACTION_COEF = 2.0

ENV_NAME = 'CartPole-v1'
ACTION_TYPE = ActionType.DISCRETE
OBS_SIZE = 4
CONTINUOUS_SIZE = 0             # 行動が連続値の場合の行動数
DISCRETE_BRANCHES = (2,)       # 行動が離散値の場合の選択可能なアクション数
ACTION_COEF = 1.0

# デバッグ
NB_STEP = 10000

class PyExecEnv:
    """ 環境"""
    def __init__(self):
        
        atexit.register(self.close)
        self._num_agents = NUM_AGENT
        self._before_done = [False for _ in range(NUM_AGENT)]

        # gymをマルチプロセスで立ち上げ
        self._envs = [gym.make(ENV_NAME, render_mode='rgb_array') for _ in range(self._num_agents)]

        # GUIを起動
        self._render_queue = Queue(maxsize=1)
        self._p = Process(target=self._update_frame, args=(self._render_queue,), daemon=True)
        self._p.start()
        
    def reset(self):
        ret = [AgentInfo(id=i, reset_info=env.reset()) for i,env in enumerate(self._envs)]
        self._update_render()
        return ret
    
    def step(self, actions):
        action_cnt = 0

        # reset
        ret:List[AgentInfo] = []
        for i, (is_before_done,env) in enumerate(zip(self._before_done, self._envs)):
            if is_before_done:
                # reset
                ret.append(AgentInfo(id=i, reset_info=env.reset()))
            else:
                # step
                if ACTION_TYPE == ActionType.CONTINUE:
                    action = actions[action_cnt] * ACTION_COEF
                elif ACTION_TYPE == ActionType.DISCRETE:
                    action = np.array(actions[action_cnt], dtype=int)[0]
                ret.append(AgentInfo(id=i, step_info=env.step(action)))
                action_cnt += 1

        # 今回の結果を保持
        self._before_done = [r.done for r in ret]

        self._update_render()
        return ret
    
    def close(self):
        [env.close() for env in self._envs]

    def _update_render(self):
        try:
            self._render_queue.put_nowait([env.render() for env in self._envs])
        except Full:
            pass

    def _update_frame(self, render_queue:Queue):
        interval = 1.0 / FPS
        fig, ax = plt.subplots(ROWS, COLS)
    
        # 初回フレーム
        render = render_queue.get()
        imgs: List[AxesImage] = []
        for i,(row,col) in enumerate(itertools.product(range(ROWS), range(COLS))):
            ax[row,col].axis('off')
            imgs.append(ax[row,col].imshow(render[i]))

        # フレームを更新
        while(True):
            render = render_queue.get()
            for i,img in enumerate(imgs):
                img.set_data(render[i])
            plt.pause(interval)

@dataclasses.dataclass
class AgentInfo:
    id: int
    obs: object
    reward: float = 0.0
    done: bool = False
    info: bool = False
    group_id: int = 0
    group_reward: float = 0.0
    max_step_reached: bool = False

    def __init__(self, id, reset_info=None, step_info=None):
        if (reset_info==None and step_info==None) or (reset_info!=None and step_info!=None):
            assert False, "reset_infoかstep_infoに値をいれる必要があります。"
        
        self.id = id

        if reset_info != None:
            # reset
            self.obs = reset_info[0]
        else:
            # step
            self.obs, self.reward, terminated, truncated, self.info = step_info
            self.done = terminated or truncated

# For python debugger to directly run this script
if __name__ == "__main__":
    import numpy as np
    import time
    env_cartpole = PyExecEnv()
    env_cartpole.reset()
    for i in range(NB_STEP):
        env_cartpole.step(np.random.randint(low=0, high=2, size=NUM_AGENT))
        logger.info(f"step: {i}/{NB_STEP}")
        time.sleep(0.1)
    env_cartpole.close()
