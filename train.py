from task import MyTask4
from agents.DDPG import DDPG
task = MyTask4(init_pose=[0,0,9,0,0,0])

agent = DDPG(task,
                 gamma=0.99,
                 UONoise_para={'theta': 0.15, 'mu': 0, 'sigma': 0.05},
                 tau = 0.001,
                 max_episode=1000,
                 max_explore_eps=1000,
                 memory_warmup=2000
            )

agent.train()