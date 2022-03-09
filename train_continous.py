import os
import gym
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='1'

from sacd.agent import SacdAgent,sac_lhc
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent_interface import NeighborhoodVehicles, RGB,Waypoints
from smarts.core.controllers import ActionSpaceType

n_experiments = 3
def reward_adapter(env_obs, env_reward):
    return env_reward

def action_adapter(model_action): 
    return model_action

def info_adapter(observation, reward, info):
    return observation.events

def observation_adapter(observation):
    return observation

max_episode_steps = 800
agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    waypoints=Waypoints(20),
    neighborhood_vehicles=NeighborhoodVehicles(radius=None),
    rgb=RGB(80, 80, 32/80),
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter)


neighbor_spec , obsadapter = None,None

AGENT_ID = "Agent-LHC"
# env = gym.make(
#     "smarts.env:hiway-v0",
#     scenarios=["scenarios/left_turn_new"],
#     agent_specs=agent_specs,
# )
# scenario_paths = [["scenarios/roundabout_easy"],["scenarios/roundabout_medium"],["scenarios/roundabout"]]
scenario_paths = ["scenarios/double_merge/cross_test"]

scenario_name = scenario_path[0].split('/')[-1]
mode = 'lstmfut'

for i in range(n_experiments):
    print(f'Progress: {i+1}/{n_experiments}')

    # create env/
    env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=i)
    env.agent_id = AGENT_ID

    env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    if use_neighbor:
        env.observation_space = adapter.OBSERVATION_SPACE
    else: 
        env.observation_space = gym.spaces.Box(low=-10000, high=10000, shape=(24,), dtype=np.float32)
        # env.observation_space = gym.spaces.Box(low=0, high=1, shape=(80,80,3), dtype=np.float32)
    # print(env.reset())
    print(f'OBS SHAPE:{env.observation_space.shape}')
    log_dir = f'/home/haochen/SMARTS_test_TPDM/sac_model/sac_log_{mode}_{scenario_name}'
    agent = sac_lhc.SAC_LHC(env,test_env=None,log_dir=log_dir,num_steps=100000,batch_size=32,
                    memory_size=20000,start_steps=5000,update_interval=1,target_update_interval=1000,
                    use_per=True,dueling_net=False,max_episode_steps=max_episode_steps,multi_step=3,continuous=True,action_space=env.action_space.shape,
                    obs_dim=env.observation_space.shape,cnn=False,simple_reward=True,use_value_net=True,target_entropy_ratio=1,use_cpprb=True,lstm=False,lstm_steps=5,
                    save_name=f'log_sac_{mode}_{scenario_name}_{i}_test',seed=i,obs_adapter=obsadapter, neighbor_spec=neighbor_spec,lstm_fut=False)
    agent.run()
    env.close()
    agent.save_models(f'/home/haochen/SMARTS_test_TPDM/sac_model/_{mode}_{scenario_name}_{i}/')