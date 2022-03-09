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

from obs_adapter import NeighbourObsAdapter


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

use_neighbor = True
use_map = True

hist_fut = False
agg = False

if use_neighbor:
    lstm_fut = False
    use_attn = True
    query_step = 10
    neighbor_interface = AgentInterface(
                max_episode_steps=None,
                action=ActionSpaceType.Lane,
                waypoints=Waypoints(20))
    neighbor_spec = AgentSpec(
            interface=neighbor_interface,
            agent_builder=None,
            observation_adapter=observation_adapter,
            reward_adapter=reward_adapter,
            action_adapter=action_adapter,
            info_adapter=info_adapter,
        )
    if use_map:
        lstm_fut = False
        adapter = NeighbourObsAdapter(mode='map', N_steps=10, dim=4, neighbors=5,add_future_state=False,
        query_step=query_step,lstm_fut=False)
    elif hist_fut:
        lstm_fut = False
        adapter = NeighbourObsAdapter(mode='hist_fut', N_steps=10, dim=4, neighbors=5,add_future_state=True,
        query_step=query_step,lstm_fut=False)
    else:
        adapter = NeighbourObsAdapter(mode='mlp', N_steps=10, dim=4, neighbors=5,add_future_state=True,
        query_step=query_step,lstm_fut=lstm_fut)

    obsadapter = adapter.obs_adapter
else:
    neighbor_spec , obsadapter = None,None
    lstm_fut = False
    use_attn=False

AGENT_ID = "Agent-LHC"
# env = gym.make(
#     "smarts.env:hiway-v0",
#     scenarios=["scenarios/left_turn_new"],
#     agent_specs=agent_specs,
# )
# scenario_paths = [["scenarios/roundabout_easy"],["scenarios/roundabout_medium"],["scenarios/roundabout"]]
scenario_paths = [["scenarios/double_merge/cross_test"]]
#"scenarios/double_merge/cross"
for scenario_path in scenario_paths:
    scenario_name = scenario_path[0].split('/')[-1]
    mode = 'lstmfut'
    if use_neighbor:
        if use_map:
            mode +="_map"
        elif hist_fut:
            mode +="_histfut"
        else:
            mode += f'_fut{query_step}'

    if use_attn:
        mode += '_attn'

    if agg:
        mode += '_agg'

    for i in [0,1,2]:
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
                        save_name=f'log_sac_{mode}_{scenario_name}_{i}_test',seed=i,obs_adapter=obsadapter, neighbor_spec=neighbor_spec,lstm_fut=lstm_fut,
                        use_attn=use_attn,umap=use_map,hist_fut=hist_fut,agg=agg)
        agent.run()
        env.close()
        agent.save_models(f'/home/haochen/SMARTS_test_TPDM/sac_model/_{mode}_{scenario_name}_{i}/')