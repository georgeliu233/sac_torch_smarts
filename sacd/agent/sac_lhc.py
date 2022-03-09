import os
import numpy as np
import torch
from torch.optim import Adam

from .base import BaseAgent
from sacd.model import  CateoricalPolicy,QNetwork,ValueNetwork
from sacd.utils import disable_gradients,update_params


class SAC_LHC(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=3e-4, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0,tau=5e-3,obs_dim=[16],continuous=False,action_space=None,
                 cnn=False,simple_reward=False,use_value_net=False,use_cpprb=False,lstm=False,lstm_steps=1,
                 save_name=None,obs_adapter=None,neighbor_spec=None,lstm_fut=False,use_attn=False,
                 umap=False,hist_fut=False,agg=False):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed,obs_dim,continuous,action_space,cnn,simple_reward,
            use_value_net,use_cpprb,lstm,lstm_steps,save_name,obs_adapter,neighbor_spec)
        
        self.tau = tau
        self.cnn = cnn
        # Define networks.
        torch.autograd.set_detect_anomaly(True)
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.continuous = continuous
        self.lstm=lstm
        if not self.continuous:
            # self.policy = CateoricalPolicy(
            #     self.obs_dim[-1], self.env.action_space.n
            #     ,cnn=cnn).to(self.device)
            # self.online_critic = TwinnedQNetwork(
            # self.obs_dim[-1], self.env.action_space.n,
            # dueling_net=dueling_net,cnn=cnn,use_value_net=use_value_net).to(device=self.device)

            # self.target_critic = TwinnedQNetwork(
            # self.obs_dim[-1], self.env.action_space.n,
            # dueling_net=dueling_net,cnn=cnn,use_value_net=use_value_net).to(device=self.device).eval()
            pass
        else:
            self.policy = CateoricalPolicy(
                self.obs_dim[-1], 1,continuous=True,action_dim=env.action_space.shape[0],cnn=cnn,lstm=lstm,lstm_fut=lstm_fut,use_attn=use_attn
                ,umap=umap,hist_fut=hist_fut,agg=agg).to(self.device)

            self.Q1 = QNetwork(self.obs_dim[-1],num_actions=env.action_space.shape[0],cnn=cnn,continuous=True,lstm=lstm,
            lstm_fut=lstm_fut,use_attn=use_attn,umap=umap,hist_fut=hist_fut,agg=agg).to(device=self.device)
            
            self.Q2 = QNetwork(self.obs_dim[-1],num_actions=env.action_space.shape[0],cnn=cnn,continuous=True,lstm=lstm,
            lstm_fut=lstm_fut,use_attn=use_attn,umap=umap,hist_fut=hist_fut,agg=agg).to(device=self.device)

            self.value = ValueNetwork(self.obs_dim[-1],cnn=cnn,lstm=lstm,lstm_fut=lstm_fut,use_attn=use_attn,umap=umap
            ,hist_fut=hist_fut,agg=agg).to(device=self.device)
            self.value_target = ValueNetwork(self.obs_dim[-1],cnn=cnn,lstm=lstm,lstm_fut=lstm_fut,use_attn=use_attn,umap=umap
            ,hist_fut=hist_fut,agg=agg).to(device=self.device)

            self.value_target.load_state_dict(self.value.state_dict())
            print('Policy:')
            print(self.policy)
            print('Q1:')
            print(self.Q1)
            print('Value:')
            print(self.value)


        # Disable gradient calculations of the target network.
        disable_gradients(self.value_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.Q2.parameters(), lr=lr)

        self.value_optim = Adam(self.value.parameters(),lr=lr)

        self.use_value_net = use_value_net


        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        if self.continuous:
            self.target_entropy =-np.prod(self.env.action_space.shape).astype(np.float32) *target_entropy_ratio
        else:
            self.target_entropy = \
                -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, state):
        # Act with randomness.
        if self.cnn:
            state = np.ascontiguousarray(np.transpose(state,(2,0,1)),np.int32)
            state = torch.ByteTensor(
                state[None, ...]).to(self.device).float() #/ 255.
        else:
            state = torch.FloatTensor(
                state[None, ...]).to(self.device)
            
        with torch.no_grad():
            action, _ = self.policy.continuous_sample(state)
        return action.cpu().numpy()

    def exploit(self, state):
        # Act without randomness.
        if self.cnn:
            state = np.ascontiguousarray(np.transpose(state,(2,0,1)),np.int32)
            state = torch.ByteTensor(
                state[None, ...]).to(self.device).float() #/ 255.
        else:
            state = torch.FloatTensor(
                state[None, ...]).to(self.device)
        with torch.no_grad():
            action = self.policy.act(state)
        return action.cpu().numpy()

    def update_target(self):
        #self.target_critic.value_net.load_state_dict(self.online_critic.value_net.state_dict())
        #polyak update:
        for update_param,target_param in zip(self.value.parameters(),self.value_target.parameters()):
            target_param.data.copy_((1-self.tau)*target_param + self.tau*update_param)

    def train_body(self,batch):
        
        states, actions, rewards, next_states, dones = batch
        current_q1 = self.Q1(states, actions)
        current_q2 = self.Q2(states, actions)

        next_v_target = self.value_target(next_states)
        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.gamma * next_v_target

        td_loss_q1 = torch.mean((target_q - current_q1) **2)
        td_loss_q2 = torch.mean((target_q - current_q2) **2)

        current_v = self.value(states)
        sample_actions, logp = self.policy.continuous_sample(states)

        # print(sample_actions.shape)
        current_q1 = self.Q1(states, sample_actions)
        current_q2 = self.Q2(states, sample_actions)
        current_min_q = torch.min(current_q1, current_q2)

        # print(current_min_q.shape,current_q1.shape)
        
        with torch.no_grad():
            target_v = current_min_q - self.alpha * logp
        td_errors = target_v - current_v
        td_loss_v = torch.mean(td_errors ** 2)

        policy_loss = torch.mean(self.alpha * logp - current_min_q.detach())

        with torch.no_grad():
            loss_a = logp + self.target_entropy
        
        alpha_loss = -torch.mean(self.log_alpha * loss_a)

        update_params(self.q1_optim,td_loss_q1)
        update_params(self.q2_optim,td_loss_q2)

        update_params(self.value_optim,td_loss_v)
        self.update_target()

        update_params(self.policy_optim,policy_loss)

        update_params(self.alpha_optim,alpha_loss)
        self.alpha = self.log_alpha.exp()

    
    def compute_td_error(self,batch):
        states, actions, rewards, next_states, dones = batch
        current_q1 = self.Q1(states, actions)
        vf_next_target = self.value_target(next_states)

        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.gamma * vf_next_target

        td_errors_q1 = target_q - current_q1
        return torch.abs(td_errors_q1).detach()+ 1e-6

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass


    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    def calc_critic_loss(self, batch, weights):
        pass

    def calc_policy_loss(self, batch, weights):
        pass

    def calc_entropy_loss(self, entropies, weights):
        pass

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.Q1.save(os.path.join(save_dir, 'Q1.pth'))
        self.Q2.save(os.path.join(save_dir, 'Q2.pth'))
        self.value.save(os.path.join(save_dir, 'value.pth'))
        self.value_target.save(os.path.join(save_dir, 'target_value.pth'))
