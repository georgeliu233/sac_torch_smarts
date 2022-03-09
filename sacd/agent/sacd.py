import os
import numpy as np
import torch
from torch.optim import Adam

from .base import BaseAgent
from sacd.model import TwinnedQNetwork, CateoricalPolicy
from sacd.utils import disable_gradients


class SacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0,tau=0.005,obs_dim=[16],continuous=False,action_space=None,
                 cnn=False,simple_reward=False,use_value_net=False,use_cpprb=False):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed,obs_dim,continuous,action_space,cnn,simple_reward,
            use_value_net,use_cpprb)
        
        self.tau = tau
        self.cnn = cnn
        # Define networks.
        self.obs_dim = obs_dim
        self.continuous = continuous
        if not self.continuous:
            self.policy = CateoricalPolicy(
                self.obs_dim[-1], self.env.action_space.n
                ,cnn=cnn).to(self.device)
            self.online_critic = TwinnedQNetwork(
            self.obs_dim[-1], self.env.action_space.n,
            dueling_net=dueling_net,cnn=cnn,use_value_net=use_value_net).to(device=self.device)

            self.target_critic = TwinnedQNetwork(
            self.obs_dim[-1], self.env.action_space.n,
            dueling_net=dueling_net,cnn=cnn,use_value_net=use_value_net).to(device=self.device).eval()
        else:
            self.policy = CateoricalPolicy(
                self.obs_dim[-1], 1,continuous=True,action_dim=env.action_space.shape[0],cnn=cnn
                ).to(self.device)
            self.online_critic = TwinnedQNetwork(
            self.obs_dim[-1], 1,
            dueling_net=dueling_net,continuous=True,cnn=cnn,use_value_net=use_value_net).to(device=self.device)

            self.target_critic = TwinnedQNetwork(
            self.obs_dim[-1], 1,
            dueling_net=dueling_net,continuous=True,cnn=cnn,use_value_net=use_value_net).to(device=self.device).eval()


        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)
        if use_value_net:
            self.value_optim = Adam(self.online_critic.value_net.parameters(),lr=lr)

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
            
        if self.continuous:
            with torch.no_grad():
                action, _ = self.policy.continuous_sample(state)
            return action.cpu().numpy()
        else:
            with torch.no_grad():
                action, _, _ = self.policy.sample(state)
            return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.FloatTensor(
            state[None, ...]).to(self.device)
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    def update_target(self):
        #self.target_critic.value_net.load_state_dict(self.online_critic.value_net.state_dict())
        #polyak update:
        for update_param,target_param in zip(self.online_critic.value_net.parameters(),self.target_critic.value_net.parameters()):
            target_param.data.copy_((1-self.tau)*target_param + self.tau*update_param)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        if self.continuous:
            curr_q1, curr_q2,value = self.online_critic(states,actions)
        else:
            curr_q1, curr_q2,value = self.online_critic(states)
            curr_q1 = curr_q1.gather(1, actions.long())
            curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2,value

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        if self.use_value_net:
            if self.continuous:
                with torch.no_grad():
                    next_actions, log_action_probs = self.policy.continuous_sample(next_states)
                    next_q1, next_q2,value = self.target_critic(next_states,next_actions)
                    next_q = value
            else:  
                with torch.no_grad():
                    _, action_probs, log_action_probs = self.policy.sample(next_states)
                    next_q1, next_q2,value = self.target_critic(next_states)
                    next_q = value
        else:

            if self.continuous:
                with torch.no_grad():
                    next_actions, log_action_probs = self.policy.continuous_sample(next_states)
                    next_q1, next_q2,_ = self.target_critic(next_states,next_actions)
                    next_q = (
                        torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                        ).sum(dim=1, keepdim=True)
            else:  
                with torch.no_grad():
                    _, action_probs, log_action_probs = self.policy.sample(next_states)
                    next_q1, next_q2,_ = self.target_critic(next_states)
                    next_q = (action_probs * (
                        torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                        )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2,_ = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)


        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def compute_td_error(self,batch):
           
        if self.use_value_net:
            states, actions, rewards, next_states, dones = batch 
            with torch.no_grad():
                current_q1,_ ,_ = self.online_critic(states,actions)
                _,_,vf_next_target = self.target_critic(next_states,actions)
                target_q = rewards + (1.0 - dones) * self.gamma_n * vf_next_target

                td_error = target_q - current_q1
        else:
            curr_q1, curr_q2,_ = self.calc_current_q(*batch)
            target_q = self.calc_target_q(*batch)

            # TD errors for updating priority weights
            td_error = curr_q1.detach() - target_q
        
        return torch.abs(td_error).detach() + 1e-6

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        value_loss = None
        td_error = None
        if self.continuous:
            next_actions, log_action_probs = self.policy.continuous_sample(states)
            _,_,values = self.online_critic(states,actions)
            with torch.no_grad():
                q1 , q2,_ = self.online_critic(states,next_actions)
                q = torch.min(q1, q2)
                    
            if self.use_value_net:
                with torch.no_grad():
                    target_v = q - self.alpha * log_action_probs
                td_error = target_v - values

                value_loss = torch.mean(td_error.pow(2)* weights)
            policy_loss = (weights * (self.alpha * log_action_probs - q)).mean()

            return policy_loss , log_action_probs.detach(),value_loss,torch.abs(td_error).detach()+ 1e-6

        else:
            # (Log of) probabilities to calculate expectations of Q and entropies.
            _, action_probs, log_action_probs = self.policy.sample(states)

            with torch.no_grad():
                # Q for every actions to calculate expectations of Q.
                q1, q2,_ = self.online_critic(states)
                q = torch.min(q1, q2)

            # Expectations of entropies.
            entropies = -torch.sum(
                action_probs * log_action_probs, dim=1, keepdim=True)

            # Expectations of Q.
            q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

            # Policy objective is maximization of (Q + alpha * entropy) with
            # priority weights.
            policy_loss = (weights * (- q - self.alpha * entropies)).mean()

            return policy_loss, entropies.detach(),value_loss,td_error

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        if self.continuous:
            entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy + entropies)
            * weights)
        else:
            entropy_loss = -torch.mean(
                self.log_alpha * (self.target_entropy - entropies)
                * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
