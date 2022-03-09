from numpy.random import normal
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical,Normal,MultivariateNormal


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Attn_Net(nn.Module):
    def __init__(self,num_heads=2,embed_dim=128):
        super(Attn_Net, self).__init__()
        self.time_attn = nn.MultiheadAttention(embed_dim, num_heads)
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        # self.net = nn.Sequential(
        #     nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     Flatten(),
        # ).apply(initialize_weights_he)

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)

class LSTMNetwork(BaseNetwork):
    def __init__(self,num_inputs,hist_connect=False):
        super(LSTMNetwork,self).__init__()
        #input:[batch,seq,hidden]
        self.net = nn.Sequential(
            # nn.Linear(num_inputs,256),
            # nn.ReLU(),
            # nn.Linear(256,256),
            # nn.ReLU(),
            # nn.Linear(256,256),
            # nn.ReLU(),
            nn.LSTM(num_inputs,256,batch_first=True),
            # nn.LSTM(256,256,batch_first=True),
            # nn.LSTM(256,256,batch_first=True)
        )
        self.net_2 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.hist_connect=hist_connect
    
    def forward(self,states):
        output,(_,_) = self.net(states)
        # output,(_,_) = self.net[1](output)
        # output,(_,_) = self.net[2](output)
        #return the last output
        out = self.net_2(output[:,-1,:])
        # return output[:,-1,:]
        return out

class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False,cnn=False,continuous=False,action_dim=2,lstm=False,
                 lstm_fut=False,use_attn=False,umap=False,hist_fut=False,agg=False):
        super().__init__()

        if not shared:
            if cnn:
                self.conv = DQNBase(num_channels)
                #out_dim = 6*6*64
                out_dim = 256
            elif lstm:
                self.conv = LSTMNetwork(num_channels)
                out_dim = 256
            
            else:
                self.conv = nn.Sequential(
                    nn.Linear(num_channels, 256),
                    nn.ReLU(),
                    # nn.Linear(256, 256),
                    # nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU()
                )
                out_dim = 256

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(out_dim+64, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(out_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(out_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1))
        
        if continuous:
            self.action_trans = nn.Sequential(
                 nn.Linear(action_dim,64),
                 nn.ReLU())

        self.shared = shared
        self.dueling_net = dueling_net
        self.continuous = continuous

    def forward(self, states,actions=None):
        if not self.shared:
            states = self.conv(states)

        if self.continuous:
            # print(actions.shape)
            actions_1 = self.action_trans(actions)
            # print(actions_1.shape)
            # print(states.shape)
            states = torch.cat([states,actions_1],dim=1)
            # print(states.shape)
            #states + self.action_trans(actions)
        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)

class ValueNetwork(BaseNetwork):
    def __init__(self,num_channels,state_dim=None,shared=False,cnn=False,lstm=False,
    lstm_fut=False,use_attn=False,umap=False,hist_fut=False,agg=False):
        super().__init__()
        self.shared = shared
        if not shared:
            if cnn:
                self.conv = DQNBase(num_channels)
                #out_dim = 6*6*64
                out_dim = 256
            
            elif lstm:
                self.conv = LSTMNetwork(num_channels)
                out_dim = 256
            else:
                self.conv = nn.Sequential(
                    nn.Linear(num_channels, 256),
                    nn.ReLU(),
                    # nn.Linear(256, 256),
                    # nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU()
                )
                out_dim = 256
            
            self.value_head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        
        else:
            self.value_head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1))


    def forward(self,states):
        if not self.shared:
            states = self.conv(states)
        return self.value_head(states)

class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False,cnn=False,continuous=False,use_value_net=False,
                 lstm_fut=False,use_attn=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net,cnn,continuous,lstm_fut=lstm_fut,use_attn=use_attn)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net,cnn,continuous,lstm_fut=lstm_fut,use_attn=use_attn)
        self.Share_conv = DQNBase(num_channels)
        if use_value_net:
            self.value_net = ValueNetwork(num_channels,shared=shared,cnn=cnn)
        self.use_value_net = use_value_net
        self.continuous = continuous

    def forward(self, states,action=None):
        if self.continuous:
            q1 = self.Q1(states,action)
            q2 = self.Q2(states,action)
        else: 
            q1 = self.Q1(states)
            q2 = self.Q2(states)
        
        if self.use_value_net:
            value = self.value_net(states)
        else:
            value=None
        return q1, q2,value


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,cnn=False,continuous=False,action_dim=2,
        log_std_bounds=(-20,2),lstm=False,lstm_fut=False,use_attn=False,umap=False,hist_fut=False,agg=False):
        super().__init__()

        if cnn:
            self.conv = DQNBase(num_channels)
            #out_dim = 6*6*64
            out_dim = 256
        
        elif lstm:
            self.conv = LSTMNetwork(num_channels)
            out_dim = 256
        else:
            self.conv = nn.Sequential(
                nn.Linear(num_channels, 256),
                nn.ReLU(),
                # nn.Linear(256, 256),
                # nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            )
            out_dim = 256

        self.head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions))
        
        self.action_trans = nn.Sequential(
            nn.Linear(action_dim,out_dim)
        )

        self.head_continuous = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU())
        
        self.mu_trans =  nn.Linear(32, action_dim)

        self.std_trans =  nn.Linear(32, action_dim)

        self.shared = shared
        self.continuous = continuous
        self.log_std_bounds = log_std_bounds
        self.action_dim = action_dim

    # def act(self, states):
    #     if not self.shared:
    #         states = self.conv(states)

    #     action_logits = self.head(states)
    #     greedy_actions = torch.argmax(
    #         action_logits, dim=1, keepdim=True)
    #     return greedy_actions
    
    def act(self,states):
        if not self.shared:
            states = self.conv(states)
        states = self.head_continuous(states)
        mu = self.mu_trans(states)
        actions = torch.tanh(mu)
        return actions

    def sample(self, states):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
    
    def continuous_sample(self,states):
        if not self.shared:
            states = self.conv(states)
        states = self.head_continuous(states)
        mu,log_std = self.mu_trans(states),self.std_trans(states)

        #log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
        #                                                              1)
        log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)

        std = log_std.exp()

        var_mat = torch.diag_embed(std)

        #print(var_mat.shape)

        normal = MultivariateNormal(mu,var_mat)
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.rsample()
        actions = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        #print(log_prob.shape)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - actions.pow(2)) + 1e-6).sum(1)
        #log_prob = log_prob.sum(1, keepdims=True)
        #print(log_prob.shape)

        return actions, log_prob


