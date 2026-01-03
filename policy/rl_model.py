
"""
Algorithm: ES_RL
Paper: Cost-aware dynamic multi-workflow scheduling in cloud data center using evolutionary reinforcement learning. ICSOC 2022.
Authors: Victoria Huang, Chen Wang, Hui Ma, Gang Chen, and Kameron Christopher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from policy.base_model import BasePolicy

running_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RLPolicy(BasePolicy):
    def __init__(self, config, policy_id=-1):
        super(RLPolicy, self).__init__()
        self.policy_id = policy_id
        self.state_num = config['state_num']
        self.action_num = config['action_num']
        self.discrete_action = config['discrete_action']
        if "add_gru" in config:
            self.add_gru = config['add_gru']
        else:
            self.add_gru = True

        self.fc1 = nn.Linear(self.state_num, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_num)
        self.apply(self.xavier_init)
        self.to(running_device)

    def forward(self, ob, removed=None):
        with torch.no_grad():
            x = torch.from_numpy(ob).float().to(running_device)
            # x = torch.from_numpy(ob).float()
            x = x.unsqueeze(0)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)

            if removed is not None:
                x[:,removed,:] = float("-inf")

            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)
                x = torch.argmax(x)
            else:
                x = torch.relu(x.squeeze())

            x = x.detach().cpu().numpy()
            return x.item(0)

    def get_logits(self, ob):
        x = torch.from_numpy(ob).float()
        x = x.unsqueeze(0)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x.squeeze(), dim=0)
        return x

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)  # if relu used, bias is set to 0.01

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.data = torch.from_numpy(out)

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def reset(self):
        pass

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.cpu().numpy())
            # param_lst.append(param.data.numpy())
        return param_lst

    def set_param_list(self, param_lst: list):
        lst_idx = 0
        for param in self.parameters():
            param.data = torch.tensor(param_lst[lst_idx]).float()
            lst_idx += 1

def create_sparse_matrix(ranges, length):
    rows = sum(ranges)
    indices = []
    values = []
    start_index = 0
    for range_length in ranges:
        for i in range(range_length):
            for j in range(start_index, start_index + range_length):
                indices.append([start_index + i, j])
                values.append(1)
        start_index += range_length
    # indices = torch.LongTensor(indices).t()
    # values = torch.FloatTensor(values)
    # return torch.sparse.FloatTensor(indices, values, torch.Size([rows, length]))
    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.tensor(values, dtype=torch.float)
    # return torch.sparse.FloatTensor(indices, values, torch.Size([rows, length]))
    return torch.sparse_coo_tensor(indices, values, torch.Size([rows, length]))


if __name__ == '__main__':
    random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(44)
    config = {
        'state_num': 8,
        'action_num': 1,
        'discrete_action': True,
    }
    rl = RLPolicy(config)
    torch.save(rl.state_dict(), 'policy.pt')
