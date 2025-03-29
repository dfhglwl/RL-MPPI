import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class DynamicsModel(nn.Module):
    """UAV动态模型的神经网络表示"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, state_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state, action):
        # 确保输入在正确的设备上
        device = self.fc1.weight.device
        if state.device != device:
            state = state.to(device)
        if action.device != device:
            action = action.to(device)
            
        x = torch.cat([state, action], dim=-1)
        
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = F.gelu(self.fc4(x))
        
        # 预测状态差异而非绝对状态
        delta = self.fc5(x)
        
        # 返回下一个状态预测
        next_state = state + delta
        
        return next_state
    
    def predict_next_state(self, state, action):
        """预测给定状态和动作后的下一状态"""
        with torch.no_grad():
            # 获取设备信息
            device = self.fc1.weight.device
            
            # 确保状态和动作是张量并且在正确的设备上
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(device)
            else:
                state_tensor = state.to(device)
                
            if isinstance(action, np.ndarray):
                action_tensor = torch.FloatTensor(action).to(device)
            else:
                action_tensor = action.to(device)
            
            # 确保维度正确
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
                
            next_state = self.forward(state_tensor, action_tensor)
            
            return next_state.squeeze(0).cpu().numpy()
    
    def predict_trajectory(self, initial_state, actions):
        """预测给定初始状态和动作序列下的轨迹"""
        states = [initial_state]
        current_state = initial_state
        
        with torch.no_grad():
            for action in actions:
                current_state = self.predict_next_state(current_state, action)
                states.append(current_state)
                
        return np.array(states)
    
    def batch_predict_trajectories(self, initial_state, action_sequences):
        """批量预测多条轨迹"""
        batch_size = len(action_sequences)
        horizon = len(action_sequences[0])
        state_dim = initial_state.shape[0]
        
        # 初始化状态张量
        states = torch.zeros((batch_size, horizon + 1, state_dim))
        states[:, 0] = torch.FloatTensor(initial_state)
        
        # 将动作序列转换为张量
        actions = torch.FloatTensor(action_sequences)
        
        # 按时间步预测
        for t in range(horizon):
            current_states = states[:, t]
            current_actions = actions[:, t]
            
            # 预测下一个状态
            next_states = self.forward(current_states, current_actions)
            states[:, t+1] = next_states
            
        return states.cpu().numpy()
    
    def predict_next_state(self, current_state, action):
        """预测给定当前状态和动作下的下一个状态"""
        # 确保输入是numpy数组
        if isinstance(current_state, list):
            current_state = np.array(current_state)
        if isinstance(action, list):
            action = np.array(action)
            
        # 转换为张量并处理设备
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(next(self.parameters()).device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(next(self.parameters()).device)
            
            # 预测下一个状态
            next_state_tensor = self.forward(state_tensor, action_tensor)
            
            # 转换回numpy数组
            next_state = next_state_tensor.squeeze(0).cpu().numpy()
            
        return next_state

class DynamicsTrainer:
    """动态模型训练器"""
    def __init__(self, state_dim, action_dim, device, lr=1e-4, hidden_dim=256):
        self.device = device
        self.model = DynamicsModel(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
    def train_step(self, states, actions, next_states, batch_size=256):
        """执行一步训练"""
        # 确保数据为张量形式
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.FloatTensor(actions).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        
        # 模型预测
        predicted_next_states = self.model(state_tensor, action_tensor)
        
        # 计算损失 - MSE损失
        loss = F.mse_loss(predicted_next_states, next_state_tensor)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"dynamics_loss": loss.item()}
    
    def train_epochs(self, memory, epochs=10, batch_size=256):
        """训练多个轮次"""
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # 从经验回放中采样
            state_batch, action_batch, _, next_state_batch, _ = memory.sample(batch_size)
            
            # 训练步骤
            step_info = self.train_step(state_batch, action_batch, next_state_batch, batch_size)
            epoch_losses.append(step_info["dynamics_loss"])
            
            losses.append(np.mean(epoch_losses))
            
        return {"avg_dynamics_loss": np.mean(losses)}
    
    def save(self, directory, iteration=None):
        """保存模型"""
        if iteration is not None:
            torch.save(self.model.state_dict(), f"{directory}/dynamics_{iteration}.pth")
        else:
            torch.save(self.model.state_dict(), f"{directory}/dynamics.pth")
            
    def load(self, directory, iteration=None):
        """加载模型"""
        if iteration is not None:
            self.model.load_state_dict(torch.load(f"{directory}/dynamics_{iteration}.pth"))
        else:
            self.model.load_state_dict(torch.load(f"{directory}/dynamics.pth"))