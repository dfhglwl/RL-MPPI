import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import numpy as np
from copy import deepcopy

class GaussianPolicy(nn.Module):
    """DSAC的高斯策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        # 强制重新创建张量确保设备一致性
        device = next(self.parameters()).device
        
        # 总是创建一个新的张量并明确放在正确的设备上
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        elif isinstance(state, list):
            state = torch.FloatTensor(state).to(device)
        elif isinstance(state, torch.Tensor):
            # 始终复制到新张量以避免可能的引用问题
            state = state.clone().detach().to(device)
        else:
            # 尝试转换其他类型
            try:
                state = torch.FloatTensor(state).to(device)
            except:
                raise TypeError(f"无法处理类型 {type(state)}，请提供numpy数组或PyTorch张量")
        
        # 确保维度正确
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # 检查是否有NaN值
        if torch.isnan(state).any():
            print("警告：状态中包含NaN值")
            state = torch.nan_to_num(state)
            
        x = F.gelu(self.fc1(state))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mean, std
    
    def sample(self, state):
        # 由于forward方法已经处理了设备转换，这里直接调用即可
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        
        # 重参数化采样
        x = mean + std * torch.randn_like(std)
        
        # 计算对数概率
        log_prob = normal.log_prob(x).sum(dim=-1, keepdim=True)
        
        return x, log_prob, mean, std
    
    def get_action(self, state, deterministic=False):
        """获取动作，确保设备一致性"""
        with torch.no_grad():
            # 由于forward方法已经处理了设备转换，这里直接调用即可
            mean, std = self.forward(state)
            
            if deterministic:
                action = mean
            else:
                normal = Normal(mean, std)
                action = normal.sample()
                
            return action.cpu().numpy().flatten()

class DistributionalCritic(nn.Module):
    """DSAC的分布式评论家网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_atoms=51):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出分布式表示的logits
        self.logits = nn.Linear(hidden_dim, num_atoms)
        
    def forward(self, state, action):
        # 强制重新创建张量确保设备一致性
        device = next(self.parameters()).device
        
        # 总是创建新的张量并明确放在正确的设备上
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        elif isinstance(state, list):
            state = torch.FloatTensor(state).to(device)
        elif isinstance(state, torch.Tensor):
            state = state.clone().detach().to(device)
        else:
            try:
                state = torch.FloatTensor(state).to(device)
            except:
                raise TypeError(f"无法处理类型 {type(state)}，请提供numpy数组或PyTorch张量")
        
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(device)
        elif isinstance(action, list):
            action = torch.FloatTensor(action).to(device)
        elif isinstance(action, torch.Tensor):
            action = action.clone().detach().to(device)
        else:
            try:
                action = torch.FloatTensor(action).to(device)
            except:
                raise TypeError(f"无法处理类型 {type(action)}，请提供numpy数组或PyTorch张量")
        
        # 确保维度正确
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # 检查是否有NaN值
        if torch.isnan(state).any() or torch.isnan(action).any():
            print("警告：输入中包含NaN值")
            state = torch.nan_to_num(state)
            action = torch.nan_to_num(action)
            
        x = torch.cat([state, action], dim=-1)
        
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        
        logits = self.logits(x)
        
        # 返回softmax概率
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def get_qvalue(self, probs):
        """计算期望Q值"""
        device = next(self.parameters()).device
        support = self.get_support(device)
        return (probs * support.unsqueeze(0)).sum(-1, keepdim=True)
    
    def get_support(self, device):
        """获取值分布支持集"""
        return torch.linspace(-10, 10, self.num_atoms).to(device)

class DSAC:
    """分布式软演员评论家算法"""
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        target_entropy=None,
        hidden_dim=256,
        lr_actor=2e-5,
        lr_critic=1e-5,
        lr_alpha=1e-5,
        auto_alpha=True,
        TD_bound=20.0,
        num_atoms=51
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.TD_bound = TD_bound
        self.auto_alpha = auto_alpha
        self.num_atoms = num_atoms
        
        # 创建策略网络
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        
        # 创建评论家网络
        self.critic = DistributionalCritic(state_dim, action_dim, hidden_dim, num_atoms).to(device)
        self.critic_target = deepcopy(self.critic).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        # 目标网络不需要梯度
        for param in self.critic_target.parameters():
            param.requires_grad = False
            
        # 自动调整温度参数
        if target_entropy is None:
            self.target_entropy = -4  # 根据论文修改为-4
        else:
            self.target_entropy = target_entropy
            
        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
    
    def select_action(self, state, evaluate=False):
        """选择动作，确保设备一致性"""
        try:
            return self.actor.get_action(state, deterministic=evaluate)
        except Exception as e:
            print(f"选择动作时出错: {e}")
            print(f"状态类型: {type(state)}")
            if isinstance(state, np.ndarray):
                print(f"状态形状: {state.shape}")
            elif isinstance(state, torch.Tensor):
                print(f"状态形状: {state.shape}, 设备: {state.device}")
            # 尝试恢复
            if isinstance(state, np.ndarray):
                # 确保状态不包含NaN或无穷值
                state = np.nan_to_num(state)
                # 尝试使用随机动作
                return np.random.uniform(low=0.0, high=1.0, size=(4,))
            raise
    
    def calc_target_q(self, reward, next_state, done):
        """计算目标Q值"""
        with torch.no_grad():
            # 1. 采样下一个动作和其熵
            next_action, next_log_prob, _, _ = self.actor.sample(next_state)
            
            # 2. 计算目标分布
            target_probs = self.critic_target(next_state, next_action)
            target_q = self.critic_target.get_qvalue(target_probs)
            
            # 3. 计算目标值（含熵项）
            return reward + (1 - done) * self.gamma * (target_q - self.alpha * next_log_prob)
    
    def update_parameters(self, memory, batch_size=256):
        """更新网络参数"""
        # 从经验回放中采样
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)
        
        # 确保数据在正确的设备上
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        # 计算当前状态的动作分布
        current_probs = self.critic(state_batch, action_batch)
        current_q = self.critic.get_qvalue(current_probs)
        
        # 计算目标值
        target_q = self.calc_target_q(reward_batch, next_state_batch, done_batch)
        
        # 计算TD误差并限制范围
        td_error = target_q - current_q
        clamped_td_error = torch.clamp(td_error, -self.TD_bound, self.TD_bound)
        target_q_bounded = current_q + clamped_td_error
        
        # 通过最小化KL散度更新分布式critic
        with torch.no_grad():
            next_action, _, _, _ = self.actor.sample(next_state_batch)
            target_probs = self.critic_target(next_state_batch, next_action)
        
        # 分布式损失（KL散度）
        critic_loss = F.kl_div(
            F.log_softmax(current_probs, dim=-1),
            target_probs.detach(),
            reduction='batchmean'
        )
        
        # 更新评论家
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 策略优化（论文式10）
        new_action, new_log_prob, _, _ = self.actor.sample(state_batch)
        new_probs = self.critic(state_batch, new_action)
        new_q = self.critic.get_qvalue(new_probs)
        
        actor_loss = (self.alpha * new_log_prob - new_q).mean()
        
        # 更新演员
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 自动调整alpha
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'q_mean': current_q.mean().item(),
            'td_error': td_error.mean().item()
        }
    
    def save(self, directory, iteration=None):
        """保存模型"""
        if iteration is not None:
            torch.save(self.actor.state_dict(), f"{directory}/actor_{iteration}.pth")
            torch.save(self.critic.state_dict(), f"{directory}/critic_{iteration}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{directory}/actor.pth")
            torch.save(self.critic.state_dict(), f"{directory}/critic.pth")
            
    def load(self, directory, iteration=None):
        """加载模型"""
        try:
            if iteration is not None:
                self.actor.load_state_dict(torch.load(f"{directory}/actor_{iteration}.pth", map_location=self.device))
                self.critic.load_state_dict(torch.load(f"{directory}/critic_{iteration}.pth", map_location=self.device))
            else:
                self.actor.load_state_dict(torch.load(f"{directory}/actor.pth", map_location=self.device))
                self.critic.load_state_dict(torch.load(f"{directory}/critic.pth", map_location=self.device))
                
            # 确保critic_target与critic在同一设备上
            self.critic_target = deepcopy(self.critic)
            
            # 打印模型参数的设备信息，帮助调试
            print(f"模型加载成功。设备信息: ")
            print(f"Actor 设备: {next(self.actor.parameters()).device}")
            print(f"Critic 设备: {next(self.critic.parameters()).device}")
            print(f"Target Critic 设备: {next(self.critic_target.parameters()).device}")
        except Exception as e:
            print(f"加载模型时出错: {e}")
