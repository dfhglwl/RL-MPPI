import torch
import numpy as np
from copy import deepcopy

class MPPIController:
    """基于RL策略初始化的MPPI控制器 - 严格按照论文实现"""
    def __init__(
        self,
        state_dim,
        action_dim,
        actor,
        critic,
        dynamics_model,
        device,
        K=6,               # 迭代次数
        N=15,              # 预测时域
        lambda_=1.0,       # 逆温度参数
        num_samples=125,   # 每次迭代使用的总样本数
        n_rl=125,          # RL引导样本数
        n_mppi=125,        # MPPI样本数
        sigma_min=0.5,     # 最小标准差
        action_low=None,   # 动作下界
        action_high=None,  # 动作上界
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = actor
        self.critic = critic
        self.dynamics = dynamics_model
        self.device = device
        
        # MPPI参数 - 使用论文中的参数
        self.K = K
        self.N = N
        self.lambda_ = lambda_
        self.num_samples = num_samples
        
        # 混合采样参数
        self.n_rl = n_rl
        self.n_mppi = n_mppi
        
        # 最小标准差 - 论文明确提到的参数
        self.sigma_min = sigma_min
        
        # 动作约束
        if action_low is None:
            self.action_low = np.array([0.0, -5.0, -5.0, -5.0])
        else:
            self.action_low = action_low
            
        if action_high is None:
            self.action_high = np.array([20.0, 5.0, 5.0, 5.0])
        else:
            self.action_high = action_high
    
    def _compute_costs(self, states, actions):
        """计算轨迹代价（式11和19）- 严格按照论文"""
        batch_size, horizon, _ = states.shape
        total_costs = np.zeros(batch_size)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        # 计算即时代价
        for t in range(horizon - 1):
            state_t = states_tensor[:, t]
            action_t = actions_tensor[:, t]
            
            # 计算位置代价
            p = state_t[:, :3]
            p_target = torch.zeros_like(p)
            cost_p = torch.sum((p - p_target) ** 2, dim=1)
            
            # 计算速度代价
            v = state_t[:, 3:6]
            cost_v = torch.sum(v ** 2, dim=1)
            
            # 新增：接近目标时的额外速度衰减代价
            speed = torch.norm(v, dim=1)
            distance_to_target = torch.norm(p, dim=1)
            close_to_target = (distance_to_target < 5.0).float()
            speed_decay_cost = 0.2 * speed * close_to_target
            
            # 计算角速度代价
            w = state_t[:, 10:13] if state_t.shape[1] > 10 else state_t[:, 6:9]
            cost_w = torch.sum(w ** 2, dim=1)
            
            # 计算姿态代价（简化，仅使用四元数的欧几里得距离）
            q = state_t[:, 6:10] if state_t.shape[1] > 10 else state_t[:, 9:13]
            q_target = torch.zeros_like(q)
            q_target[:, 0] = 1.0  # 单位四元数
            cost_q = torch.sum((q - q_target) ** 2, dim=1)
            
            # 加权组合代价 - 使用论文中的权重
            step_cost = 0.02 * cost_p + 0.01 * cost_v + 0.001 * cost_w + 0.001 * cost_q + speed_decay_cost
            
            # 新增：边界惩罚 - 防止规划轨迹超出边界
            boundary = 10.0
            boundary_cost = torch.zeros_like(cost_p)
            for i in range(3):  # x, y, z 坐标
                boundary_mask = torch.abs(p[:, i]) > boundary * 0.8
                boundary_cost[boundary_mask] += (torch.abs(p[boundary_mask, i]) - boundary * 0.8) ** 2 * 0.1
            
            # 添加边界成本
            step_cost += boundary_cost
            
            # 累加代价
            total_costs += step_cost.cpu().numpy()
            
            # 新增：轨迹曲率考虑 - 如果t > 0，计算速度方向变化
            if t > 0:
                v_prev = states_tensor[:, t-1, 3:6]
                v_curr = v
                # 只在速度足够大时计算
                v_norm_prev = torch.norm(v_prev, dim=1)
                v_norm_curr = torch.norm(v_curr, dim=1)
                valid_speeds = (v_norm_prev > 0.1) & (v_norm_curr > 0.1)
                
                # 初始化曲率成本为零
                curvature_cost = torch.zeros_like(cost_p)
                
                if valid_speeds.any():
                    # 计算速度方向变化（只对有效速度计算）
                    cos_angle = torch.sum(v_prev[valid_speeds] * v_curr[valid_speeds], dim=1) / (
                        v_norm_prev[valid_speeds] * v_norm_curr[valid_speeds]
                    )
                    # 限制在[-1, 1]范围内
                    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                    
                    # 计算曲率成本 - 惩罚剧烈的方向变化
                    direction_change = 1.0 - cos_angle
                    # 当距离目标近时，鼓励适度转弯
                    near_target = distance_to_target[valid_speeds] < 8.0
                    
                    # 给valid_speeds索引对应的curvature_cost赋值
                    temp_cost = torch.zeros(valid_speeds.sum())
                    temp_cost[near_target] = -0.5 * direction_change[near_target]  # 近距离时鼓励适度转弯
                    temp_cost[~near_target] = 2.0 * direction_change[~near_target]  # 远距离时惩罚剧烈转向
                    
                    # 将计算结果分配回原始tensor
                    curvature_cost[valid_speeds] = temp_cost
                
                # 添加到步骤成本
                total_costs += curvature_cost.cpu().numpy()
        
        # 计算终端代价（使用Q值）- 论文公式(19)
        final_states = states_tensor[:, -1]
        
        with torch.no_grad():
            # 使用策略网络获取终端动作
            mean, _ = self.actor(final_states)
            
            # 使用评论家网络计算Q值 - 关键修改：使用Q值作为终端成本
            q_mean, _ = self.critic(final_states, mean)
            
            # 按照公式19，终端成本是Q值的负值
            terminal_costs = -q_mean.squeeze(-1).cpu().numpy()
            
        # 总代价 = 累积即时代价 + 终端代价
        total_costs += terminal_costs
        
        return total_costs
    
    def _compute_weights(self, costs):
        """计算样本权重"""
        # 数值稳定性处理
        min_cost = np.min(costs)
        costs_normalized = costs - min_cost
        
        # 计算指数权重
        exponentiated = np.exp(-1.0 / self.lambda_ * costs_normalized)
        
        # 归一化
        weights = exponentiated / (np.sum(exponentiated) + 1e-10)
        
        return weights
    
    def _update_control_dist(self, action_sequences, costs):
        """更新控制分布（式18）- 动态更新方差（论文式18b）"""
        weights = self._compute_weights(costs)
        
        # 计算加权平均作为新的均值 - 式18a
        u_new = np.zeros_like(action_sequences[0])
        for i in range(len(action_sequences)):
            u_new += weights[i] * action_sequences[i]
            
        # 计算加权方差 - 式18b
        sigma_new = np.zeros_like(action_sequences[0])
        for i in range(len(action_sequences)):
            deviation = action_sequences[i] - u_new
            sigma_new += weights[i] * (deviation ** 2)
        
        # 取平方根并应用最小标准差约束
        sigma_new = np.sqrt(sigma_new)
        sigma_new = np.maximum(sigma_new, self.sigma_min)
        
        return u_new, sigma_new
    
    def _rl_guided_sampling(self, state):
        """从RL策略生成引导样本 - 混合采样策略的一部分"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 生成动作序列
        action_sequences = []
        state_sequences = []
        
        for _ in range(self.n_rl):
            # 初始化序列
            actions = []
            states = [state]
            current_state = state
            
            for t in range(self.N):
                # 从策略网络采样动作
                current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _, _ = self.actor.sample(current_state_tensor)
                    action = action.squeeze(0).cpu().numpy()
                    
                # 应用动作约束
                action = np.clip(action, self.action_low, self.action_high)
                actions.append(action)
                
                # 使用动态模型预测下一状态
                with torch.no_grad():
                    next_state = self.dynamics.predict_next_state(current_state, action)
                
                states.append(next_state)
                current_state = next_state
            
            action_sequences.append(np.array(actions))
            state_sequences.append(np.array(states))
        
        return np.array(action_sequences), np.array(state_sequences)
    
    def _hybrid_sampling(self, state, mean_sequence, std_sequence):
        """混合采样策略 (HSS) - 结合RL样本和MPPI样本"""
        # 从RL策略生成样本
        rl_actions, rl_states = self._rl_guided_sampling(state)
        
        # 生成随机噪声用于MPPI采样
        noise = np.random.normal(0, 1, size=(self.n_mppi, self.N, self.action_dim))
        
        # 应用均值和方差生成MPPI样本
        mppi_action_sequences = np.expand_dims(mean_sequence, axis=0) + noise * np.expand_dims(std_sequence, axis=0)
        
        # 应用动作约束
        mppi_action_sequences = np.clip(mppi_action_sequences, self.action_low, self.action_high)
        
        # 使用动态模型批量预测MPPI状态序列
        mppi_state_sequences = []
        
        for i in range(self.n_mppi):
            current_state = state.copy()
            states = [current_state]
            
            # 对每个时间步预测状态
            for t in range(self.N):
                action = mppi_action_sequences[i, t]
                next_state = self.dynamics.predict_next_state(current_state, action)
                states.append(next_state)
                current_state = next_state
            
            mppi_state_sequences.append(np.array(states))
        
        # 合并RL样本和MPPI样本
        combined_actions = np.vstack([rl_actions, mppi_action_sequences])
        combined_states = np.vstack([rl_states, np.array(mppi_state_sequences)])
        
        return combined_actions, combined_states
    
    def _initialize_sequence(self, state):
        """使用RL策略初始化控制序列（算法步骤12）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        current_state = state
        
        # 初始化序列
        mean_sequence = np.zeros((self.N, self.action_dim))
        std_sequence = np.zeros((self.N, self.action_dim))
        
        # 生成状态序列
        states = [current_state]
        
        for t in range(self.N):
            # 从策略网络获取均值和标准差
            current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mean, std = self.actor(current_state_tensor)
                action = mean.squeeze(0).cpu().numpy()
            
            # 应用动作约束
            action = np.clip(action, self.action_low, self.action_high)
            
            # 保存均值和标准差
            mean_sequence[t] = action
            std_sequence[t] = std.squeeze(0).cpu().numpy()
            
            # 预测下一状态
            with torch.no_grad():
                next_state = self.dynamics.predict_next_state(current_state, action)
            
            states.append(next_state)
            current_state = next_state
        
        # 确保标准差满足最小值约束
        std_sequence = np.maximum(std_sequence, self.sigma_min)
        
        return mean_sequence, std_sequence
    
    def control(self, state):
        """执行完整MPPI优化（算法步骤11-22）"""
        # 初始化控制分布 - 使用RL策略
        mean_sequence, std_sequence = self._initialize_sequence(state)
        
        # 迭代优化
        for k in range(self.K):
            # 混合采样策略(HSS) - 组合RL样本和MPPI样本
            combined_actions, combined_states = self._hybrid_sampling(state, mean_sequence, std_sequence)
            
            # 计算代价
            costs = self._compute_costs(combined_states, combined_actions)
            
            # 按代价排序，获取前num_samples个
            sorted_indices = np.argsort(costs)[:self.num_samples]
            elite_actions = combined_actions[sorted_indices]
            elite_costs = costs[sorted_indices]
            
            # 更新控制分布 - 同时更新均值和方差
            mean_sequence, std_sequence = self._update_control_dist(elite_actions, elite_costs)
            
        # 返回第一个动作
        action = mean_sequence[0]
        
        # 应用约束
        action = np.clip(action, self.action_low, self.action_high)
        
        return action