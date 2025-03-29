import numpy as np

class Quadrotor:
    def __init__(self, g=9.82):
        # 动力学参数优化（更接近真实四旋翼）
        self.g = g
        self.Jx = 0.015  # X轴惯量 (kg·m²)
        self.Jy = 0.015  # Y轴惯量
        self.Jz = 0.025  # Z轴惯量
        self.m = 0.5  # 质量 (kg)
        self.l = 0.2  # 轴距 (m)
        self.c = 0.01  # 反扭矩系数
        self.J_B = np.diag([self.Jx, self.Jy, self.Jz])
        self.g_I = np.array([0, 0, -self.g])
        self.dt = 0.05

        self.goal_pos = np.array([0, 0, 0])  # 目标位置
        self.max_steps = 1000  # 对应论文参数
        self.step_count = 0  # 步数计数器

        self.target_radius = 1.5  # 对应论文ϱ
        self.wp = 0.02  # 对应ϖ_p
        self.wv = 0.01  # 对应ϖ_v
        self.ww = 0.001  # 对应ϖ_w
        self.wq = 0.001  # 对应ϖ_q
        self.max_steps = 1000  # 论文Maximum Episode Length

        # 环境边界优化
        self.boundary = 5.0  # 初始化范围±5m

        # 添加状态边界检查
        self.max_position = 100.0  # 位置最大值
        self.max_velocity = 50.0   # 速度最大值
        self.max_angular_velocity = 20.0  # 角速度最大值

    def reset(self):
        """初始化状态（与论文实验设置一致）"""
        # 位置初始化：在10m立方体内（±5m范围）
        r_state = np.random.uniform(-5, 5, size=(3,))

        # 速度初始化：±0.5 m/s（更平稳的初始条件）
        v_state = np.random.uniform(-0.5, 0.5, size=(3,))

        # 姿态初始化：轻微随机扰动（论文式23）
        q_angle = np.random.uniform(-0.1, 0.1)  # 已经是浮点数
        q_state = self.toQuaternion(q_angle, np.array([0, 0, 1], dtype=np.float64))  # 明确指定轴的类型

        # 角速度初始化：±0.5 rad/s
        w_state = np.random.uniform(-0.5, 0.5, size=(3,))

        self.state = np.hstack([r_state, v_state, q_state, w_state])
        self.step_count = 0
        return self.state.copy()

    def _compute_reward(self, state):
        """实现论文式22的奖励函数"""
        current_pos = state[0:3]
        distance = np.linalg.norm(current_pos - self.goal_pos)
        done = False
        info = {
            'distance': distance,
            'position': current_pos.copy(),
            'velocity': np.linalg.norm(state[3:6]),
            'attitude_error': 0.0
        }

        # 目标到达奖励（论文式22条件1）
        if distance <= self.target_radius:
            info['attitude_error'] = 0.0
            return 200.0, True, info

        # 位置惩罚（论文式22条件2）
        pos_penalty = self.wp * (distance ** 2)

        # 速度惩罚（论文式22条件3）
        vel_penalty = self.wv * np.sum(state[3:6] ** 2)

        # 角速度惩罚（论文式22条件4）
        ang_vel_penalty = self.ww * np.sum(state[10:13] ** 2)

        # 姿态误差计算（论文式23）
        R_current = self.dir_cosine(state[6:10])
        R_target = np.eye(3)  # 目标姿态为identity四元数
        attitude_error = 0.5 * np.trace(np.eye(3) - R_target.T @ R_current)
        attitude_penalty = self.wq * attitude_error

        info['attitude_error'] = attitude_error

        # 总奖励（论文式22）
        total_reward = - (pos_penalty + vel_penalty + ang_vel_penalty + attitude_penalty)

        # 超时判断（论文实验设置）
        done = self.step_count >= self.max_steps
        self.step_count += 1

        return total_reward, done, info

    def step(self, state, action):
        """动力学模型与论文式20一致"""
        # 状态解析
        rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = state
        self.r_I = np.array([rx, ry, rz])
        self.v_I = np.array([vx, vy, vz])
        self.q = np.array([q0, q1, q2, q3])
        self.w_B = np.array([wx, wy, wz])

        # 动作解析（符合论文表I）
        thrust = action[0]  # 总推力 (N)
        torque = action[1:4]  # 三轴扭矩 (Nm)
        
        # 限制动作范围
        thrust = np.clip(thrust, 0.0, 4.0)  # 总推力限制
        torque = np.clip(torque, -1.0, 1.0)  # 扭矩限制
        
        # 从总推力和扭矩反解四个电机的推力
        # 根据四旋翼动力学关系:
        # thrust = f1 + f2 + f3 + f4
        # Mx = l * (f4 - f2)
        # My = l * (f3 - f1)
        # Mz = c * (f1 - f2 + f3 - f4)
        
        Mx, My, Mz = torque
        
        # 解线性方程组得到各电机推力
        f1 = thrust/4 - My/(4*self.l) + Mz/(4*self.c)
        f2 = thrust/4 - Mx/(4*self.l) - Mz/(4*self.c)
        f3 = thrust/4 + My/(4*self.l) + Mz/(4*self.c)
        f4 = thrust/4 + Mx/(4*self.l) - Mz/(4*self.c)
        
        # 确保电机推力非负
        f1, f2, f3, f4 = np.clip([f1, f2, f3, f4], 0.0, 1.0)
        
        # 更新总推力和扭矩（考虑裁剪后的实际值）
        thrust = f1 + f2 + f3 + f4
        self.thrust_B = np.array([0.0, 0.0, thrust])
        
        # 力矩计算
        Mx = self.l * (f4 - f2)
        My = self.l * (f3 - f1)
        Mz = self.c * (f1 - f2 + f3 - f4)
        self.M_B = np.array([Mx, My, Mz])

        # 方向余弦矩阵（论文式23）
        C_B_I = self.dir_cosine(self.q)  # Body to Inertial
        C_I_B = C_B_I.T  # Inertial to Body

        # 动力学方程（论文式20）
        dr_I = self.v_I
        dv_I = (np.dot(C_I_B, self.thrust_B) / self.m) + self.g_I
        dq = 0.5 * np.dot(self.omega(self.w_B), self.q)
        dw_B = np.linalg.inv(self.J_B) @ (self.M_B - np.cross(self.w_B, self.J_B @ self.w_B))

        # 欧拉积分
        new_state = state + np.hstack([dr_I, dv_I, dq, dw_B]) * self.dt
        
        # 添加状态限制，防止数值不稳定
        # 限制位置
        if np.any(abs(new_state[0:3]) > self.boundary):
           done = True  # 飞出边界视为失败
           reward = -200  # 大惩罚
        # 限制速度
        new_state[3:6] = np.clip(new_state[3:6], -self.max_velocity, self.max_velocity)
        # 归一化四元数，确保单位长度
        quat_norm = np.linalg.norm(new_state[6:10])
        if quat_norm > 1e-6:  # 避免除以零
            new_state[6:10] = new_state[6:10] / quat_norm
        else:
            new_state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])  # 默认单位四元数
        # 限制角速度
        new_state[10:13] = np.clip(new_state[10:13], -self.max_angular_velocity, self.max_angular_velocity)

        # 奖励计算（论文式22）
        reward, done, info = self._compute_reward(new_state)
        
        # 检查状态是否包含NaN或Inf
        if np.any(np.isnan(new_state)) or np.any(np.isinf(new_state)):
            print("警告：状态包含NaN或Inf值，重置为有效值")
            new_state = self.reset()  # 重置为有效初始状态
            done = True  # 终止回合
            reward = -1000  # 给予负奖励
            
        # 检查奖励是否为NaN或Inf
        if np.isnan(reward) or np.isinf(reward):
            reward = -1000  # 限制为有效负奖励

        return new_state, reward, done, info

    def dir_cosine(self, q):
        """符合航空航天标准的方向余弦矩阵（论文式23）"""
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2*(q2**2 + q3**2),   2*(q1*q2 + q0*q3),   2*(q1*q3 - q0*q2)],
            [2*(q1*q2 - q0*q3),   1 - 2*(q1**2 + q3**2),   2*(q2*q3 + q0*q1)],
            [2*(q1*q3 + q0*q2),   2*(q2*q3 - q0*q1),   1 - 2*(q1**2 + q2**2)]
        ])

    def omega(self, w):
        """四元数运动学矩阵（论文式推导）"""
        return 0.5 * np.array([
            [0,  -w[0], -w[1], -w[2]],
            [w[0],  0,  w[2], -w[1]],
            [w[1], -w[2],  0,  w[0]],
            [w[2],  w[1], -w[0],  0]
        ])

    def toQuaternion(self, angle, axis):
        """轴角转四元数（防止零向量）"""
        axis = np.asarray(axis, dtype=np.float64)  # 明确指定为float64类型
        norm = np.linalg.norm(axis)
        if (norm < 1e-8):
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis /= norm
        sin_half = np.sin(angle / 2)
        return np.array([
            np.cos(angle / 2),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ], dtype=np.float64)  # 确保输出也是浮点数
