import numpy as np


class MyAPF:
    def __init__(self, agents, landmarks):
        self.agent = agents
        self.landmark = landmarks
        self.l0 = landmarks[0].state.p_pos  # 目标点坐标
        self.l1 = landmarks[1].state.p_pos  # 障碍物1的坐标
        self.l2 = landmarks[2].state.p_pos  # 障碍物2的坐标
        self.last_force = np.zeros(2)
        # 引力增益和阈值
        self.attract_k = 2.0
        self.attract_swing_threshold = 0.2  # 如果到达该范围内，则认为到达目标点，引力设为0
        # 斥力增益和阈值
        self.repulsion_k = 5.0
        self.repulsion_Threshold = landmarks[1].size + 0.15

    def compute_force(self, agent, obs):  # 此处的agent时网络的agent和env中的agent有所不同
        delta_a_l = np.array([[-obs[4], -obs[5]], [obs[6], obs[7]], [obs[8], obs[9]]]) * -1  # l_pos-agent_pos
        delta_a_a = np.array(
            [[obs[10], obs[11]], [obs[12], obs[13]], [obs[14], obs[15]]]) * -1  # agent和其他agent的坐标差，other - agent
        global_target_point = np.array([obs[4], obs[5]])  # 对于agent0来说，使用全局目标点，方向指向目标点
        local_target_point = np.array([obs[10], obs[11]])  # 对于其他agent来说，将agent0作为目标点，方向指向目标点
        if agent.name == 'agent_0':
            attract_force = self.compute_attract(global_target_point)
            repulsion_force = np.zeros(2)
        else:
            attract_force = self.compute_attract(local_target_point) + 0.2*self.compute_attract(global_target_point)
            repulsion_force = self.compute_repulsion(delta_a_l[1:], delta_a_a[1:], local_target_point)  # 暂时不计算agent之间的力
        angel_a_r = self.compute_angle(attract_force, repulsion_force)
        force = self.compute_middle_vector(attract_force, repulsion_force) if angel_a_r > 120 else (attract_force + repulsion_force)
        force = self.compute_middle_vector(force, self.last_force, truth1=0.7) if self.compute_angle(force, self.last_force) > 90 else force
        self.last_force = force
        return force / np.sqrt(np.sum(np.square(force))) if np.sqrt(np.sum(np.square(force))) != 0 else force

    def compute_attract(self, delta):  # 计算引力
        attract_force = np.zeros(2)
        distance = np.sqrt(np.sum(np.square(delta)))
        if self.attract_swing_threshold < distance:
            attract_force = self.attract_k * delta
        return attract_force

    def compute_repulsion(self, delta_l, delta_a, delta_goal):  # 计算斥力  先不计算agent之间的斥力
        repulsion_force = np.zeros(2)
        for l in delta_l:
            distance_l_to_a = np.sqrt(np.sum(np.square(l)))
            distance_g_to_a = np.sqrt(np.sum(np.square(delta_goal)))
            if distance_l_to_a < self.repulsion_Threshold:
                repulsion_force1 = self.repulsion_k * l / distance_l_to_a * \
                                   (1.0 / distance_l_to_a - 1.0 / self.repulsion_Threshold) / \
                                   (distance_l_to_a ** 2) * (distance_g_to_a ** 2)
                repulsion_force2 = self.repulsion_k * delta_goal / distance_g_to_a * \
                                   (1.0 / distance_l_to_a - 1.0 / self.repulsion_Threshold) ** 2 * distance_g_to_a
                repulsion_force += repulsion_force1 + repulsion_force2
        return repulsion_force

    def compute_angle(self, vector1, vector2):  # 计算两个向量之间的夹角
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
            return np.degrees(np.arccos(cos_angle))
        else:
            return 0.0

    def compute_middle_vector(self, vector1, vector2, truth1=0.5):
        return ((vector1/np.sqrt(np.sum(np.square(vector1))))*truth1 + (vector2/np.sqrt(np.sum(np.square(vector2))))*(1-truth1)) / 2


class MyTEST:
    def __init__(self):
        self.last_force = np.zeros(2)
        # 引力增益和阈值
        self.attract_k = 1.0
        self.attract_swing_threshold = 0.1  # 如果到达该范围内，则认为到达目标点，引力设为0
        # 斥力增益和阈值
        self.repulsion_k = 5
        self.repulsion_Threshold = 0.1 + 0.15

    def compute_force(self, obs):  # 此处的agent时网络的agent和env中的agent有所不同
        delta_a_l = np.array([[-obs[0][0], -obs[0][1]], [obs[1][0], obs[1][1]],
                              [obs[2][0], obs[2][1]]]) * -1  # agent和landmark之间的坐标差，l_pos-agent_pos
        attract_force = self.compute_attract(delta_a_l[0])
        repulsion_force = self.compute_repulsion(delta_a_l[1:], delta_a_l[0])
        if self.compute_angle(attract_force, repulsion_force) > 120:
            force = self.compute_middle_vector(attract_force, repulsion_force)
        else:
            force = attract_force + repulsion_force
        if self.compute_angle(force, self.last_force) > 90:
            force = self.compute_middle_vector(force, self.last_force, truth1=0.7)
        # force = attract_force + repulsion_force if self.compute_angle(attract_force, repulsion_force) < 120 else self.compute_middle_vector(attract_force, repulsion_force)
        self.last_force = force
        return force / np.sqrt(np.sum(np.square(force)))

    def compute_attract(self, delta):  # 计算引力
        attract_force = np.zeros(2)
        distance = np.sqrt(np.sum(np.square(delta)))
        if self.attract_swing_threshold < distance:
            attract_force = self.attract_k * delta
        return attract_force

    def compute_repulsion(self, delta_l, delta_goal):  # 计算斥力
        repulsion_force = np.zeros(2)
        for l in delta_l:
            distance_l_to_a = np.sqrt(np.sum(np.square(l)))
            distance_g_to_a = np.sqrt(np.sum(np.square(delta_goal)))
            if distance_l_to_a < self.repulsion_Threshold:
                repulsion_force1 = self.repulsion_k * l / distance_l_to_a * \
                                   (1.0 / distance_l_to_a - 1.0 / self.repulsion_Threshold) / \
                                   (distance_l_to_a ** 2) * (distance_g_to_a ** 2)
                repulsion_force2 = self.repulsion_k * delta_goal / distance_g_to_a * \
                                   (1.0 / distance_l_to_a - 1.0 / self.repulsion_Threshold) ** 2 * distance_g_to_a
                repulsion_force += repulsion_force1 + repulsion_force2
        return repulsion_force

    def compute_angle(self, vector1, vector2):  # 计算两个向量之间的夹角
        return np.degrees(np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))))

    def compute_middle_vector(self, vector1, vector2, truth1=0.5):
        return ((vector1/np.sqrt(np.sum(np.square(vector1))))*truth1 + (vector2/np.sqrt(np.sum(np.square(vector2))))*(1-truth1)) / 2


class PathPlotter:
    def __init__(self, obs_xy, goal_xy):
        self.fig, self.ax = plt.subplots()
        self.obs_xy = obs_xy
        self.goal_xy = goal_xy
        # 绘制障碍物
        obs_x = [p[0] for p in self.obs_xy]  # 提取障碍物的x坐标
        obs_y = [p[1] for p in self.obs_xy]  # 提取障碍物的y坐标
        self.ax.scatter(obs_x, obs_y, color='black', marker='s')  # 将障碍物绘制为黑色正方形
        repulsors = [(p, 0.25) for p in self.obs_xy]  # 存储每个斥力场域的中心点和半径
        for repulsor in repulsors:
            center, radius = repulsor
            circle = plt.Circle(center, radius, color='gray', alpha=0.3)  # 障碍物周围的斥力场域用灰色填充
            self.ax.add_patch(circle)  # 将圆形添加到图像中
        # 绘制目标点
        self.ax.scatter(goal_xy[0], goal_xy[1], color='green', marker='o')
        # 设置坐标轴范围和标签
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        plt.ion()  # 开启交互模式

    def update_path(self, paths):
        self.ax.lines.clear()  # 清空之前绘制的路径
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        self.ax.plot(path_x, path_y, color='red')
        self.fig.canvas.draw()  # 绘制更新后的图像
        plt.pause(0.1)  # 暂停一段时间，以便图像有足够的时间被绘制出来

    def path_plot_finish(self):
        plt.show(block=False)  # 非阻塞地显示图形
        # 等待图形窗口关闭
        while plt.get_fignums():
            plt.pause(0.1)
        print("程序运行结束")


def plot_path(paths, obs_xy, goal_xy):
    import matplotlib.pyplot as plt
    # 绘制障碍物
    obs_x = [p[0] for p in obs_xy]  # 提取障碍物的x坐标
    obs_y = [p[1] for p in obs_xy]  # 提取障碍物的y坐标
    plt.scatter(obs_x, obs_y, color='black', marker='s')  # 将障碍物绘制为黑色正方形
    repulsors = []  # 存储每个斥力场域的中心点和半径
    for ob in obs_xy:
        center = ob  # 中心点为障碍物的位置
        radius = 0.25  # 半径为0.25
        repulsors.append((center, radius))
    for repulsor in repulsors:
        center, radius = repulsor
        circle = plt.Circle(center, radius, color='gray', alpha=0.3)  # 障碍物周围的斥力场域用灰色填充
        plt.gca().add_patch(circle)  # 将圆形添加到图像中
    # 绘制目标点
    plt.scatter(goal_xy[0], goal_xy[1], color='green', marker='o')  # 将目标点绘制为绿色圆形
    # 绘制路径
    path_x = [p[0] for p in paths]  # 提取所有点的x坐标
    path_y = [p[1] for p in paths]  # 提取所有点的y坐标
    plt.plot(path_x, path_y, color='red')  # 将所有点连接起来，并用红色线条绘制
    # 设置坐标轴范围和标签
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    # 显示绘制的图像
    plt.show()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    # 相关参数设置
    step_size, max_iters = 0.1, 500  # 步长0.5寻路1000次用时4.37s, 步长0.1寻路1000次用时21s
    step_size_ = 2
    start, goal = [-1.0, -1.0], [1.0, 1.0]
    obstacles = np.array([goal, [0.0, -0.02], [0.7, 0.5]])
    current_pos = np.array(start)
    wxw = MyTEST()
    path = [current_pos.copy()]
    success = False
    pp = PathPlotter(obstacles[1:], obstacles[0])
    for iters in range(max_iters):
        obs = obstacles - current_pos
        forces = wxw.compute_force(obs)
        current_pos += forces * step_size
        path.append(current_pos.copy())
        pp.update_path(path)
        if np.sqrt(np.sum(np.square(obs[0]))) < 0.1:
            success = True
            break
    print(success)
    print(path)
    pp.path_plot_finish()
    # plot_path(path, obstacles[1:], obstacles[0])
