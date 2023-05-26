# 设置场景：三个agent、两个障碍物（一个动态的一个静态的）、一个目标点

import numpy as np
from multienvs.core_v1 import World, Agent, Landmark
from multienvs.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        self.formation_dis = None
        self.formation_k = None
        self.path_track_dis = None
        self.path_track_k = None

    def make_world(self):
        world = World()
        # 先设置世界属性
        world.dim_c = 2  # 通信通道维度
        num_agents = 4  # 设置了三艘船和一个虚拟领航点
        num_landmarks = 3  # 设置一个目标点和两个静态障碍物
        world.collaborative = True  # 该环境设置为“协作场景”（collaborative）
        # 添加 agent
        world.agents = [Agent() for i in range(num_agents)]
        # 设置0号是虚拟领航者
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.08  # 虚拟领航者的半径（实际是0.06，render中-0.02），跟随者的船体长0.08，宽0.05
        # 单独对虚拟领航者设置
        world.agents[0].collide = False  # 设置成不可碰撞

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        # 设置0号是目标点，1和2号是障碍物
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2  # 两个障碍物的尺寸（半径）目标点也是这个尺寸
        world.landmarks[0].collide = False
        # 设置初始条件
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # agent的属性数值设置  先这样设置，后期分开设置颜色
        for i, agent in enumerate(world.agents):
            if i == 0: agent.color = np.array([1, 0.5, 0.5])  # 红色
            elif i == 1: agent.color = np.array([0.99, 0.38, 0.28])  # 粉色
            elif i == 2: agent.color = np.array([0.59, 0.98, 0.59])  # 绿色
            elif i == 3: agent.color = np.array([0, 0.65, 0.99])  # 蓝色
            agent.state.p_pos = np.array([np.random.uniform(-1, 1), np.random.uniform(-0.2, 0.2)])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # landmark的属性数值设置
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.color = np.array([1, 0, 0])
                landmark.state.p_pos = np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(2.5, 2.9)])
            elif i == 1:
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.state.p_pos = np.array([np.random.uniform(-1, -0.1), np.random.uniform(1, 2)])
            elif i == 2:
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.state.p_pos = np.array([np.random.uniform(0.1, 1), np.random.uniform(1, 2)])

            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        # 每个agent的位置
        aa = agent.state.p_pos
        a0 = world.agents[0].state.p_pos
        a1 = world.agents[1].state.p_pos
        a2 = world.agents[2].state.p_pos
        a3 = world.agents[3].state.p_pos
        l0 = world.landmarks[0].state.p_pos
        if agent.name == "agent_0":  # 如果是虚拟领航者
            distance = self.calculate_distance(a0, l0)
            rew += 8 * np.exp(-1.8 * distance)
            rew -= distance

        elif agent.name == "agent_1":
            distance = self.calculate_distance(a1, a0)
            rew += 2 * np.exp(-1.8 * distance)
            rew -= distance * 1.2

        elif agent.name == "agent_2":
            distance = self.calculate_distance(a2, a0)
            rew += 2 * np.exp(-1.8 * distance)
            rew -= distance * 1.2

        elif agent.name == "agent_3":
            distance = self.calculate_distance(a3, a0)
            rew += 2 * np.exp(-1.8 * distance)
            rew -= distance * 1.2

        # 障碍规避部分
        if agent.collide:
            if self.is_collision(agent, world.landmarks[1]) or self.is_collision(agent, world.landmarks[2]):
                rew -= 2
            for other in world.entities:
                if other is agent or other is world.landmarks[0]:
                    continue
                if self.is_collision(agent, other):
                    rew -= 2
        return rew

    def formation_reward(self, agent, other1, other2):  # 三艘船并列跟踪虚拟领航者
        rew = 0
        agent_pos = agent.state.p_pos
        other1_pos = other1.state.p_pos
        other2_pos = other2.state.p_pos
        dis_dir = [agent_pos - other1_pos, agent_pos - other2_pos]
        distance = [self.calculate_distance(agent_pos, other1_pos), self.calculate_distance(agent_pos, other2_pos)]
        # 判断当前agent所处的位置
        if dis_dir[0][0] * dis_dir[1][0] < 0:  # 判断agent是否在两船中间
            if distance[0] < 0.3 and distance[1] < 0.3:
                rew += 1
            else:
                rew -= distance[0]*2 + distance[1]*2
        else:  # agent在编队两侧
            if distance[0] <= distance[1] and distance[0] < 0.3:
                rew += 1
            elif distance[0] > distance[1] and distance[1] < 0.3:
                rew += 1
            else:
                rew -= 1
        y_abs = abs(dis_dir[0][1]) + abs(dis_dir[1][1])
        if y_abs < 0.1:
            rew += 2
        return rew

    def observation(self, agent, world):
        # 获取所有实体在该agent参考框架中的位置
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)  # 每个landmark对当前智能体的相对位置

        # 所有其他代理人的通信
        comm = []
        other_pos = []
        # other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)  # 所有其他智能体向该智能体发送的通信信息。（可能用不上）
            other_pos.append(other.state.p_pos - agent.state.p_pos)  # 其他智能体对当前智能体的相对位置
            # other_vel.append(other.state.p_vel)
        # 返回一个包含所有实体在该智能体参考框架中的位置、所有实体的颜色信息以及其他所有智能体之间的通信信息的观测向量
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)  # 拼接数组

    def is_collision(self, agent1, agent2):  # 返回碰撞级别,级别对应惩罚值
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + 0.06
        return True if dist_min > dist else False

    def calculate_distance(self, pos1, pos2):
        delta_pos = pos1 - pos2
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist


