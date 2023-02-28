# 设置场景：三个agent、两个障碍物（一个动态的一个静态的）、一个目标点

import numpy as np
from multienvs.core_v1 import World, Agent, Landmark
from multienvs.scenario import BaseScenario


class Scenario(BaseScenario):
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
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # 单独对虚拟领航者设置
        world.agents[0].collide = False  # 设置成不可碰撞

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        # 设置0号是目标点，1和2号是障碍物
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
        # 设置初始条件
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # agent的属性数值设置  先这样设置，后期分开设置颜色
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # landmark的属性数值设置
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        # 每个agent的位置,应该设置每个agent的奖励，因为会引用多次
        h0 = world.agents[0].state.p_pos
        h1 = world.agents[1].state.p_pos
        h2 = world.agents[2].state.p_pos
        h3 = world.agents[3].state.p_pos
        distance = [[self.calculate_distance(h0, h0), self.calculate_distance(h0, h1), self.calculate_distance(h0, h2), self.calculate_distance(h0, h3)],
                    [self.calculate_distance(h1, h0), self.calculate_distance(h1, h1), self.calculate_distance(h1, h2), self.calculate_distance(h1, h3)],
                    [self.calculate_distance(h2, h0), self.calculate_distance(h2, h1), self.calculate_distance(h2, h2), self.calculate_distance(h2, h3)],
                    [self.calculate_distance(h3, h0), self.calculate_distance(h3, h1), self.calculate_distance(h3, h2), self.calculate_distance(h3, h3)]]

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

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def calculate_distance(self, pos1, pos2):
        delta_pos = pos1 - pos2
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist
