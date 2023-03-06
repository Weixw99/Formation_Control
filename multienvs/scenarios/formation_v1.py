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
        world.landmarks[0].collide = False
        # 设置初始条件
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # agent的属性数值设置  先这样设置，后期分开设置颜色
        for i, agent in enumerate(world.agents):
            if i == 0: agent.color = np.array([1, 0.5, 0.5])
            elif i == 1: agent.color = np.array([0.99, 0.38, 0.28])
            elif i == 2: agent.color = np.array([0.59, 0.98, 0.59])
            elif i == 3: agent.color = np.array([0, 0.65, 0.99])
            agent.state.p_pos = np.array([np.random.uniform(-1, 1), np.random.uniform(-0.7, -0.9)])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # landmark的属性数值设置
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.color = np.array([1, 0, 0])
                landmark.state.p_pos = np.array([np.random.uniform(-1, 1), np.random.uniform(0.9, 1.0)])
            else:
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.state.p_pos = np.array([np.random.uniform(-1, 1), np.random.uniform(-0.2, 0.4)])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        # 每个agent的位置
        h0 = world.agents[0].state.p_pos
        h1 = world.agents[1].state.p_pos
        h2 = world.agents[2].state.p_pos
        h3 = world.agents[3].state.p_pos
        l0 = world.landmarks[0].state.p_pos
        formation_dis = 0.2  # 编队期望距离
        formation_k = 0.5  # 编队弹性连接力系数
        path_track_dis = 0.3  # 弹性路径跟踪期望距离
        path_track_k = 0.8

        if agent.name == "agent 0":  # 如果是虚拟领航者
            distance_aim = self.calculate_distance(h0, l0)
            if distance_aim < 0.3:
                rew += 1.6
            rew -= distance_aim * 0.8
        elif agent.name == "agent 1":
            dis_dir1 = [h1 - h2, h1 - h3]  # 如果x计算出为正，则other在agent在左边
            distance = [self.calculate_distance(h1, h0), self.calculate_distance(h1, h1),
                        self.calculate_distance(h1, h2), self.calculate_distance(h1, h3)]
            # 编队控制部分
            if dis_dir1[0][0]*dis_dir1[1][0] < 0:  # 判断agent1是否在两船中间
                # 计算与其他智能体之间的弹力
                f12 = -formation_k * (distance[2] - formation_dis)**2
                f13 = -formation_k * (distance[3] - formation_dis)**2
            else:  # agent1在编队两侧
                if distance[2] < distance[3]:
                    f12 = -formation_k * (distance[2] - formation_dis)**2
                    f13 = 0
                else:
                    f12 = 0
                    f13 = -formation_k * (distance[3] - formation_dis)**2
            # 弹性路径跟踪部分
            f10 = -path_track_k * (distance[0] - path_track_dis)
            f = f10 + f12 + f13
            rew += f
            # 障碍规避部分
            if self.is_collision(agent, world.landmarks[1]) or self.is_collision(agent, world.landmarks[2]):
                rew -= 10
            for other in world.agents:
                if agent is other:
                    continue
                if self.is_collision(other, agent):
                    rew -= 10

        elif agent.name == "agent 2":
            dis_dir2 = [h2 - h1, h2 - h3]
            distance = [self.calculate_distance(h2, h0), self.calculate_distance(h2, h1),
                        self.calculate_distance(h2, h2), self.calculate_distance(h2, h3)]
            # 编队控制部分
            if dis_dir2[0][0] * dis_dir2[1][0] < 0:  # 判断agent1是否在两船中间
                # 计算与其他智能体之间的弹力
                f21 = -formation_k * (distance[1] - formation_dis) ** 2
                f23 = -formation_k * (distance[3] - formation_dis) ** 2
            else:  # agent1在编队两侧
                if distance[1] < distance[3]:
                    f21 = -formation_k * (distance[1] - formation_dis) ** 2
                    f23 = 0
                else:
                    f21 = 0
                    f23 = -formation_k * (distance[3] - formation_dis) ** 2
            # 弹性路径跟踪部分
            f20 = -path_track_k * (distance[0] - path_track_dis)
            f = f20 + f21 + f23
            rew += f
            # 障碍规避部分
            if self.is_collision(agent, world.landmarks[1]) or self.is_collision(agent, world.landmarks[2]):
                rew -= 10
            for other in world.agents:
                if agent is other:
                    continue
                if self.is_collision(other, agent):
                    rew -= 10

        elif agent.name == "agent 3":
            dis_dir3 = [h3 - h1, h3 - h2]
            distance = [self.calculate_distance(h3, h0), self.calculate_distance(h3, h1),
                        self.calculate_distance(h3, h2), self.calculate_distance(h3, h3)]
            # 编队控制部分
            if dis_dir3[0][0] * dis_dir3[1][0] < 0:  # 判断agent1是否在两船中间
                # 计算与其他智能体之间的弹力
                f31 = -formation_k * (distance[1] - formation_dis) ** 2
                f32 = -formation_k * (distance[2] - formation_dis) ** 2
            else:  # agent1在编队两侧
                if distance[1] < distance[2]:
                    f31 = -formation_k * (distance[1] - formation_dis) ** 2
                    f32 = 0
                else:
                    f31 = 0
                    f32 = -formation_k * (distance[2] - formation_dis) ** 2
            # 弹性路径跟踪部分
            f30 = -path_track_k * (distance[0] - path_track_dis)
            f = f30 + f31 + f32
            rew += f
            # 障碍规避部分
            if self.is_collision(agent, world.landmarks[1]) or self.is_collision(agent, world.landmarks[2]):
                rew -= 10
            for other in world.agents:
                if agent is other:
                    continue
                if self.is_collision(other, agent):
                    rew -= 10
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
