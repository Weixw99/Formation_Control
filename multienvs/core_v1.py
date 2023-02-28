# 包含各种对象的类（实体、地标、代理等）在整个代码中使用
import numpy as np


# physical/external base state of all entities
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
# 代理人的状态（包括通信和内部/精神状态)
class AgentState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


# 物理世界实体的属性和状态
class Entity(object):
    def __init__(self):
        # name
        self.name = ''
        # properties（属性）:
        self.size = 0.050
        # 实体可以移动/被推动
        self.movable = False
        # 实体与他人相撞:如果一个智能体的 collide 属性为 True，则该智能体可以与其他智能体发生碰撞；否则，它将穿过其他智能体，不会受到碰撞的影响。
        self.collide = True
        # 材料密度（影响质量）
        self.density = 25.0
        # color（颜色）
        self.color = None
        # 最大速度和加速度
        self.max_speed = None
        self.accel = None
        # state（状态）
        self.state = EntityState()
        # 初始质量
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of agent entities  代理实体的属性
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent默认是可以移动的
        self.movable = True
        # 不能发送通信信号
        self.silent = False
        # 不能观察世界
        self.blind = False
        # 物理马达噪音量
        self.u_noise = None
        # 通信噪声量
        self.c_noise = None
        # 控制范围
        self.u_range = 1.0
        # 状态（state）
        self.state = AgentState()
        # 动作（action）
        self.action = Action()
        # 要执行的脚本行为
        self.action_callback = None


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


# Multi_agent world
class World(object):
    def __init__(self):
        # 代理和实体的列表(可以在执行时更改!)
        self.agents = []
        self.landmarks = []
        # 通信通道维度
        self.dim_c = 0
        # 位置维度
        self.dim_p = 2
        # 颜色维度
        self.dim_color = 3
        # 仿真时间步长
        self.dt = 0.1
        # 物理阻尼
        self.damping = 0.25  # 使用damping参数来模拟摩擦和其能量损失的影响，从而减少实体的速度
        # 联系响应参数
        self.contact_force = 1e+2  # 超参数，表示碰撞时实体施加的力大小
        self.contact_margin = 1e-3  # 是一个常数，用于控制实体之间的最小距离
        self.t = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

        # update state of the world

    def step(self):
        # 收集应用于实体（entities）entities的力量
        p_force = [None] * len(self.entities)
        # 应用agent的物理控制
        p_force = self.apply_action_force(p_force)
        # 应用环境的力
        p_force = self.apply_environment_force(p_force)
        # 整合物理状态
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        self.t += 1

    # 收集agent的行动力量
    def apply_action_force(self, p_force):
        # 设置被应用力
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # 收集作用于实体的物理力量
    def apply_environment_force(self, p_force):
        # 简单（但效率不高）的碰撞反应
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):  # 计算所有实体之间的力
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)  # 得到两个实体碰撞的力,如果没有碰撞几乎为0
                if f_a is not None:  # A和B之间有力
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]  # 对a来说，action的力和环境中计算出来的力相加结合
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]  # 对b来说，action的力和环境中计算出来的力相加结合
        return p_force  # 最后所有的力计算完了，得出来的向量为最后的总力

    # 获得两个实体之间任何接触的碰撞力
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):  # 如果是True，则该实体可以于其他agent发生碰撞
            return [None, None]  # 有非碰撞者
        if entity_a is entity_b:
            return [None, None]  # 不能自己撞自己
        # 计算实体之间的实际距离
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # 最小允许距离
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k  # 两个实体之间的接触程度， 越大表示两个实体重叠得越多，需要更大的力来分离它们。
        # 上述函数是一个类似于指数函数的一个曲线，在距离很小时趋近于0，而在距离超过最小距离时快速增加到1
        force = self.contact_force * delta_pos / dist * penetration  # delta_pos / dist可以看做是一个单位向量，指代方向
        force_a = +force if entity_a.movable else None   # force是一个向量，表示力的指向
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # 整合物理状态
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)  # 更新实体速度，用于减缓速度，如果不施加外力，速度将一直减少
            if i == 3:
                if entity.state.p_pos[0] < -0.95:
                    entity.state.p_vel += [0.04, 0]
                elif -0.95 <= entity.state.p_pos[0] <= 0.95 and entity.state.p_vel[0] > 0:
                    entity.state.p_vel += [0.04, 0]

                elif entity.state.p_pos[0] > 0.95:
                    entity.state.p_vel += [-0.04, 0]
                elif entity.state.p_pos[0] <= 0.95 and entity.state.p_vel[0] < 0:
                    entity.state.p_vel += [-0.04, 0]
            if i == 4:
                if entity.state.p_pos[0] < -0.8:
                    entity.state.p_vel += [0.014, 0]
                elif -0.8 <= entity.state.p_pos[0] <= 0.65 and entity.state.p_vel[0] > 0:
                    entity.state.p_vel += [0.014, 0]

                elif entity.state.p_pos[0] > 0.65:
                    entity.state.p_vel += [-0.014, 0]
                elif entity.state.p_pos[0] <= 0.65 and entity.state.p_vel[0] < 0:
                    entity.state.p_vel += [-0.014, 0]
            else:
                if p_force[i] is not None:
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt  # 牛二：F=ma，根据dt求出dv，即单位时间内的速度变化
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))  # 对速度平方开方，得实际速度
                if speed > entity.max_speed:
                    # 如果超过最大速度，则将速度调整为最大速度。 将速度向量除以其模长得到单位向量，再乘上最大速度得到新的速度向量，从而保持速度方向不变。
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt  # 根据物理公式 s = v * t，可以得到新位置 s' = s + v * t

    def update_agent_state(self, agent):
        # 设置通信状态（目前是直接设置）
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise