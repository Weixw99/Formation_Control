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
# 代理人的状态（包括通信和内部/精神状态
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
        self.damping = 0.25
        # 联系响应参数
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.t = 0
