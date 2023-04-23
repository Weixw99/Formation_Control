import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multienvs.multi_discrete import MultiDiscrete


# 目前的代码假设在运行时没有代理被创建/销毁
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }  # 在注册环境时，会将metadata信息传递给Gym，用于描述环境的基本属性和特征。

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.world = world
        # scenario callbacks
        self.reset_callback = reset_callback  # reset_callback对应的是scenario.reset_world
        self.reward_callback = reward_callback  # reward_callback对应的是scenario.reward
        self.observation_callback = observation_callback  # observation_callback对应的是scenario.observation
        self.info_callback = info_callback  # 如果有benchmark是true，则info_callback对应于scenario.benchmark_data
        self.done_callback = done_callback
        self.shared_viewer = shared_viewer  # render那一部分的

        self.agents = world.policy_agents
        # 设置所需的矢量gym环境属性 设置gym环境的属性
        self.n = len(world.policy_agents)
        # 环境参数
        self.discrete_action_space = True
        # 如果为真，行动是一个数字0...N，否则行动是一个one-hot N-dimensional vector
        self.discrete_action_input = False
        # 如果为真，即使行动是连续的，行动也将被离散地执行。
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # 如果是真的，每个agent都有相同的奖励
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        # self.shared_reward = False
        self.time = 0

        # 配置空间
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical 动作空间
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # 总的行动空间
            if len(total_action_space) > 1:
                # 所有行动空间都是离散的，所以简化为多离散行动空间
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

            # rendering
            if self.shared_viewer:
                self.viewers = [None]
            else:
                self.viewers = [None] * self.n
            self._reset_render()

    def step(self, action_n, state):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])  # 设置所有 agents 的动作
        # 根据action更新world的状态
        self.world.step()  # 执行的是core文件中定义world的析造函数
        # 记录每个agent的observation
        for agent in self.agents:  # 先进入函数判断是否有回调函数，基本都会有，然后就会调用类中引用的函数
            obs_n.append(self._get_obs(agent))  # 相当于内部函数调用scenario.observation，也就是formation_v1中的observation
            reward_n.append(self._get_reward(agent))  # 同上，调用scenario.reward，也就是formation_v1中的reward
            done_n.append(self._get_done(agent))  # 同上，但没有调用，一直返回的是False
            info_n['n'].append(self._get_info(agent))  # 只要没有基准测试，返回是空
        # 在合作的情况下，所有的agents都能获得的总奖励
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):  # 该函数在主程序中引用，返回到主程序中初始化的观测
        # reset world
        self.reset_callback(self.world)  # 会直接执行formation_v1中的reset_world，主要设置了初始位置、速度和颜色
        self._reset_render()  # 初始化渲染器，就在这个类中
        # 记录每个agent的observation
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # 为一个特定的agent设置环境动作
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):  # 判断动作空间是不是多维离散空间
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:  # 判断输入是否为离散值，如果是，则将对应的动作值映射到对应的物理行为上
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0  # 即上下左右
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:  # 判断是否使用强制离散动作，如果是，则将输出的动作中最大值设为1，其余设为0，
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:  # 如果动作空间是离散的，则将对应的动作值映射到对应的物理行为上；
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:  # 如果是连续的，则直接将输出的动作值作为物理行为。
                    agent.action.u = action[0]
            sensitivity = 5.0  # 物理动作灵敏度
            if agent.accel is not None:
                sensitivity = agent.accel
            action = action[1:]

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0  # 常常被用于程序调试和单元测试中，如果assert条件不成立，程序就会停止运行，并且会提示AssertionError异常

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    # 获取某一特定agent的observation
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    # 获取某一特定agent的reword
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # 重置渲染环境中的几何图形和几何变换，当重置这两个变量时，就相当于清空了渲染环境中已有的几何图形，以便后续重新渲染新的几何图形。
    def _reset_render(self):
        self.render_geoms = None  # 存储了当前需要渲染的几何图形对象,重置为空
        self.render_geoms_xform = None  # 存储了对应的几何变换矩阵,重置为空

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for i, agent in enumerate(self.world.agents):
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
                    mylog = open('recode.log', mode='a')
                    print(agent.state.p_pos, file=mylog)
                    mylog.close()

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # 只有在我们需要时才导入渲染（对于无头机器不导入）。
                # from gym.envs.classic_control import rendering
                from multienvs import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # 创建渲染几何
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multienvs import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            for i, agent in enumerate(self.world.agents):
                if i == 0:  # 对虚拟领航点设置
                    geom = rendering.make_circle(agent.size - 0.02)
                else:  # 对三艘船进行设置
                    boat_width = 0.05
                    boat_height = 0.06
                    bias = 0.04
                    l, r, t, b = -boat_width / 2, boat_width / 2, boat_height, 0
                    geom = rendering.FilledPolygon(
                        [(l, b - bias), (l, t - bias), (0, 0.08 - bias), (r, t - bias), (r, b - bias)])
                xform = rendering.Transform()
                geom.set_color(*agent.color, alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            for i, landmark in enumerate(self.world.landmarks):
                if i == 0:  # 目标点设置为三角形
                    circle = landmark.size / 2
                    l, r, t, b = 0.086 / 2, circle / 2, circle, 0
                    geom = rendering.FilledPolygon([(-l, -r), (b, t), (l, -r)])
                else:
                    geom = rendering.make_circle(landmark.size)
                xform = rendering.Transform()
                geom.set_color(*landmark.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            # 在查看器中添加几何图形
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multienvs import rendering
            # 更新边界，使之以agent为中心
            cam_range = 1.7
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range,
                                       pos[1] + cam_range)
            # update geometry positions

            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))
        return results

