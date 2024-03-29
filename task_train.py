import numpy as np
import tensorflow as tf
import time
import pickle
import os
import common.utils as util
from agent import make_env, get_trainers
import config
import APF


class Runner:
    def __init__(self, parameters):
        self.parameters = parameters
        # 创建环境
        self.env = make_env(parameters.scenario)
        # 创建agent
        obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]
        adversaries_num = min(self.env.n, parameters.adversaries_num)
        self.trainers = get_trainers(self.env, adversaries_num, obs_shape_n, parameters)
        print('Use scenario:{},algo:{},device:{}'.format(parameters.scenario, parameters.algo_name, parameters.device))

        # 初始化
        util.initialize()  # 确保所有的变量都已经被正确地初始化。
        # 判断是否加载上次训练好的模型,如果加载上次的模型查看效果，就读取上次的训练结果。
        if parameters.evaluate:
            print('Loading previous state...')
            util.load_state(parameters.load_dir)  # 加载模型
            print(f"load_dir:{parameters.load_dir}")
        else:  # 如果训练代码，则在环境文件下创建新的文件存放网络参数
            if not os.path.exists(parameters.save_dir):
                os.makedirs(parameters.save_dir)
            self.saver = tf.compat.v1.train.Saver()
            print(f"save_dir:{parameters.save_dir}")

        if parameters.use_wandb:
            self.wandb_run = config.MyWandb(parameters)
            self.wandb_run.wandb_init()
        self.episode_step = 0
        self.train_step = 0
        self.noise_std = self.parameters.noise_std_init if not parameters.evaluate else 0

    def run(self):
        episode_rewards = []  # sum of rewards for all agents
        agent_rewards = [[] for _ in range(self.env.n)]  # individual agent reward
        if self.parameters.is_exp:  # 判断是否在进行实验
            print('Starting connection...')
            from DRL_Client import TCPClient
            usv1 = TCPClient('192.168.1.3', 1230)
            usv2 = TCPClient('192.168.1.4', 1231)
            usv3 = TCPClient('192.168.1.5', 1232)
            dis_rate = 20
        print('Starting iterations...')
        t_start = time.time()
        while self.train_step < self.parameters.max_train_num:
            train_info = {}
            self.train_step += 1
            self.parameters.train_step = self.train_step
            episode_rewards.append(0)
            for a in agent_rewards: a.append(0)
            obs_n = self.env.reset()
            if self.parameters.use_apf:
                self.parameters.apf_noise -= self.parameters.apf_noise_decay if self.parameters.apf_noise > 0 else 0
                apf = APF.MyAPF(self.env.world.agents, self.env.world.landmarks)
            if not self.parameters.evaluate:
                for agent in self.env.agents:
                    agent.u_noise = self.noise_std
                # Decay noise_std(衰减噪音)
                if self.parameters.use_noise_decay:
                    self.noise_std = self.noise_std - self.parameters.noise_std_decay if self.noise_std - self.parameters.noise_std_decay > self.parameters.noise_std_min else self.parameters.noise_std_min
            for _ in range(self.parameters.max_episode_len):
                if self.parameters.is_exp:
                    rev_data1 = usv1.get_data()  # 等待接收数据,收到处理好返回 (N, E)
                    rev_data2 = usv2.get_data()  # 等待接收数据,收到处理好返回
                    rev_data3 = usv3.get_data()  # 等待接收数据,收到处理好返回
                    usv_position = [[rev_data1[1] / dis_rate, rev_data1[0] / dis_rate],  # (E, N)
                                    [rev_data2[1] / dis_rate, rev_data2[0] / dis_rate],
                                    [rev_data3[1] / dis_rate, rev_data3[0] / dis_rate]]  # 统一数量级
                    print('usv_position:', usv_position)
                    obs_n[1][2], obs_n[1][3] = usv_position[0][0], usv_position[0][1]
                    obs_n[2][2], obs_n[2][3] = usv_position[1][0], usv_position[1][1]
                    obs_n[3][2], obs_n[3][3] = usv_position[2][0], usv_position[2][1]
                # 获取apf的力
                if self.parameters.use_apf:
                    for agent, obs in zip(self.env.world.policy_agents, obs_n):
                        agent.apf_noise = apf.compute_force(agent, obs)*self.parameters.apf_noise
                # 获取action
                action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
                # step
                new_obs_n, rew_n, done_n, info_n = self.env.step(action_n, obs_n)
                if self.parameters.is_exp:
                    send_pos1 = [new_obs_n[1][3] * dis_rate, new_obs_n[1][2] * dis_rate]  # (N, E)
                    send_pos2 = [new_obs_n[2][3] * dis_rate, new_obs_n[2][2] * dis_rate]
                    send_pos3 = [new_obs_n[3][3] * dis_rate, new_obs_n[3][2] * dis_rate]

                    usv1.send_data(send_pos1)  # 将下一步的目标点发送出去
                    usv2.send_data(send_pos2)  # 将下一步的目标点发送出去
                    usv3.send_data(send_pos3)  # 将下一步的目标点发送出去
                # collect experience
                for i, agent in enumerate(self.trainers):  # 将新的经验元组（experience）添加到每个智能体的经验缓存（replay buffer）中
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
                obs_n = new_obs_n

                self.episode_step += 1
                if all(done_n):
                    break

                for i, rew in enumerate(rew_n):  # 奖励累计
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew
                # 显示
                if self.parameters.evaluate:
                    time.sleep(0.1)
                    self.env.render()
                    continue

                # 如果不是在显示或基准模式下，更新所有训练者
                for agent in self.trainers:
                    agent.preupdate()
                    loss = agent.update(self.trainers, self.episode_step)
                    if loss:
                        train_info['%s/q_loss' % agent.name] = loss[0]
                        train_info['%s/p_loss' % agent.name] = loss[1]
                        train_info['%s/target_q_mean' % agent.name] = loss[2]
                        train_info['%s/rew' % agent.name] = loss[3]
                        train_info['%s/target_q_next' % agent.name] = loss[4]
                        train_info['%s/target_q_sta' % agent.name] = loss[5]
            # 保存模型，显示训练输出
            if self.train_step % self.parameters.save_rate == 0 or self.parameters.evaluate:
                # 打印状态
                print("episodes_steps: {}, train_steps: {}, mean reward: {}, agent reward: {}, time: {}".format(
                        self.episode_step, self.train_step, np.mean(episode_rewards[-self.parameters.save_rate:]),
                        [np.mean(rew[-self.parameters.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                if self.parameters.evaluate: continue
                # save model
                util.save_state(self.parameters.save_dir, saver=self.saver)
                train_info['reward'] = np.mean(episode_rewards[-self.parameters.save_rate:])
                for i, rew in enumerate(agent_rewards):
                    train_info['agent_%i/reward' % i] = np.mean(rew[-self.parameters.save_rate:])
                if self.parameters.use_wandb: self.wandb_run.wandb_log(train_info, len(episode_rewards))

        print('Finished total of {} episodes.'.format(self.train_step))
        if self.parameters.use_wandb: self.wandb_run.wandb_finish()


if __name__ == '__main__':
    parameter = config.get_config()
    with util.single_threaded_session():  # 创建一个TensorFlow会话,确保TensorFlow运算的正确执行,同时避免了多线程环境下的竞争和不一致性问题。
        runner = Runner(parameter)
        runner.run()
