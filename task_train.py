import numpy as np
import tensorflow as tf
import time
import pickle
import os
import common.utils as util
from agent import make_env, get_trainers
import config


def train(parameters):
    with util.single_threaded_session():  # 创建一个TensorFlow会话,确保TensorFlow运算的正确执行,同时避免了多线程环境下的竞争和不一致性问题。
        # 创建环境
        env = make_env(parameters.scenario, parameters.benchmark)
        # 创建agent
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        adversaries_num = min(env.n, parameters.adversaries_num)
        trainers = get_trainers(env, adversaries_num, obs_shape_n, parameters)
        print('Use scenario:{},algo:{},device:{}'.format(parameters.scenario, parameters.algo_name, parameters.device))

        # 初始化
        util.initialize()  # 确保所有的变量都已经被正确地初始化。

        parameters.save_dir = os.path.join(parameters.save_dir, parameters.scenario)
        if not os.path.exists(parameters.save_dir):
            os.makedirs(parameters.save_dir)
        total_files = len([file for file in os.listdir(parameters.save_dir)])
        # 判断是否加载上次训练好的模型,如果加载上次的模型查看效果，就读取上次的训练结果。
        if parameters.display or parameters.restore or parameters.benchmark:
            if parameters.load_dir == "":
                parameters.save_dir = os.path.join(parameters.save_dir, f'{total_files}' + '/')
                parameters.load_dir = parameters.save_dir
            print('Loading previous state...')
            util.load_state(parameters.load_dir)  # 加载模型
        else:  # 如果训练代码，则在环境文件下创建新的文件存放网络参数
            parameters.save_dir = os.path.join(parameters.save_dir, f'{total_files + 1}' + '/')
            os.makedirs(parameters.save_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.compat.v1.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        if parameters.use_wandb:
            wandb_run = config.MyWandb(parameters)
            wandb_run.wandb_init()
            train_info = {}
        print('Starting iterations...')
        while True:
            # 获取action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n, obs_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= parameters.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):  # 将新的经验元组（experience）添加到每个智能体的经验缓存（replay buffer）中
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):  # 奖励累计
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew


            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
            # 递增全局步数计数器
            train_step += 1
            # 用于衡量所学policy的基准
            if parameters.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > parameters.benchmark_iters and (done or terminal):
                    file_name = parameters.benchmark_dir + parameters.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue
            # 显示
            if parameters.display:
                time.sleep(0.1)
                env.render()
                continue

            # 如果不是在显示或基准模式下，更新所有训练者
            for agent in trainers:
                agent.preupdate()
                loss = agent.update(trainers, train_step)
                if loss:
                    train_info['%s/q_loss' % agent.name] = loss[0]
                    train_info['%s/p_loss' % agent.name] = loss[1]
                    train_info['%s/target_q_mean' % agent.name] = loss[2]
                    train_info['%s/rew' % agent.name] = loss[3]
                    train_info['%s/target_q_next' % agent.name] = loss[4]
                    train_info['%s/target_q_sta' % agent.name] = loss[5]
            # 保存模型，显示训练输出
            if terminal and (len(episode_rewards) % parameters.save_rate == 0):
                util.save_state(parameters.save_dir, saver=saver)
                # 打印状态
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-parameters.save_rate:]),
                    [np.mean(rew[-parameters.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-parameters.save_rate:]))
                train_info['reward'] = final_ep_rewards[-1]
                for i, rew in enumerate(agent_rewards):
                    final_ep_ag_rewards.append(np.mean(rew[-parameters.save_rate:]))
                    train_info['agent_%i/reward' % i] = final_ep_ag_rewards[-1]

            # 保存最后回合的奖励，以便以后绘制训练曲线。
            if len(episode_rewards) > parameters.train_num:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                if parameters.use_wandb: wandb_run.wandb_finish()
                break
            if parameters.use_wandb: wandb_run.wandb_log(train_info, len(episode_rewards))

if __name__ == '__main__':
    parameter = config.get_config()
    train(parameter)
