import argparse
import tensorflow as tf
import os
import socket
import wandb

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
model_path = curr_path + '/models/'


def get_config():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="formation_v1", help="定义要使用 MPE 中的哪个环境")
    parser.add_argument("--algo_name", type=str, default="ma-ddpg", help="算法名称")
    parser.add_argument("--device", type=str, default="/gpu:0" if tf.test.is_gpu_available() else 'cpu', help="检测GPU")
    parser.add_argument("--train_num", type=int, default=100000, help="训练的回合数")

    parser.add_argument("--max_episode_len", type=int, default=240, help="每回合的步数")
    parser.add_argument("--adversaries_num", type=int, default=0, help="环境中的对手数量")
    parser.add_argument("--good_policy", type=str, default="maddpg", help="用于环境中“良好”（非对手）策略的算法")
    parser.add_argument("--adv_policy", type=str, default="maddpg", help="用于环境中对手策略的算法")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    parser.add_argument("--batch_size", type=int, default=1024, help="批量大小")
    parser.add_argument("--units_num", type=int, default=64, help="MLP 中的单元数")

    # Checkpointing
    parser.add_argument("--exp_name", type=str, default='maddpg_formation', help="实验名称，用作保存所有结果的文件名")
    parser.add_argument("--save_rate", type=int, default=10, help="每次完成此数量的训练时都会保存模型")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False,
                        help='恢复存储在load-dir（或save-dir如果未load-dir 提供）中的先前训练状态，并继续训练')
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help='在屏幕上显示存储在load-dir（或save-dir如果没有load-dir 提供）中的训练策略，但不继续训练')

    parser.add_argument("--noise_std_init", type=float, default=0.2, help="探索的高斯噪音标准")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="探索的高斯噪音标准")
    parser.add_argument("--noise_decay_steps", type=float, default=10000, help="在 noise_std 衰减到最小值之前有多少步")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    # wandb
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage]")
    # apf
    parser.add_argument("--use_apf", action='store_false', default=True, help="是否使用APF（人工势场算法）")
    parser.add_argument("--apf_noise", type=int, default=0.1, help="初始噪声值")
    parser.add_argument("--apf_decay_steps", type=int, default=10000, help="使用apf步数")

    args = parser.parse_args()
    args.use_wandb = True if not args.evaluate else False
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    args.apf_noise_decay = (args.apf_noise - 0) / args.apf_decay_steps

    args.save_dir = os.path.join(curr_path, f'models/{args.scenario}')
    args.model_files_num = len([file for file in os.listdir(args.save_dir)])
    args.load_dir = os.path.join(args.save_dir, f'{args.model_files_num}/')
    args.save_dir = os.path.join(args.save_dir, f'{args.model_files_num+1}/')
    # tf.device(args.device)
    return args


class MyWandb:
    def __init__(self, parameters):
        self.run = None
        self.parameters = parameters
        self.project_name = 'maddpg_formation'
        self.wandb_user_name = 'weixw99'
        self.pa_name = socket.gethostname()
        self.exp_name = "share_add_track"+"_train_num: " + str(parameters.train_num)
        self.group_name = parameters.scenario
        self.wandb_dir = str(parameters.save_dir)

    def wandb_init(self):
        self.run = wandb.init(config=self.parameters,
                              project=self.project_name,
                              entity=self.wandb_user_name,
                              notes=self.pa_name,
                              name=self.exp_name,
                              group=self.group_name,
                              dir=self.wandb_dir,
                              job_type="training",
                              reinit=True)

    def wandb_finish(self):
        self.run.finish()

    def wandb_log(self, train_info, episodes_num):
        for k, v in train_info.items():
            wandb.log({k: v}, step=episodes_num)
