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
    parser.add_argument("--device", type=str, default="cuda" if tf.test.is_gpu_available() else 'cpu', help="检测GPU")
    parser.add_argument("--train_num", type=int, default=70000, help="训练的回合数")

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
    parser.add_argument("--save_dir", type=str, default=model_path, help="保存中间训练结果和模型的目录")
    parser.add_argument("--save_rate", type=int, default=10, help="每次完成此数量的训练时都会保存模型")
    parser.add_argument("--load_dir", type=str, default="", help="从中加载训练状态和模型的目录")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False,
                        help='恢复存储在load-dir（或save-dir如果未load-dir 提供）中的先前训练状态，并继续训练')
    parser.add_argument("--display", action="store_true", default=False,
                        help='在屏幕上显示存储在load-dir（或save-dir如果没有load-dir 提供）中的训练策略，但不继续训练')
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help='对保存的策略运行基准评估，将结果保存到benchmark-dir文件夹')
    parser.add_argument("--benchmark_iter", type=int, default=100000, help="运行基准测试的迭代次数")
    parser.add_argument("--benchmark_dir", type=str, default="./benchmark_files/", help="保存基准数据的目录")
    parser.add_argument("--plots_dir", type=str, default="./learning_curves/", help="保存训练曲线的目录")
    # wandb
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage]")
    return parser.parse_args()


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
