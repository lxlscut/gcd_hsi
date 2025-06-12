import os
import torch
import inspect

from datetime import datetime
from loguru import logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args, runner_name=None, exp_id=None):
    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        if args.exp_name is None:
            raise ValueError("Need to specify the experiment name")

            # 想要的时间格式示例: 2025-03-04_15-48-21.123
            # %Y：四位数年份；%m：两位数月份；%d：两位数日期
            # %H：24 小时制；%M：分钟；%S：秒；%f：微秒(6 位)，这里 [:-3] 截到毫秒
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]

        # 拼接到 exp_name 前面，整体形成一个唯一的文件夹名称
        now = f"{args.exp_name}_({timestamp_str})"
        log_dir = os.path.join(root_dir, 'log', now)

        # 如果该文件夹已经存在，就重新生成一个时间戳再试
        while os.path.exists(log_dir):
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            now = f"{args.exp_name}_({timestamp_str})"
            log_dir = os.path.join(root_dir, 'log', now)

    else:
        # 如果传入的 exp_id 已经存在，可以继续使用它(按你的逻辑来就好)
        log_dir = os.path.join(root_dir, 'log', f"{exp_id}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    print(runner_name)
    print(args)

    return args


class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples
