from torch.optim import lr_scheduler


class PolyLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-8):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


def option(optimizer, lr_policy_name, iters):
    if lr_policy_name == "poly":
        lr_policy = PolyLR(optimizer, iters, power=0.9)
    elif lr_policy_name == "step":
        lr_policy = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif lr_policy_name == "multi_step":
        lr_policy = lr_scheduler.MultiStepLR(optimizer, [30, 80], 0.1)
    elif lr_policy_name == "exponential":
        lr_policy = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif lr_policy_name == "cosine":
        lr_policy = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    elif lr_policy_name == "lambda":
        lr_policy = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iters: 0.95 ** iters)
    elif lr_policy_name == "onecycle":
        lr_policy = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=30, epochs=10)

    return lr_policy