import torch
import math

step_size = 0
class ZCAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, min_lr=1e-5, max_lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=5, T_0=5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, min_lr=min_lr, max_lr=max_lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup, T_0=T_0)
        super(ZCAdam, self).__init__(params, defaults)

        # 将学习率设置为一个变量lr，并保存为实例变量self.lr
        self.lr = lr

    def __setstate__(self, state):
        super(ZCAdam, self).__setstate__(state)


    def step(self, closure=None):
        global step_size
        global scheduled_lr
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # 梯度值的指数移动平均数
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # 平方梯度值的指数移动平均数
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                # 计算一阶、二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                # 一阶、二阶参数偏置调整
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # warmup_lr = state['step'] / group['warmup'] * group['lr']
                # scheduled_lr = warmup_lr
                # if scheduled_lr < group['max_lr']:
                #     break
                # else:
                #     # 在step方法中，将当前学习率更新为余弦退火的结果，并将其作为优化器的学习率。
                #     cos_lr = group['lr'] + (group['max_lr'] - group['lr']) / 2 * (
                #                 1 + math.cos(math.pi * ((state['step']) - group['warmup']) / 100 / group['T_0']))
                #     scheduled_lr = cos_lr

                if state['step'] / 100 < group['warmup']:
                    warmup_lr = state['step'] / group['warmup'] * group['lr']
                    scheduled_lr = warmup_lr
                    if scheduled_lr > group['max_lr']:
                        break
                else:
                    # 在step方法中，将当前学习率更新为余弦退火的结果，并将其作为优化器的学习率。
                    cos_lr = group['lr'] + (group['max_lr'] - group['lr']) / 2 * (
                                        1 + math.cos(math.pi * ((state['step']) / 100 - group['warmup']) / group['T_0']))
                    scheduled_lr = cos_lr

                    # scheduled_lr = group['lr']
                # wandb.log({
                #     'lr': scheduled_lr,
                #     'step': state['step'],
                # })
                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)


                p.data.copy_(p_data_fp32)

        return loss
