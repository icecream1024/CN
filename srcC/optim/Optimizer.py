import torch
import torch.optim as optim
import math
class CLRAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, lr_base=1e-4, lr_max=1e-2, step_size=1400, gamma=0.99915, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_norm_constraint=0.9):
        defaults = dict(lr=lr, lr_base=lr_base, lr_max=lr_max, step_size=step_size, gamma=gamma, betas=betas, eps=eps, weight_decay=weight_decay, grad_norm_constraint=grad_norm_constraint)
        super(CLRAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq, prev_grad = state['exp_avg'], state['exp_avg_sq'], state['prev_grad']
                beta1, beta2 = group['betas']

                state['step'] += 1
# 首先，代码检查state字典是否为空，如果为空，则表示该参数的状态信息尚未初始化。在这种情况下，代码会为该参数初始化一些状态信息。
# - 'exp_avg': 用于计算参数的一阶矩估计（通常称为动量），初始值为与参数形状相同的全零张量。
# - 'exp_avg_sq': 用于计算参数的二阶矩估计，初始值为与参数形状相同的全零张量。
# - 'prev_grad': 用于存储上一次参数的梯度，初始值为与梯度形状相同的全零张量。
# 接下来，代码从state字典中获取参数的状态信息，并将其分别赋值给exp_avg、exp_avg_sq和prev_grad变量。
# 然后，代码从group字典中获取beta1和beta2的值，这些值通常用于计算动量和二阶矩估计的加权平均。

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Weight decay
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # CLR learning rate schedule
                cycle = math.floor(1 + state['step'] / (2 * group['step_size']))
                x = abs(state['step'] / group['step_size'] - 2 * cycle + 1)
                lr = group['lr_base'] + (group['lr_max'] - group['lr_base']) * max(0, 1 - x) * group['gamma'] ** state['step']
            # 这一行根据 CLR 的公式计算了当前的学习率。group['lr_base'] 和 group['lr_max'] 分别表示学习率的最小值和最大值，
            # group['gamma'] 表示学习率衰减因子。max(0, 1 - x) 是为了确保 x 在 0 到 1 之间，从而计算学习率的权重。
            # 最后，group['gamma'] ** state['step'] 是学习率的衰减部分，表示学习率随着训练步数的增加而衰减
                
                # Gradient norm constraint
                if group['grad_norm_constraint'] is not None:
                    grad_norm = torch.norm(grad)
                    if grad_norm > group['grad_norm_constraint']:
                        grad = grad * group['grad_norm_constraint'] / grad_norm

                # Step size
                denom = corrected_exp_avg_sq.sqrt().add_(group['eps'])

                # Step
                step_size = lr / bias_correction1
                p.data.addcdiv_(-step_size, corrected_exp_avg, denom)

                # Update previous gradient
                prev_grad.copy_(grad)

        return loss
