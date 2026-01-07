    import torch
from typing import List, Tuple, Union
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

class AdamATan2(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        eps: float = 1e-8 # Standard Adam epsilon
    ):
        # ... [Keep your validation logic same as before] ...
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps):
        for p in group["params"]:
            if p.grad is None: continue
            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps = [], [], [], [], []
            beta1, beta2 = group["betas"]
            
            self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps)
            if not params_with_grad: continue

            # 1. Increment step
            torch._foreach_add_(state_steps, 1)

            # 2. Apply Weight Decay (if any)
            if group["weight_decay"] != 0:
                # grad = grad + weight_decay * param
                torch._foreach_add_(grads, params_with_grad, alpha=group["weight_decay"])

            # 3. Update momentum (exp_avg)
            # m = beta1 * m + (1 - beta1) * grad
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            # 4. Update variance (exp_avg_sq)
            # v = beta2 * v + (1 - beta2) * grad^2
            torch._foreach_mul_(exp_avg_sqs, beta2)
            # We need grad^2 for the update
            grads_sq = torch._foreach_pow(grads, 2)
            torch._foreach_add_(exp_avg_sqs, grads_sq, alpha=1 - beta2)

            # 5. Bias Correction
            # We calculate the bias terms as scalars
            # Note: On TPU, it's often faster to do this per-group
            for i in range(len(params_with_grad)):
                step = state_steps[i]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # The "atan2" logic implementation
                # This assumes the kernel logic was: atan2(m_corr, sqrt(v_corr) + eps)
                m_corr = exp_avgs[i] / bias_correction1
                v_corr = exp_avg_sqs[i] / bias_correction2
                
                # Update: p = p - lr * atan2(m_corr, sqrt(v_corr) + eps)
                update_term = torch.atan2(m_corr, torch.sqrt(v_corr) + group["eps"])
                params_with_grad[i].add_(update_term, alpha=-group["lr"])
