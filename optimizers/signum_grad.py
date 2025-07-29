import torch

class SignumGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, beta=0.9, weight_decay=0e-2, eps=1e-8):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("betas must be in [0,1)")
        defaults = dict(lr=lr, beta=beta,
                        weight_decay=weight_decay, eps=eps)
        
        self.__name__ = 'SignumGrad'
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1 = group["beta"]
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data

                if g.is_sparse:
                    raise RuntimeError("Ano does not support sparse gradients")
                
                # Get or initialize momentum
                state = self.state[p]
                # State initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    
                exp_avg = state["exp_avg"]
                
                state["step"] += 1
                t = state["step"]
                
                if t > 0:
                    # m_k
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)

                    update = lr * g.abs() * torch.sign(exp_avg)

                    # Decoupled weight-decay
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)
                        
                    # Update parameters with sign of momentum
                    p.add_(-update)
        return loss
