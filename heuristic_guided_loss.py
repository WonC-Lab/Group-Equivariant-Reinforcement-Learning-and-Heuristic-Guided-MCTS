import torch
import torch.nn as nn
import torch.optim as optim

class HeuristicGuidedLoss(nn.Module):
    """
    Calculates a hybrid loss containing:
    1. Reinforcement Learning Loss (Policy Gradient)
    2. Heuristic Guide Loss (Kullback-Leibler Divergence)
    """
    def __init__(self, beta_start=1.0, beta_decay=0.99, beta_min=0.01):
        super().__init__()
        self.beta = beta_start
        self.beta_decay = beta_decay
        self.beta_min = beta_min
        
        # PyTorch KL Divergence Loss
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def decay_beta(self):
        """
        Decays the heuristic regularization coefficient after each training epoch/step.
        """
        self.beta = max(self.beta * self.beta_decay, self.beta_min)

    def forward(self, policy_logits, actions, rl_returns, heuristic_target_probs):
        """
        Parameters
        ----------
        policy_logits          : (Batch, NumActions) raw logits from neural net
        actions                : (Batch,) integer indexes of chosen actions
        rl_returns             : (Batch,) scalar rewards/returns (G_t)
        heuristic_target_probs : (Batch, NumActions) heuristic recommended probabilities (Must sum to 1.0)
        """
        # 1. Classical RL Policy Gradient Loss: -log_prob(a) * Return
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        selected_log_probs = log_probs[torch.arange(len(actions)), actions]
        
        # Negative sign since we perform gradient descent to maximize returns
        rl_loss = -torch.mean(selected_log_probs * rl_returns)

        # 2. Heuristic Guidance Loss via KL Divergence: KL( Heuristic || Agent )
        kl_loss = self.kl_loss_fn(log_probs, heuristic_target_probs)

        # 3. Hybrid Total Loss
        total_loss = rl_loss + self.beta * kl_loss

        return total_loss, rl_loss.item(), kl_loss.item()
