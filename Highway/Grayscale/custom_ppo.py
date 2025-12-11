# custom_ppo.py
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3.ppo import PPO
from custom_rollout_buffer import CustomRolloutBuffer

class CustomPPO(PPO):
    """
    PPO with:
    - Custom reward shaping
    - Uses Custom Rollout Buffer
    """

    def _setup_model(self):
        super()._setup_model()
        # Replace rollout buffer with custom version
        self.rollout_buffer = CustomRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs
        )

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                # Get batch size
                batch_size = rollout_data.observations.shape[0]

                # Safe infos extraction
                infos = self.rollout_buffer.get_infos(range(batch_size))

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # value loss
                value_loss = F.mse_loss(rollout_data.returns, values)

                # entropy loss
                entropy_loss = -th.mean(entropy)

                # Reward shaping
                lane_change = th.tensor([i.get("is_lane_change", 0) for i in infos], device=self.device, dtype=th.float32)
                crashed = th.tensor([i.get("crashed", 0) for i in infos], device=self.device, dtype=th.float32)
                speed = th.tensor([i.get("speed", 0) for i in infos], device=self.device, dtype=th.float32)

                aux_loss = (0.5 * lane_change + 5.0 * crashed + -0.1 * (1 - speed / 30)).mean()

                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + aux_loss

                # Optimize
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        return loss.item()
