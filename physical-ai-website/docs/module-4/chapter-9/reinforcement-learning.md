---
sidebar_position: 1
---

# Reinforcement Learning for Physical AI and Humanoid Robotics

## Introduction

Reinforcement Learning (RL) represents a paradigm shift in how humanoid robots can learn to interact with their environment through trial and error. Unlike traditional control methods that rely on predefined models and controllers, RL enables robots to learn optimal behaviors through interaction with their environment. This chapter explores the application of RL techniques to physical AI and humanoid robotics, focusing on how these systems can learn complex motor skills, adapt to new environments, and develop embodied intelligence through experience.

## Foundations of Reinforcement Learning in Physical Systems

### The Embodied RL Framework

Reinforcement learning in physical systems differs significantly from traditional RL applications due to the constraints and opportunities presented by real-world physics:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym
from gym import spaces

class PhysicalEnvironment:
    """Environment for embodied reinforcement learning"""

    def __init__(self, robot_model, environment_config):
        self.robot = robot_model
        self.config = environment_config
        self.state_space = self._define_state_space()
        self.action_space = self._define_action_space()
        self.time_step = 0
        self.max_steps = 1000

    def _define_state_space(self):
        """Define the state space for the physical environment"""
        # State includes robot joint positions, velocities, and sensor readings
        state_dim = (
            self.robot.num_joints * 2 +  # positions and velocities
            self.robot.num_sensors +      # sensor readings
            3  # position in environment
        )
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )

    def _define_action_space(self):
        """Define the action space for the robot"""
        # Actions are joint torques or position targets
        action_dim = self.robot.num_joints
        return spaces.Box(
            low=-1.0, high=1.0,
            shape=(action_dim,), dtype=np.float32
        )

    def reset(self):
        """Reset the environment to initial state"""
        self.time_step = 0
        self.robot.reset()
        return self._get_state()

    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        # Apply action to robot
        self.robot.apply_action(action)

        # Simulate physics
        self.robot.simulate()

        # Get next state
        next_state = self._get_state()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        done = self._is_done()
        info = {}

        self.time_step += 1

        return next_state, reward, done, info

    def _get_state(self):
        """Get current state of the environment"""
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        sensor_readings = self.robot.get_sensor_readings()
        robot_position = self.robot.get_position()

        state = np.concatenate([
            joint_positions,
            joint_velocities,
            sensor_readings,
            robot_position
        ])

        return state.astype(np.float32)

    def _calculate_reward(self):
        """Calculate reward based on current state"""
        # Example reward function
        reward = 0.0

        # Encourage forward motion
        robot_velocity = self.robot.get_velocity()
        reward += robot_velocity[0] * 0.1  # Forward velocity

        # Penalize energy consumption
        torques = self.robot.get_applied_torques()
        energy_penalty = np.sum(np.abs(torques)) * 0.01
        reward -= energy_penalty

        # Penalize joint limits violations
        joint_positions = self.robot.get_joint_positions()
        joint_limits = self.robot.joint_limits
        for i, pos in enumerate(joint_positions):
            if pos < joint_limits[i][0] or pos > joint_limits[i][1]:
                reward -= 1.0  # Penalty for violating joint limits

        return reward

    def _is_done(self):
        """Check if episode is done"""
        if self.time_step >= self.max_steps:
            return True

        # Check for dangerous states
        robot_orientation = self.robot.get_orientation()
        if abs(robot_orientation[2]) > np.pi / 3:  # Too tilted
            return True

        return False
```

### Deep Reinforcement Learning Architectures

Deep RL architectures for humanoid robots must handle high-dimensional state and action spaces:

```python
class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for humanoid robot control"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are normalized to [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Action standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass through both networks"""
        action_mean = self.actor(state)
        value = self.critic(state)

        # Create distribution for action sampling
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)

        return dist, value

    def get_action(self, state):
        """Sample action from policy"""
        dist, value = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, value

    def evaluate_action(self, state, action):
        """Evaluate action for training"""
        dist, value = self.forward(state)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_prob, entropy, value

class PPOAgent:
    """Proximal Policy Optimization agent for humanoid robots"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 epsilon=0.2, epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs

    def update(self, states, actions, rewards, log_probs, values, dones):
        """Update the policy using PPO"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advantages = self._calculate_gae(rewards, values, dones)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy multiple times
        for _ in range(self.epochs):
            # Get new policy values
            new_log_probs, entropy, new_values = self.network.evaluate_action(states, actions)

            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)

            # Total loss
            total_loss = actor_loss + 0.5 * value_loss - 0.01 * entropy.mean()

            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

    def _calculate_gae(self, rewards, values, dones, tau=0.95):
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * tau * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages
```

## Advanced RL Techniques for Humanoid Robots

### Hierarchical Reinforcement Learning

Hierarchical RL decomposes complex humanoid tasks into manageable subtasks:

```python
class HierarchicalPolicy:
    """Hierarchical policy for complex humanoid behaviors"""

    def __init__(self, num_skills, state_dim, action_dim):
        self.num_skills = num_skills
        self.skill_selector = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_skills),
            nn.Softmax(dim=-1)
        )

        # Individual skill policies
        self.skill_policies = nn.ModuleList([
            ActorCriticNetwork(state_dim, action_dim)
            for _ in range(num_skills)
        ])

    def select_skill(self, state):
        """Select skill based on current state"""
        skill_probs = self.skill_selector(state)
        skill = torch.multinomial(skill_probs, 1).item()
        return skill, skill_probs[skill].item()

    def get_action(self, state, skill):
        """Get action for specific skill"""
        dist, value = self.skill_policies[skill](state)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value

class SkillDiscoveryNetwork(nn.Module):
    """Network for discovering and learning skills"""

    def __init__(self, state_dim, action_dim, max_skills=10):
        super().__init__()
        self.max_skills = max_skills

        # Option-critic architecture
        self.option_policy = nn.ModuleList([
            ActorCriticNetwork(state_dim, action_dim)
            for _ in range(max_skills)
        ])

        # Termination condition network
        self.termination = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_skills),
            nn.Sigmoid()
        )

        # Option-value network
        self.option_value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_skills)
        )

    def forward(self, state):
        """Forward pass for skill discovery"""
        option_values = self.option_value(state)
        termination_probs = self.termination(state)

        return option_values, termination_probs
```

### Multi-Agent Reinforcement Learning

Multi-agent systems enable coordination between multiple humanoid robots:

```python
class MultiAgentEnvironment:
    """Environment for multi-agent humanoid robotics"""

    def __init__(self, num_agents, robot_configs):
        self.num_agents = num_agents
        self.agents = [PhysicalEnvironment(config) for config in robot_configs]
        self.communication_range = 5.0  # meters

    def step(self, actions):
        """Execute actions for all agents"""
        next_states = []
        rewards = []
        dones = []
        infos = []

        # Execute individual actions
        for i, agent in enumerate(self.agents):
            action = actions[i]
            next_state, reward, done, info = agent.step(action)

            # Add communication-based rewards
            reward += self._calculate_communication_reward(i, next_state)

            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # Check for global termination
        global_done = all(dones)

        return next_states, rewards, [global_done] * self.num_agents, infos

    def _calculate_communication_reward(self, agent_idx, state):
        """Calculate reward based on communication with other agents"""
        agent_pos = state[-3:]  # Last 3 elements are position
        reward = 0.0

        # Encourage coordination
        for i, other_agent in enumerate(self.agents):
            if i != agent_idx:
                other_pos = other_agent._get_state()[-3:]
                distance = np.linalg.norm(agent_pos - other_pos)

                if distance < self.communication_range:
                    reward += 0.1  # Small reward for being in communication range

        return reward

class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient"""

    def __init__(self, agent_id, state_dim, action_dim, num_agents):
        self.agent_id = agent_id
        self.num_agents = num_agents

        # Actor network (local policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Critic network (centralized Q-function)
        # Takes states and actions of all agents
        critic_input_dim = state_dim * num_agents + action_dim * num_agents
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def get_action(self, state, add_noise=False):
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor)
        action = action.squeeze(0)

        if add_noise:
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1, 1)

        return action.detach().numpy()
```

## Learning from Demonstration

### Imitation Learning for Humanoid Robots

Imitation learning allows robots to learn from expert demonstrations:

```python
class BehaviorCloning:
    """Behavior cloning for humanoid robot skill learning"""

    def __init__(self, state_dim, action_dim):
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train(self, expert_states, expert_actions, epochs=100):
        """Train the policy to mimic expert behavior"""
        for epoch in range(epochs):
            states = torch.FloatTensor(expert_states)
            actions = torch.FloatTensor(expert_actions)

            predicted_actions = self.network(states)
            loss = self.criterion(predicted_actions, actions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_action(self, state):
        """Get action based on learned policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.network(state_tensor)
        return action.squeeze(0).detach().numpy()

class GenerativeAdversarialImitationLearning:
    """GAIL for learning from demonstrations"""

    def __init__(self, state_dim, action_dim):
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

    def update_discriminator(self, expert_states, expert_actions,
                           policy_states, policy_actions):
        """Update discriminator to distinguish expert from policy"""
        # Expert data (label = 1)
        expert_data = torch.cat([expert_states, expert_actions], dim=1)
        expert_labels = torch.ones(expert_states.size(0), 1)

        # Policy data (label = 0)
        policy_data = torch.cat([policy_states, policy_actions], dim=1)
        policy_labels = torch.zeros(policy_states.size(0), 1)

        # Concatenate all data
        all_data = torch.cat([expert_data, policy_data], dim=0)
        all_labels = torch.cat([expert_labels, policy_labels], dim=0)

        # Train discriminator
        disc_output = self.discriminator(all_data)
        disc_loss = F.binary_cross_entropy(disc_output, all_labels)

        self.discriminator_optimizer.zero_grad()
        disc_loss.backward()
        self.discriminator_optimizer.step()

    def update_policy(self, states, actions):
        """Update policy to fool discriminator"""
        # Get discriminator output for policy actions
        state_action_pairs = torch.cat([states, actions], dim=1)
        disc_output = self.discriminator(state_action_pairs)

        # Policy loss: maximize log probability of discriminator being fooled
        policy_loss = -torch.log(disc_output + 1e-8).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
```

## Transfer Learning and Domain Adaptation

### Sim-to-Real Transfer

Transferring learned behaviors from simulation to real robots:

```python
class DomainRandomization:
    """Domain randomization for sim-to-real transfer"""

    def __init__(self, base_env_config):
        self.base_config = base_env_config
        self.randomization_ranges = {
            'mass_range': [0.8, 1.2],  # Factor to multiply mass
            'friction_range': [0.5, 1.5],  # Factor for friction
            'gravity_range': [0.9, 1.1],  # Factor for gravity
            'sensor_noise_range': [0.0, 0.05],  # Range for sensor noise
        }

    def randomize_environment(self):
        """Randomize environment parameters"""
        randomized_config = self.base_config.copy()

        # Randomize physical parameters
        randomized_config['robot_mass'] *= np.random.uniform(
            *self.randomization_ranges['mass_range']
        )

        randomized_config['friction_coeff'] *= np.random.uniform(
            *self.randomization_ranges['friction_range']
        )

        randomized_config['gravity'] *= np.random.uniform(
            *self.randomization_ranges['gravity_range']
        )

        # Add sensor noise
        randomized_config['sensor_noise'] = np.random.uniform(
            *self.randomization_ranges['sensor_noise_range']
        )

        return randomized_config

class DomainAdaptationNetwork(nn.Module):
    """Network for domain adaptation"""

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # sim vs real
            nn.Softmax(dim=1)
        )

        # Task policy
        self.task_policy = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state, domain_adapt=False):
        """Forward pass with optional domain adaptation"""
        features = self.feature_extractor(state)

        if domain_adapt:
            # Domain adaptation mode
            domain_output = self.domain_classifier(features)
            policy_output = self.task_policy(features)
            return policy_output, domain_output
        else:
            # Normal mode
            policy_output = self.task_policy(features)
            return policy_output
```

## Safe Reinforcement Learning

### Safety-Constrained Learning

Ensuring safe exploration during learning:

```python
class SafeRLAgent:
    """Safe reinforcement learning agent"""

    def __init__(self, state_dim, action_dim, safety_constraints):
        self.safety_constraints = safety_constraints
        self.cbf_network = ControlBarrierFunction(state_dim, action_dim)
        self.rl_agent = PPOAgent(state_dim, action_dim)

    def safe_action(self, state):
        """Get safe action considering constraints"""
        # Get nominal action from RL policy
        nominal_action, log_prob, value = self.rl_agent.network.get_action(state)

        # Apply safety filter using Control Barrier Functions
        safe_action = self.cbf_network.filter_action(state, nominal_action)

        return safe_action, log_prob, value

class ControlBarrierFunction(nn.Module):
    """Control Barrier Function for safety"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Neural network to learn the barrier function
        self.barrier_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1] representing safety level
        )

    def forward(self, state):
        """Evaluate barrier function"""
        return self.barrier_net(state)

    def filter_action(self, state, nominal_action):
        """Filter action to ensure safety"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        nominal_action_tensor = torch.FloatTensor(nominal_action).unsqueeze(0)

        # Calculate barrier function value
        h = self.forward(state_tensor)

        # Calculate Lie derivatives (simplified)
        # In practice, these would be computed using automatic differentiation
        Lf_h = torch.zeros_like(h)  # Drift term
        Lg_h = torch.ones_like(nominal_action_tensor)  # Control term

        # Safety constraint: Lf_h + Lg_h * u >= -alpha * h
        alpha = 1.0  # Safety margin parameter
        safety_constraint = -Lf_h - alpha * h

        # Project action to satisfy safety constraint
        safe_action = nominal_action_tensor.clone()

        # Simplified projection - in practice this would be more complex
        constraint_violation = Lf_h + Lg_h * safe_action + alpha * h
        if constraint_violation < 0:
            # Adjust action to satisfy constraint
            adjustment = -constraint_violation / Lg_h
            safe_action = safe_action + adjustment

        # Clamp to action bounds
        safe_action = torch.clamp(safe_action, -1, 1)

        return safe_action.squeeze(0).detach().numpy()
```

## Practical Implementation Considerations

### Hardware-Aware RL

Accounting for real-world constraints in RL:

```python
class HardwareAwareEnvironment:
    """Environment that models hardware constraints"""

    def __init__(self, base_env, hardware_specs):
        self.base_env = base_env
        self.hardware_specs = hardware_specs

        # Initialize hardware models
        self.motor_model = MotorModel(hardware_specs['motors'])
        self.sensor_model = SensorModel(hardware_specs['sensors'])
        self.battery_model = BatteryModel(hardware_specs['battery'])

    def step(self, action):
        """Execute step with hardware modeling"""
        # Apply hardware constraints to action
        constrained_action = self.motor_model.apply_constraints(action)

        # Add sensor noise and delays
        noisy_state, reward, done, info = self.base_env.step(constrained_action)
        noisy_state = self.sensor_model.add_noise(noisy_state)

        # Update battery model
        energy_consumed = self.motor_model.get_energy_consumption(constrained_action)
        self.battery_model.consume_energy(energy_consumed)

        # Add battery level to state
        battery_state = np.array([self.battery_model.get_level()])
        final_state = np.concatenate([noisy_state, battery_state])

        # Add energy penalty to reward
        reward -= energy_consumed * 0.01

        return final_state, reward, done, info

class MotorModel:
    """Model of robot motors with realistic constraints"""

    def __init__(self, motor_specs):
        self.max_torque = motor_specs['max_torque']
        self.max_velocity = motor_specs['max_velocity']
        self.torque_constant = motor_specs['torque_constant']
        self.friction = motor_specs['friction']

    def apply_constraints(self, action):
        """Apply motor constraints to action"""
        # Scale action to torque limits
        constrained_action = np.clip(action, -1, 1)
        constrained_action = constrained_action * self.max_torque

        # Apply friction model
        constrained_action = np.sign(constrained_action) * np.maximum(
            0, np.abs(constrained_action) - self.friction
        )

        return constrained_action

    def get_energy_consumption(self, action):
        """Calculate energy consumption for action"""
        # Simplified energy model
        torque = np.abs(action)
        return np.sum(torque) * 0.001  # Simplified model

class SensorModel:
    """Model of robot sensors with noise and delays"""

    def __init__(self, sensor_specs):
        self.noise_std = sensor_specs['noise_std']
        self.delay_steps = sensor_specs.get('delay_steps', 0)
        self.delay_buffer = []

    def add_noise(self, state):
        """Add realistic sensor noise to state"""
        noise = np.random.normal(0, self.noise_std, state.shape)
        return state + noise
```

## Case Studies and Applications

### Learning Complex Motor Skills

Example of learning walking gaits using RL:

```python
class WalkingSkillLearning:
    """Learning to walk using reinforcement learning"""

    def __init__(self, robot_env):
        self.env = robot_env
        self.agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )

    def train_walking_policy(self, episodes=1000):
        """Train policy for walking"""
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                # Get action from policy
                action = self.agent.network.get_action(
                    torch.FloatTensor(state).unsqueeze(0)
                )[0].numpy()

                # Execute action
                next_state, reward, done, info = self.env.step(action)

                # Store experience
                self.agent.memory.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            # Update policy periodically
            if episode % 10 == 0:
                self.agent.update()

            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Steps = {episode_steps}")

    def evaluate_policy(self, num_episodes=10):
        """Evaluate trained policy"""
        total_reward = 0
        successful_walks = 0

        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            while steps < 500:  # Max 500 steps
                action = self.agent.network.get_action(
                    torch.FloatTensor(state).unsqueeze(0)
                )[0].numpy()

                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                steps += 1

                if done:
                    break

            total_reward += episode_reward

            # Check if robot moved forward successfully
            robot_pos = state[-3]  # Assuming x position is last element
            if robot_pos > 2.0:  # Moved at least 2 meters forward
                successful_walks += 1

        avg_reward = total_reward / num_episodes
        success_rate = successful_walks / num_episodes

        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.2f}")

        return avg_reward, success_rate
```

## Challenges and Future Directions

### Sample Efficiency

One of the main challenges in RL for physical systems is sample efficiency:

```python
class SampleEfficientRL:
    """Approaches for improving sample efficiency"""

    def __init__(self, env):
        self.env = env
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        self.auxiliary_tasks = []

    def add_auxiliary_task(self, task_fn, weight=0.1):
        """Add auxiliary task to improve learning"""
        self.auxiliary_tasks.append((task_fn, weight))

    def compute_auxiliary_rewards(self, state, action, next_state):
        """Compute rewards for auxiliary tasks"""
        aux_rewards = []

        for task_fn, weight in self.auxiliary_tasks:
            aux_reward = task_fn(state, action, next_state)
            aux_rewards.append(aux_reward * weight)

        return aux_rewards

class PrioritizedReplayBuffer:
    """Prioritized replay buffer for efficient learning"""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, priority=None):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)

        if priority is None:
            priority = 1.0  # Default priority

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = priority ** self.alpha

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch with prioritization"""
        if len(self.buffer) == 0:
            return [], [], [], [], [], []

        total = len(self.buffer)
        priorities = np.array(self.priorities[:total])
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(total, batch_size, p=probabilities)

        states, actions, rewards, next_states, dones, weights = [], [], [], [], [], []

        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Calculate importance sampling weights
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), weights)
```

## Key Takeaways

- Reinforcement learning enables humanoid robots to learn complex behaviors through interaction with the environment
- Deep RL architectures must handle high-dimensional state and action spaces typical in robotics
- Hierarchical RL decomposes complex tasks into manageable subtasks
- Safety considerations are crucial when applying RL to physical systems
- Domain randomization and adaptation techniques enable sim-to-real transfer
- Sample efficiency remains a significant challenge in physical RL
- Hardware-aware modeling improves the practical applicability of learned policies
- Imitation learning can accelerate the learning process by leveraging expert demonstrations

## Looking Forward

The next chapter will explore multi-agent humanoid systems, examining how multiple robots can coordinate and collaborate to achieve complex tasks. We'll look at distributed control, communication protocols, and emergent behaviors in multi-robot systems.