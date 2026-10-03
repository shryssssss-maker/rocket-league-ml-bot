---
license: apache-2.0
pipeline_tag: reinforcement-learning
tags:
- rlgym
- rocket-league
- RLBot
- PPO
---

# CanoPy

CanoPy is a self-playing reinforcement learning Rocket League agent designed for the `RLBot Championship 2025`.  
It uses PPO (Proximal Policy Optimization) to learn 2v2 gameplay through self-play. The agent is trained to play effectively on both blue and orange teams and can generalize to various team compositions.

## Model Details

- **Framework:** RLGym + RLBot v5
- **Algorithm:** PPO (via `rlgym-ppo`)
- **Team size:** 2v2
- **Action repeat:** 8
- **Observations:** `DefaultObs` with normalized positions, angles, velocities, and boost
- **Action space:** Lookup table actions with repeat frames
- **Reward shaping:** Combined reward including:
  - Speed toward ball
  - In-air bonus
  - Ball velocity toward goal
  - Goal scoring reward

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6615494716917dfdc645c44e/1v9m5G8WSuJACQOs0AdDp.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6615494716917dfdc645c44e/WTXHjHXw1ZmmMvZEr_DI5.png)

## Training Configuration (from `config.json`)

- **Number of processes:** 4  
- **Minimum inference ratio:** 80%  
- **Steps per checkpoint:** 1,000,000  
- **PPO batch size:** 100,000  
- **PPO minibatch size:** 50,000  
- **PPO epochs per update:** 2  
- **Experience buffer size:** 300,000  
- **Policy network layers:** [256, 128]  
- **Critic network layers:** [256, 128]  
- **Policy learning rate:** 0.0001  
- **Critic learning rate:** 0.0001  
- **PPO entropy coefficient:** 0.01  
- **Standardize returns:** true  
- **Standardize observations:** false  
- **Total training steps:** 1,000,000,000  
- **Checkpoint directory:** ./checkpoints

## Intended Use

CanoPy is intended for research, competition, and experimentation within the RLBot framework. It is designed to compete in the ML bot bracket of the RLBot Championship 2025.  

## Limitations

- Performance is dependent on training; untrained or partially trained models may perform poorly.  
- The bot has been trained for standard Rocket League 2v2 matches; it may not generalize to unusual map sizes, mutators, or game modes.  
- Does not include human-like strategy beyond what PPO has learned from self-play.

## Evaluation

CanoPy can be evaluated using the `evaluate()` function in the training script. Expected evaluation includes average episode returns and gameplay against copies of itself.  
- **Note:** To meet RLBot Championship submission requirements, further testing against Psyonix Pro bots may be necessary.

## Contact / Author

- **Author:** FlameF0X /// Discord handler `@flame_f0x`
- **Competition:** RLBot Championship 2025
