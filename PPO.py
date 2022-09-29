from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
from time import time
import gym



class PPOBuffer():
    def __init__(self) -> None:
        self.s = []
        self.a = []
        self.s_ = []
        self.r = []
        self.p = []
        self.d = []
        self.size = 0

    def clear(self) -> None:
        self.s = []
        self.a = []
        self.s_ = []
        self.r = []
        self.p = []
        self.d = []
        self.size = 0

    def append(self, state, action, reward, state_, probability, done) -> None:
        self.s.append(state)
        self.a.append(action)
        self.r.append(reward)
        self.s_.append(state_)
        self.p.append(probability)
        self.d.append(int(done))
        self.size += 1

    def recall(self):
        return np.array(self.s), np.array(self.a), np.array(self.r), np.array(self.s_), np.array(self.p), np.array(self.d)

    def get_batches(self, batch_size: int):
        assert self.size >= batch_size
        n = int(self.size / batch_size)
        indices = list(range(int(batch_size*n)))
        random.shuffle(indices)
        indices = np.array(indices).reshape(n, batch_size)
        return indices


class CriticContinuosNetwork(tf.keras.Model):
    def __init__(self, hidden_size: int) -> None:
        super(CriticContinuosNetwork, self).__init__()

        self.d1 = tf.keras.layers.Dense(hidden_size[0])
        self.d2 = tf.keras.layers.Dense(hidden_size[1])
        self.v = tf.keras.layers.Dense(1)

    def call(self, state, training=False):
        x = self.d1(state)
        x = tf.keras.activations.tanh(x)
        x = self.d2(x)
        x = tf.keras.activations.tanh(x)
        v = self.v(x)
        return v


class ActorContinuosNetwork(tf.keras.Model):
    def __init__(self, output_size, hidden_size: int, activation: str = 'relu') -> None:
        super(ActorContinuosNetwork, self).__init__()

        self.d1 = tf.keras.layers.Dense(hidden_size[0])
        self.d2 = tf.keras.layers.Dense(hidden_size[1])
        self.alpha = tf.keras.layers.Dense(output_size,
                                           activation=activation)
        self.beta = tf.keras.layers.Dense(output_size,
                                          activation=activation)

    def call(self, state, training=False):
        x = self.d1(state)
        x = tf.keras.activations.tanh(x)
        x = self.d2(x)
        x = tf.keras.activations.tanh(x)
        alpha = self.alpha(x) + 1
        beta = self.alpha(x) + 1
        # dist = tfp.distributions.Beta(alpha,
        #                               beta)
        # return dist
        return alpha, beta


class PPOContinuosAgent:
    '''Model free actor critic on-policy
    '''

    def __init__(
        self,
        alpha: float,  # 3e-4
        beta: float,  # 3e-4
        gamma: float,  # 0.99
        gae_lambda: float,  # 0.95
        clip_epsilon: float,  # 0.2
        entropy_coefficient: float,  # 0.2
        state_space,
        action_space,
        action_range: Tuple[float, float],
        actor_activation: str,
        buffer: PPOBuffer,
        checkpoint_dir: str,
        hidden_size: Tuple[float, float],
        env_name: str
    ):

        self.env_name = env_name
        self.alpha = alpha  # actor learning rate.
        self.beta = beta    # critic learning rate.
        self.gamma = gamma  # reward discount rate.
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coefficient = entropy_coefficient
        self.state_space = state_space
        self.action_space = action_space
        self.action_range = action_range
        self.actor_activation = actor_activation
        self.hidden_size = hidden_size
        self.buffer = buffer
        self.checkpoint_dir = checkpoint_dir
        self.epochs = 0

        self.actor_network = ActorContinuosNetwork(output_size=self.action_space.shape[0],
                                                   hidden_size=self.hidden_size[0],
                                                   activation=self.actor_activation)
        self.actor_network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha)
        )

        self.critic_network = CriticContinuosNetwork(
            hidden_size=self.hidden_size[1])
        self.critic_network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.beta)
        )

    def __getstate__(self) -> Dict:
        """This allows the class to be pickled
        No other piece of state should leak outside of these variables.
        """
        data = {}
        data["env_name"] = self.__dict__["env_name"]
        data["alpha"] = self.__dict__["alpha"]
        data["beta"] = self.__dict__["beta"]
        data["gamma"] = self.__dict__["gamma"]
        data["gae_lambda"] = self.__dict__["gae_lambda"]
        data["clip_epsilon"] = self.__dict__["clip_epsilon"]
        data["entropy_coefficient"] = self.__dict__["entropy_coefficient"]
        data["action_space"] = self.__dict__["action_space"]
        data["action_range"] = self.__dict__["action_range"]
        data["actor_activation"] = self.__dict__["actor_activation"]
        data["state_space"] = self.__dict__["state_space"]
        data["hidden_size"] = self.__dict__["hidden_size"]
        data["checkpoint_dir"] = self.__dict__["checkpoint_dir"]
        data["buffer"] = self.__dict__["buffer"]
        data["epochs"] = self.__dict__["epochs"]

        return data

    def __setstate__(self, state: dict) -> None:
        for k in iter(state):
            self.__setattr__(k, state[k])

    def save(self):
        path = self.checkpoint_dir
        joblib.dump(self, os.path.join(path, f"agent.pkl"))
        tf.keras.models.save_model(
            self.actor_network, os.path.join(path, f"actor_network"))
        tf.keras.models.save_model(
            self.critic_network, os.path.join(path, f"critic_network"))

    @staticmethod
    def load(path: str) -> PPOContinuosAgent:
        data = joblib.load(os.path.join(path, "agent.pkl"))
        assert isinstance(data, PPOContinuosAgent)
        data.actor_network = tf.keras.models.load_model(
            os.path.join(path, "actor_network"))
        data.critic_network = tf.keras.models.load_model(
            os.path.join(path, "critic_network"))
        return data

    def select_action(self, state, training=False) -> int:
        alpha, beta = self.actor_network(tf.convert_to_tensor([state], dtype=tf.float32),
                                          training)

       
        distribution = tfp.distributions.Beta(alpha, beta)
        sampled_actions = distribution.sample()

        sampled_actions_probabilities = distribution.log_prob(sampled_actions)
        
        actions = tf.squeeze(sampled_actions, axis=0)
        probabilities = tf.squeeze(sampled_actions_probabilities, axis=0)
        return actions.numpy(), probabilities.numpy()

    def calculate_advantages_and_returns(self, memories):
        s, r, s_, terminal = memories

        v = self.critic_network(tf.convert_to_tensor(s, dtype=tf.float32)).numpy()
        v_ = self.critic_network(tf.convert_to_tensor(s_, dtype=tf.float32)).numpy()

        v = v.flatten()
        v_ = v_.flatten()
    
        deltas = r + self.gamma * v_ - v

        advantages = [0]

        time_steps = reversed(list(range(len(r))))
        for i in time_steps:
            advantage = deltas[i] + self.gamma * \
                self.gae_lambda * advantages[-1] * (1-terminal[i])
            advantages.append(advantage)

        advantages = list(reversed(advantages))
        advantages = advantages[:-1]  # remove last zero.
        advantages = np.array(advantages)
        
        norm_advantages = (advantages - advantages.mean()) / advantages.std()
        norm_advantages = norm_advantages.reshape((-1, 1))
        # norm_advantages = tf.convert_to_tensor(norm_advantages, dtype=tf.float32)

        returns = advantages + v
        returns = returns.reshape((-1, 1))
        # returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        assert returns.shape == (len(r), 1)
        assert norm_advantages.shape == (len(r), 1)

        return norm_advantages, returns



    def learn(self, batch_size: int, epochs: int):

        states, actions, rewards, states_, _probs, dones = self.buffer.recall()

        advantages, returns = self.calculate_advantages_and_returns(( # shape(?, 1), shape(?, 1)
            states, rewards, states_, dones
        ))

        # states = tf.convert_to_tensor(states, dtype=tf.float32)   # shape(?, 3)
        # actions = tf.convert_to_tensor(actions, dtype=tf.float32) # shape(?, 1)
        # rewards = tf.convert_to_tensor(rewards, dtype=tf.float32) # shape(?, 1)
        # states_ = tf.convert_to_tensor(states_, dtype=tf.float32) # shape(?, 3)
        # _probs = tf.convert_to_tensor(_probs, dtype=tf.float32)   # shape(?, 1)
        # dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        for _ in range(epochs):
            batches = self.buffer.get_batches(batch_size)
            for batch in batches:   
                b_states = tf.convert_to_tensor(states[batch], dtype=tf.float32) # tf.gather(states, batch)
                b_actions = tf.convert_to_tensor(actions[batch], dtype=tf.float32) # tf.gather(actions, batch)
                b_probs = tf.convert_to_tensor(_probs[batch], dtype=tf.float32) #  tf.gather(_probs, batch)
                b_advantages = tf.convert_to_tensor(advantages[batch], dtype=tf.float32) # tf.gather(advantages, batch) # shape (batch_size, 1)
                b_returns = tf.convert_to_tensor(returns[batch], dtype=tf.float32)  # tf.gather(returns, batch) # shape (batch_size, 1)
                
                with tf.GradientTape() as tape:
                
                    a, b = self.actor_network(b_states)
                    distribution = tfp.distributions.Beta(a, b)
                    t0 = time()
                    probs = distribution.log_prob(b_actions) # shape (batch_size, n_actions)
                    print(f'TTP = {time()-t0}')

                    entropy = distribution.entropy()  # shape (batch_size, n_actions)
                    
                    entropy = tf.reduce_sum(entropy, axis=1, keepdims=True) # shape (batch_size, 1)
                    
                    # remember we are dealing with log of probabilities and
                    # multiplications and division becomes additions and subtractions.
                    a = tf.reduce_sum(probs, axis=1, keepdims=True) # shape (batch_size, 1)
                    b = tf.reduce_sum(b_probs, axis=1, keepdims=True) # shape (batch_size, 1)
                    prob_ratio = a - b
                    prob_ratio = tf.exp(prob_ratio)  # shape (batch_size, 1)

                    weighted_probs = b_advantages * prob_ratio # shape (batch_size, 1)
                    clipped_prob_ratio = tf.clip_by_value(prob_ratio,     # shape (batch_size, 1)
                                               -1-self.clip_epsilon, 
                                               1+self.clip_epsilon)
                    weighted_clipped_probs = b_advantages * clipped_prob_ratio  # shape (batch_size, 1)

                    actor_loss = tf.math.minimum(weighted_probs,    # shape (batch_size, 1)
                                                 weighted_clipped_probs)
                    actor_loss -= self.entropy_coefficient * entropy
                    actor_loss = tf.reduce_mean(actor_loss)

                    actor_gradients = tape.gradient(actor_loss,
                                                    self.actor_network.trainable_variables
                                                    )
                    # actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, 
                    #                                             1.0)
                    self.actor_network.optimizer.apply_gradients(
                        zip(actor_gradients, 
                            self.actor_network.trainable_variables)
                    )

                with tf.GradientTape() as tape:

                    v = self.critic_network(b_states) # shape (batch_size, 1)
                    critic_loss = (v-b_returns)**2    # shape (batch_size, 1)
                    critic_loss = tf.reduce_mean(critic_loss)
                    critic_gradients = tape.gradient(critic_loss,
                                                     self.critic_network.trainable_variables
                                                     )
                    # critic_gradients, _ = tf.clip_by_global_norm(critic_gradients,
                    #                                              1.0)
                    self.critic_network.optimizer.apply_gradients(
                        zip(critic_gradients,
                            self.critic_network.trainable_variables)
                    )
            
            self.epochs += 1
        
        return actor_loss.numpy(), critic_loss.numpy()


if __name__ == '__main__':

    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode=None)
    alpha = 0.0003  # 3e-4
    beta = 0.0003
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coefficient = 0.001
    action_range = (env.action_space.low, env.action_space.high)
    max_action_value = env.action_space.high[0]
    actor_activation = 'relu'
    hidden_size = ((128, 128), (128, 128))
    batch_size = 64
    N = 2048
    num_epochs = 10

    buffer = PPOBuffer()

    agent = PPOContinuosAgent(
        env_name=env_name,
        hidden_size=hidden_size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coefficient=entropy_coefficient,
        state_space=env.observation_space,
        action_space=env.action_space,
        action_range=action_range,
        actor_activation=actor_activation,
        buffer=buffer,
        checkpoint_dir=None
    )



    def action_adapter(action, max_action_value):
        return 2 * (action-0.5) * max_action_value


    def clip_reward(x):
        if x < -1:
            return -1
        elif x > 1:
            return 1
        else:
            return x


    episodic_return = []
    total_steps = 0
    max_steps = 1_000_000

    score_history = []



    i = 0
    while total_steps < max_steps:
        state, info = env.reset()
        rewards = []
        done = False
        score = 0
        TTS=[]
        while not done:
            # select action.
            t0=time()
            action, probability = agent.select_action(state, training=True)
            TTS.append(time()-t0)
            # take action.
            # we need to adapt the action to suit the environment specifications.
            env_action = action_adapter(action, max_action_value)
            next_state, reward, terminated, truncated, info = env.step(env_action)
            clipped_reward = clip_reward(reward)

            agent.buffer.append(state, action, clipped_reward,
                                next_state, probability, done)
            total_steps += 1
            rewards.append(reward)

            state = next_state
            score += reward

            if terminated or truncated:
                done = True
                # after each episode.
                episodic_return.append(np.sum(rewards))
                rolling_episodic_return = np.mean(episodic_return[-100:])
                

            if buffer.size == N:
                t0 = time()
                actor_loss, critic_loss = agent.learn(batch_size, num_epochs)
                print(f'TTL = {time()-t0}')
               
                buffer.clear()
        


        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        print(f'TTS {np.mean(TTS)}')
        print('Episode {} total steps {} avg score {:.1f}'.
                format(i, total_steps, avg_score))
        
        i += 1
            

    env = gym.make(env_name, render_mode='human')
    while True:
        state, info = env.reset()
        while True:
            # select action.
            action, probability = agent.select_action(state, training=True)
            # take action.
            # we need to adapt the action to suit the environment specifications.
            env_action = action_adapter(action, max_action_value)
            next_state, _, terminated, truncated, info = env.step(env_action)
            state = next_state
            if terminated or truncated:
                break
