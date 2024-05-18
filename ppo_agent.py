import tensorflow as tf
import numpy as np

class PPOAgent:
    def __init__ (self, model, action_size, gamma = 0.99, lambda_ = 0.95):
        self.model = model
        self.gamma = gamma
        self.lambda_ = lambda_
        self.action_size = action_size

    def act(self, state):
        state = state.reshape([1, -1])
        probs, _ = self.model.predict(state)
        action = np.random.choice(self.action_size, p=probs[0])
        return action

    def get_gae (self, rewards, values, next_values, dones):
        rewards = np.array(rewards, dtype=np.float32).flatten()
        values = np.array(values, dtype=np.float32).flatten()
        next_values = np.array(next_values, dtype=np.float32).flatten()
        dones = np.array(dones, dtype=np.float32).astype(int).flatten()

        deltas = rewards + self.gamma * next_values * (1-dones) - values
        gaes = np.zeros_like(rewards, dtype=np.float32)
        future_gae = 0.0
        for t in reversed (range(len(deltas))):
            gaes[t] = deltas[t] + self.gamma * self.lambda_ * future_gae * (1- dones[t])
            future_gae = gaes[t]
        return gaes
    
    def learn (self , states, actions, rewards, next_states, dones):
        actions = np.array(actions).reshape(-1,1).astype(np.int32)
        next_states = next_states.reshape([len(next_states), -1]).astype(np.float32)
        states = states.reshape([len(states), -1]).astype(np.float32)

        _, next_values = self.model.predict(next_states)
        _, values = self.model.predict(states)

        next_values = next_values.astype(np.float32).flatten()
        values = values.astype(np.float32).flatten()

        gaes = self.get_gae(rewards, values, next_values, dones)
        target_values = gaes + values

        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)

        with tf.GradientTape() as tape:
            probs, values = self.model(states)
            action_indices = tf.concat([tf.range(actions.shape[0])[:, tf.newaxis],actions], axis = -1)
            action_probs = tf.gather_nd(probs, action_indices)

            advantages = target_values - values
            advantages = tf.convert_to_tensor(target_values - values.numpy(), dtype=tf.float32)
            advantages = tf.reshape(advantages, [-1, 1])

            actor_loss = -tf.math.log(action_probs + 1e-8) * advantages
            actor_loss = tf.reduce_mean (actor_loss)

            critic_loss = tf.keras.losses.MSE(values, target_values)
            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))