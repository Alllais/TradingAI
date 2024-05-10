import tensorflow as tf

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
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        gaes = np.zeros_like(rewards)
        for t in reverse (range(len(deltas))):
            gaes[t] = deltas[t] + self.gamma * self.lambda_ * (1 - dones[t] * gaes[t+1])
        return gaes
    
    def learn (self , states, actions, rewards, next_states, dones):
        next_states = next_states.reshape([len(next_states), -1])
        _, next_values = self.model.predict(next_states)
        states = states.reshape([len(states), -1])
        _, values = self.model.predict(states)

        gaes = self.get_gae(rewards, values, next_values, dones)
        target_values = gaes + values

        with tf.GradientTape() as tape:
            probs, values = self.model(states)
            action_probs = tf.gather_nd(probs, actions)
            advantages = target_values - values
            actor_loss = -tf.math.log(action_probs) * advantages
            critic_loss = tf.keras.losses.MSE(values, target_values)
            total_loss = actor_loss + critic_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))