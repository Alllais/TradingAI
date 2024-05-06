import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_actor_model (state_size, action_size):
    inputs = Input(shape = (state_size,))
    layer1 = Dense(64, activation ='relu')(inputs)
    layer2 = Dense(64, activation ='relu')(layer1)
    outputs = Dense(action_size, activation ='softmax')(layer2)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 0.001))
    return model

def create_critic_model (state_size):
    inputs = Input(shape=(state_size,))
    layer1 = Dense(64, activation = 'relu')(inputs)
    layer2 = Dense(64, activation = 'relu')(layer1)
    outputs = Dense(1, activation = 'linear')(layer2)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'mse', optimizer = Adam(learning_rate=0.001))
    return model

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99

        self.actor = create_actor_model(state_size, action_size)
        self.critic = create_critic_model(state_size)

    def act(self, state):
        state =state.reshape([1, self.state_size])
        probabilities =self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p = probabilities)
        return action
    
    def learn (self, state, action, reward, next_state, done):
        state = state.reshape([1, self.state_size])
        next_state = next_state.reshape([1, self.state_size])

        critic_value_next = self.critic.predict(next_state)
        critic_value = self.critic.predict(state)

        target = reward + (1 - int(done)) * self.gamma * critic_value_next
        delta = target - critic_value

        actions = np.zeros([1, self.action_size])
        actions[np.arange(1), action] = 1

        self.actor.fit(state, actions * delta, verbose = 0)
        self.critic.fit(state, target, verbose = 0)