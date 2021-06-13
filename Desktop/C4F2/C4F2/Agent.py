import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential, Model , load_model
from keras.layers import Dense,Input
from keras.optimizers import Adam
import keras.backend as K
import tensorflow_probability as tfp



class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=64, fc2_dims=64,fc3_dims=32,fc4_dims=16):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.fc4 = Dense(self.fc4_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        value = self.fc3(value)
        value = self.fc4(value)
        pi = self.pi(value)

        return pi


class Agent:
    def __init__(self,lr=0.00005,gamma=0.99,n_actions=2,input_dims=9,fname='tmp.h5'):
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))
        self.policy=tf.saved_model.load('model/')
        

    def select_action(self, state,conn=None, vehicle_ids=None):
        if state[0] == 0:
            if (state[3]+state[4])+(2*(state[7]+state[8])) > 70:
                return 1
            if (state[3]+state[4])+(2*(state[7]+state[8]))< 10:
                return 0
        else:
            if (state[1]+state[2])+(2*(state[5]+state[6])) > 70:
                return 0
            if (state[1]+state[2])+(2*(state[5]+state[6])) < 10:
                return 1
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
    def save_model(self):
        self.policy.save(save_format="tf",filepath="model")