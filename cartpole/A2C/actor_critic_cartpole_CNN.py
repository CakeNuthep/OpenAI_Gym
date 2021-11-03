import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v1")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

"""
## Implement Actor Critic network
This network learns two functions:
1. Actor: This takes as input the state of our environment and returns a
probability value for each action in its action space.
2. Critic: This takes as input the state of our environment and returns
an estimate of total rewards in the future.
In our implementation, they share the initial layer.
"""


num_actions = 2
num_hidden = 128

REM_STEP = 4
ROWS = 160
COLS = 240
image_memory = np.zeros((ROWS, COLS,REM_STEP))
state_size = (ROWS, COLS,REM_STEP)

inputs = layers.Input(shape=state_size)
net = layers.Conv2D(64, 5, strides=(3, 3),padding="valid", activation="relu", data_format="channels_last")(inputs)
net = layers.Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_last")(net)
net = layers.Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_last")(net)
# self.add(Conv2D(64, 5, strides=(3, 3),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_last"))
# self.add(Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_last"))
# self.add(Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_last"))
# self.add(Flatten(input_shape=input_shape))
# self.add(Dense(24,activation="relu",kernel_initializer='he_uniform',name="layer1"))
# self.add(Dense(24,activation="relu",kernel_initializer='he_uniform',name="layer2"))
# self.add(Dense(action_space, activation="softmax", kernel_initializer='he_uniform'))

net = layers.Flatten()(net)

common = layers.Dense(num_hidden, activation="relu")(net)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

"""
## Train
"""

optimizer = keras.optimizers.Adam(learning_rate=0.00000001)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0


def imshow(image, rem_step=0):
    cv2.imshow(str(rem_step), image[...,rem_step])
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        return

def GetImage():
    img = env.render(mode='rgb_array')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb_resized = cv2.resize(img_rgb, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    img_rgb_resized[img_rgb_resized < 255] = 0
    img_rgb_resized = img_rgb_resized / 255
    

    image_memory_roll = np.roll(image_memory, 1, axis = 2)
    image_memory[:,:,0] = img_rgb_resized
    image_memory[:,:,1] = image_memory_roll[:,:,1]
    image_memory[:,:,2] = image_memory_roll[:,:,2]
    image_memory[:,:,3] = image_memory_roll[:,:,3]
    
    # show image frame   
    #self.imshow(self.image_memory,0)
    
    return np.expand_dims(image_memory, axis=0)

def reset():
    env.reset()
    for i in range(REM_STEP):
        state = GetImage()
    return state

def step(action):
    next_state, reward, done, info = env.step(action)
    next_state = GetImage()
    return next_state, reward, done, info

while True:  # Run until solved
    state = reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            env.render(); #Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        # print('loss_value ',loss_value)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 480:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break