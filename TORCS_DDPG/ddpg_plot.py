import pickle
import matplotlib.pyplot as plt

FOLDER_NAME = 'TORCS_DDPG'
# Load rewards from the saved file
with open(f'{FOLDER_NAME}/ddpg_rewards.pkl', 'rb') as f:
    all_rewards = pickle.load(f)

plt.plot(all_rewards, marker='o', color='orange')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.grid(True)
plt.show()

