import matplotlib.pyplot as plt

FOLDER_NAME = 'TORCS_NEAT'


def read_rewards_from_file(filename):
    with open(filename, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]


def plot_average_reward_per_generation(rewards, pop_size=2):
    average_rewards = [sum(rewards[i:i+pop_size]) / pop_size for i in range(0, len(rewards), pop_size)]
    
    plt.figure(figsize=(10,6))
    plt.plot(average_rewards, marker='o', linestyle='-')
    plt.xlabel('Generations')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Generation')
    plt.grid(True)
    plt.show()


def plot_reward_per_genome(rewards):
    plt.figure(figsize=(10,6))
    plt.plot(rewards, marker='o', linestyle='-')
    plt.xlabel('Genomes')
    plt.ylabel('Reward')
    plt.title('Reward per Genome')
    plt.grid(True)
    plt.show()


def main():
    rewards = read_rewards_from_file(f'{FOLDER_NAME}/reward/neat_rewards.txt')
    plot_average_reward_per_generation(rewards)
    plot_reward_per_genome(rewards)


if __name__ == '__main__':
    main()
