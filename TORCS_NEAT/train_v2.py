import numpy as np
import gym
from torcs_env.gym_torcs import TorcsEnv
import pickle
import json
import neat
import os
# Define the fitness function for NEAT
# Define the fitness function for NEAT
def eval_genomes_with_hyprm(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0
        episode_rewards = []
        for _ in range(hyprm['num_evaluation_steps']):
            state = env.reset(relaunch=True, render=False, sampletrack=True)
            episode_reward = 0
            for _ in range(hyprm['maxlength']):
                action = net.activate(state)
                action = np.concatenate([action[:2], [-1]])
                next_state, reward, done, _ = env.step(action)
                genome.fitness += reward
                episode_reward += reward
                if done:
                    break
                state = next_state
            genome.fitness = episode_reward
            episode_rewards.append(episode_reward)  # Store episode reward
            print(f"Genome {genome_id} - Episode reward: {episode_reward:.2f}")
            
def train_neat(config_file, num_generations):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT evolution for the specified number of generations
    winner = population.run(eval_genomes_with_hyprm, num_generations)

    # Save the best genome of each generation if needed
    for generation, best_genome in enumerate(stats.best_genomes, start=1):
        with open(f"best_genome_gen_{generation}.pkl", "wb") as f:
            pickle.dump(best_genome, f)

if __name__ == "__main__":
    env = TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")

    config_file = os.path.join(os.path.dirname(__file__), 'config_file.txt')  # Your NEAT configuration file
    num_generations = 100  # Replace with the number of generations you want to run
    hyprm = {'num_evaluation_steps': 1000, 'maxlength': 10000}  # Adjust these values based on your needs
    train_neat(config_file, num_generations)