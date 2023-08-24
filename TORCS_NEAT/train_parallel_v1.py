import neat
import os
import numpy as np
from torcs_env.gym_torcs import TorcsEnv
from multiprocessing import Pool, Manager, current_process


# Genome evaluation function
def eval_genome(genome_id, genome, config, return_dict):
    worker_number = current_process()._identity[0] - 1  # Get the worker number
    genome.fitness = 0.0  # Initialize fitness to 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")
    ob = env.reset(relaunch=True, render=False, sampletrack=True)

    episode_reward = 0.0

    for _ in range(1000):  # You may want to adjust this based on episode length
        inputs = ob  # Assuming the observation is a 1D array
        action = net.activate(inputs)
        action = np.array(action)
        ob, reward, done, _ = env.step(action)

        episode_reward += reward
        if done:
            break

    genome.fitness = episode_reward
    return_dict[genome_id] = (worker_number, episode_reward)


def train_neat():
    # Load NEAT configuration from a config file
    config_path = os.path.join(os.path.dirname(__file__), 'config_file.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Set up multiprocessing pool for parallel genome evaluation
    num_workers = 2  # Change the number of workers to 2
    pool = Pool(processes=num_workers)
    manager = Manager()
    return_dict = manager.dict()

    # Create the population
    p = neat.Population(config)

    # Add a reporter to track progress
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run the training for a specified number of generations
    generations = 300
    for gen in range(generations):
        genomes = list(p.population.items())
        jobs = []

        for genome_id, genome in genomes:
            job = pool.apply_async(eval_genome, (genome_id, genome, config, return_dict))
            jobs.append(job)

        for job in jobs:
            job.get()

        for genome_id, genome in genomes:
            worker_number, genome_reward = return_dict[genome_id]
            genome.fitness = genome_reward

            # Output the worker number and reward for each genome
            print(f"Generation {gen}, Worker {worker_number}, Genome {genome_id}, Reward: {genome_reward}")

        # End of generation, call NEAT's logic
        p.reporters.end_generation(p.population, p.species, p.generation)

    # Display the best genome
    winner = stats.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the winner's network if needed
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('winner_net.txt', 'w') as f:
        f.write(str(winner_net))


if __name__ == '__main__':
    train_neat()
