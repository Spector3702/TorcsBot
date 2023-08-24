import neat
import os
import numpy as np
import argparse

from torcs_env.gym_torcs import TorcsEnv

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, required=True, help="spceify how many generations to train.")
args = parser.parse_args()


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        env = TorcsEnv(path="torcs_env/quickrace.xml")
        ob = env.reset(relaunch= True, render=False, sampletrack=True)

        for _ in range(1000):  # You may want to change this depending on the maximum length of an episode
            inputs = ob  # Assuming the observation is a 1D array
            action = net.activate(inputs)
            action = np.array(action)
            ob, reward, done, _ = env.step(action)

            genome.fitness += reward
            if done:
                break


def train(generations):
    # Load configuration.
    config_path = os.path.join(os.path.dirname(__file__), 'config_file.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 1)

if __name__ == '__main__':
    generations = args.generations
    train(generations)
