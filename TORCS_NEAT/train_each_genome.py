import neat
import os
import numpy as np
import argparse
import pickle

from torcs_env.gym_torcs import TorcsEnv

FOLDER_NAME = 'TORCS_NEAT'

parser = argparse.ArgumentParser()
parser.add_argument("--genome", type=str, required=True, help="Path to the serialized genome file.")
args = parser.parse_args()


def eval_genomes(genome_id, genome, config):
    genome.fitness = 4.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = TorcsEnv(path="torcs_env/quickrace.xml")
    ob = env.reset(relaunch= True, render=False, sampletrack=True)

    for _ in range(1000): 
        inputs = ob 
        action = net.activate(inputs)
        action = np.array(action)
        ob, reward, done, _ = env.step(action)

        genome.fitness += reward
        if done:
            break

    os.makedirs(f'{FOLDER_NAME}/fitnesses', exist_ok=True)
    with open(f"{FOLDER_NAME}/fitnesses/fitness_{genome_id}.txt", 'w') as file:
        file.write(str(genome.fitness))


def train(genome_file):
    # Load configuration.
    config_path = os.path.join(os.path.dirname(__file__), 'config_file.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    genome_id, genome = load_genome(genome_file)

    # Evaluate the genome
    eval_genomes(genome_id, genome, config)


def load_genome(file_name):
    with open(file_name, 'rb') as file:
        data_loaded = pickle.load(file)
        
    genome_id = data_loaded['genome_id']
    genome = data_loaded['genome']
    
    return genome_id, genome


if __name__ == '__main__':
    genome_file = args.genome
    train(genome_file)
