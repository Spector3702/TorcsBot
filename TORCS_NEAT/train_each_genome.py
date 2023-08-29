import os
import argparse
import pickle

from TORCS_NEAT.utils import eval_genome, load_config

FOLDER_NAME = 'TORCS_NEAT'

parser = argparse.ArgumentParser()
parser.add_argument("--genome", type=str, required=True, help="Path to the serialized genome file.")
args = parser.parse_args()


def load_genome(file_name):
    with open(file_name, 'rb') as file:
        data_loaded = pickle.load(file)
        
    genome_id = data_loaded['genome_id']
    genome = data_loaded['genome']
    
    return genome_id, genome


def eval_genomes_and_save_fitness(genome_id, genome, config):
    eval_genome(genome, config)

    fitness_path = f'./{FOLDER_NAME}/fitnesses'
    os.makedirs(fitness_path, exist_ok=True)
    with open(f'{fitness_path}/genome_{genome_id}_fitness.txt', 'a') as file:
        file.write(str(genome.fitness) + '\n')


def train(genome_file):
    config = load_config('config_file.txt')
    genome_id, genome = load_genome(genome_file)

    # Evaluate the genome
    eval_genomes_and_save_fitness(genome_id, genome, config)


if __name__ == '__main__':
    genome_file = args.genome
    train(genome_file)
