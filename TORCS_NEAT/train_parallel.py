import os
import neat
import time
import pickle
import threading
import argparse

from TORCS_NEAT.utils import load_config

FOLDER_NAME = 'TORCS_NEAT'

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, required=True, help="spceify how many generations to train.")
args = parser.parse_args()


def saving_genome(genome_id, genome):
    os.makedirs(f'{FOLDER_NAME}/genomes', exist_ok=True)
    file_name = f"{FOLDER_NAME}/genomes/genome_{genome_id}.pkl"

    data_to_save = {
        'genome_id': genome_id,
        'genome': genome
    }

    with open(file_name, 'wb') as file:
        pickle.dump(data_to_save, file)

    return file_name


def retrieve_fitness(genome_id):
    file_name = f"{FOLDER_NAME}/fitnesses/fitness_{genome_id}.txt"

    while not os.path.exists(file_name):
        time.sleep(1)  
    
    with open(file_name, 'r') as file:
        fitness = float(file.read().strip())

    return fitness


def run_docker_container(genome_id, genome):
    genome_file = saving_genome(genome_id, genome)
    
    # Mount the directory to ensure both genome and fitness files are accessible
    cmd = f'docker run --rm -v {os.getcwd()}:/TorcsBot spector3702/neat-parallel:latest /bin/bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & python TORCS_NEAT/train_each_genome.py --genome {genome_file}"'
    os.system(cmd)
    
    fitness = retrieve_fitness(genome_id)
    genome.fitness = fitness


def train_genome_in_docker(genomes, config):
    threads = [] 

    for genome_id, genome in genomes:
        t = threading.Thread(target=run_docker_container, args=(genome_id, genome))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()
    

def main(generations):
    config = load_config('config_file.txt') 
    pop = neat.Population(config)

    # Add a statistics reporter to gather and print information about the evolution
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(train_genome_in_docker, generations)

    # Print the details of the best genome
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    generations = args.generations
    main(generations)