import os
import neat
import time
import pickle
import threading
import argparse

from TORCS_NEAT.utils import compute_population

FOLDER_NAME = 'TORCS_NEAT'
GEN = 0

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
    file_name = f"{FOLDER_NAME}/fitnesses/genome_{genome_id}_fitness.txt"

    while not os.path.exists(file_name):
        time.sleep(1)

    with open(file_name, 'r') as file:
        lines = file.readlines()

    while len(lines) != (GEN + 1):
        time.sleep(1)
        print(f'wait for writing fitness, current lines: {len(lines)}')
        with open(file_name, 'r') as file:
            lines = file.readlines()

    return float(lines[-1].strip())


def run_docker_container(genome_id, genome):
    genome_file = saving_genome(genome_id, genome)
    
    if os.path.exists('/.dockerenv'):
        host_path = os.environ['HOST_PATH']
    else:
        host_path = os.getcwd()
    
    # Mount the directory to ensure both genome and fitness files are accessible
    cmd = f'docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v {host_path}:/TorcsBot spector3702/neat-parallel:latest /bin/bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & python TORCS_NEAT/train_each_genome.py --genome {genome_file}"'
    os.system(cmd)
    
    fitness = retrieve_fitness(genome_id)
    genome.fitness = fitness


def train_genome_in_docker(genomes, config):
    global GEN
    threads = [] 

    for genome_id, genome in genomes:
        t = threading.Thread(target=run_docker_container, args=(genome_id, genome))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    GEN += 1
    

def main(generations):
    global GEN
    checkpoint_path = f'./{FOLDER_NAME}/checkpoints/'
    checkpoint_file_prefix='neat-checkpoint-'

    population, generation = compute_population(1, checkpoint_path, checkpoint_file_prefix)
    GEN = generation

    winner = population.run(train_genome_in_docker, generations)

    with open(f'{FOLDER_NAME}/best_genome.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == '__main__':
    generations = args.generations
    main(generations)