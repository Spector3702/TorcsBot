import os
import neat
import time
import pickle

FOLDER_NAME = 'TORCS_NEAT'


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
    with open(file_name, 'r') as file:
        fitness = float(file.read().strip())
    return fitness


# Define a function to train a single genome in a Docker container
def train_genome_in_docker(genome_id, genome):
    genome_file = saving_genome(genome_id, genome)
    
    # Mount the directory to ensure both genome and fitness files are accessible
    cmd = f'docker run --rm -v {os.getcwd()}:/TorcsBot neat-parallel:latest /bin/bash -c "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 & python TORCS_NEAT/train_each_genome.py --genome {genome_file}"'
    os.system(cmd)
    
    fitness = retrieve_fitness(genome_id)
    
    return fitness


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config_file.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path) 
    pop = neat.Population(config)

    genomes = [(i, g) for i, g in enumerate(pop.population.values())]

    fitnesses = []
    for genome_id, genome in genomes:
        # For simplicity, we're running them sequentially here. In practice, you'd want to run them in parallel and monitor their progress.
        fitness = train_genome_in_docker(genome_id, genome)
        genome.fitness = fitness
        fitnesses.append(fitness)
        time.sleep(2)  # Optional: To ensure that containers don't start at the exact same time

    # Rest of the NEAT training loop (e.g., selecting the best genomes, breeding, etc.)


if __name__ == '__main__':
    main()