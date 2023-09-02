import pickle
import os
import argparse

from TORCS_NEAT.utils import eval_genome, compute_population

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, required=True, help="spceify how many generations to train.")
args = parser.parse_args()

FOLDER_NAME = 'TORCS_NEAT'


def eval_genomes(genomes, config):
    fitness_path = f'./{FOLDER_NAME}/fitnesses'
    os.makedirs(fitness_path, exist_ok=True)

    for idx, (_, genome) in enumerate(genomes):
        eval_genome(genome, config)

        with open(f'{fitness_path}/genome_{idx}_fitness.txt', 'a') as file:
            file.write(str(genome.fitness) + '\n')


def train_neat(generations):
    checkpoint_path = f'./{FOLDER_NAME}/checkpoints/'
    checkpoint_file_prefix='neat-checkpoint-'

    population, _ = compute_population(1, checkpoint_path, checkpoint_file_prefix)
    winner = population.run(eval_genomes, n=generations)

    with open(f'{FOLDER_NAME}/best_genome.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == '__main__':
    generations = args.generations
    train_neat(generations)
