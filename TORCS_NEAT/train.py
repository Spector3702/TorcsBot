import neat
import pickle
import os
import glob
import argparse

from TORCS_NEAT.utils import eval_genome, load_config, compute_population

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, required=True, help="spceify how many generations to train.")
args = parser.parse_args()

FOLDER_NAME = 'TORCS_NEAT'


def eval_genomes(genomes, config):
    reward_path = f'./{FOLDER_NAME}/reward/'
    os.makedirs(reward_path, exist_ok=True)

    for genome_id, genome in genomes:
        eval_genome(genome, config)

        with open(f'{FOLDER_NAME}/reward/neat_rewards.txt', 'a') as output:
            output.write(str(genome.fitness) + '\n')


def train_neat(generations):
    checkpoint_path = f'./{FOLDER_NAME}/checkpoints/'
    checkpoint_file_prefix='neat-checkpoint-'

    population = compute_population(1, checkpoint_path, checkpoint_file_prefix)
    winner = population.run(eval_genomes, n=generations)

    with open(f'{FOLDER_NAME}/best_genome.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == '__main__':
    generations = args.generations
    train_neat(generations)
