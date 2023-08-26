import neat
import argparse

from TORCS_NEAT.utils import eval_genome, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, required=True, help="spceify how many generations to train.")
args = parser.parse_args()


def eval_genomes(genomes, config):
    for _, genome in genomes:
        eval_genome(genome, config)


def train(generations):
    config = load_config('config_file.txt')
    p = neat.Population(config)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, generations)

if __name__ == '__main__':
    generations = args.generations
    train(generations)
