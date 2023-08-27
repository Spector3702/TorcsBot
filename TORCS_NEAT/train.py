import neat
<<<<<<< .merge_file_a08788
import pickle
import os
import random
import glob
=======
>>>>>>> .merge_file_a04004
import argparse

from TORCS_NEAT.utils import eval_genome, load_config

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, required=True, help="spceify how many generations to train.")
args = parser.parse_args()

FOLDER_NAME = 'TORCS_NEAT'

<<<<<<< .merge_file_a08788
def evaluate_individual(genome, config, num_episodes):
    env = TorcsEnv(path="torcs_env/quickrace.xml")
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    max_time_steps = 1000

    for episode in range(num_episodes):
        random_seed = episode
        random.seed(random_seed)
        state = env.reset(relaunch=True, render=False, sampletrack=True)
        episode_reward = 0
        
        for _ in range(max_time_steps):
            inputs = state
            action = net.activate(inputs)
            action = [float(a) for a in action]

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
=======
def eval_genomes(genomes, config):
    for _, genome in genomes:
        eval_genome(genome, config)
>>>>>>> .merge_file_a04004

            state = next_state

        print(f"Genome: {genome.key}, Current Reward: {episode_reward:.2f}")

    return episode_reward

def train_neat(config_file, generations, num_episodes):
    checkpoint_path = f'./{FOLDER_NAME}/checkpoints/'
    reward_path = f'./{FOLDER_NAME}/reward/'
    # Create the folder if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)  
    if not os.path.exists(reward_path):
        os.makedirs(reward_path)

    checkpoint_file_prefix = 'neat-checkpoint-'
    global last_gen

    if len(glob.glob(os.path.join(checkpoint_path, checkpoint_file_prefix + '*'))) == 0:
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        population = neat.Population(config)
        last_gen = 0
        all_rewards = []
    else:
        gen_num = []
        for file in glob.glob(os.path.join(checkpoint_path, checkpoint_file_prefix + '*')):
            gen_num.append(int(file[len(checkpoint_path + checkpoint_file_prefix):]))
        population = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_path, checkpoint_file_prefix + str(max(gen_num))))
        last_gen = population.generation
        with open(os.path.join(FOLDER_NAME, 'reward', 'neat_rewards.pkl'), 'rb') as f:
            all_rewards = pickle.load(f)

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            reward_list = []
            for _ in range(num_episodes):
                reward = evaluate_individual(genome, config, num_episodes)
                reward_list.append(reward)

            genome.fitness = max(reward_list)
            all_rewards.append(reward_list)

            with open(f'{FOLDER_NAME}/reward/neat_rewards.pkl', 'wb') as output:
                pickle.dump(all_rewards, output, protocol=pickle.HIGHEST_PROTOCOL)

<<<<<<< .merge_file_a08788
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(1, filename_prefix=checkpoint_path + checkpoint_file_prefix)) # gen_interval = 1

    winner = population.run(eval_genomes, n=generations)

    with open('best_genome.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)
=======
def train(generations):
    config = load_config('config_file.txt')
    p = neat.Population(config)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, generations)
>>>>>>> .merge_file_a04004

if __name__ == '__main__':
    generations = args.generations
    num_episodes = 1
    config_file = os.path.join(os.path.dirname(__file__), 'config_file.txt')
    train_neat(config_file, generations = generations, num_episodes = num_episodes)
