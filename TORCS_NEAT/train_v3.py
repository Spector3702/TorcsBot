import neat
from torcs_env.gym_torcs import TorcsEnv
import pickle
import os
import random
import glob

FOLDER_NAME = 'TORCS_NEAT'

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

            state = next_state

        print(f"Genome: {genome.key}, Current Reward: {episode_reward:.2f}")

    return episode_reward

def train_neat(config_file, generations, num_episodes):
    checkpoint_path = './TORCS_NEAT/checkpoints/neat-checkpoint-'
    global last_gen

    # if run first time
    if (len(glob.glob(checkpoint_path + '*')) == 0):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)#將config_file中的參數加入config這個variable中
        population = neat.Population(config)
        last_gen = 0
        all_rewards = []
    # if continue training
    else:
        gen_num = []
        for file in glob.glob(checkpoint_path + '*'):
            gen_num.append(int(file[len(checkpoint_path):]))
        population = neat.Checkpointer.restore_checkpoint(checkpoint_path + str(max(gen_num)))
        last_gen = population.generation
        with open(f'{FOLDER_NAME}/reward/neat_rewards.pkl', 'rb') as f:
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

    
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(1, filename_prefix = checkpoint_path)) # gen_interval = 1

    winner = population.run(eval_genomes, n=generations)

    with open('best_genome.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

if __name__ == '__main__':
    generations = 100
    num_episodes = 1
    config_file = os.path.join(os.path.dirname(__file__), 'config_file.txt')
    train_neat(config_file, generations = generations, num_episodes = num_episodes)
