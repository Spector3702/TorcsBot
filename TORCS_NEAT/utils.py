import neat
import os
import glob

from torcs_env.gym_torcs import TorcsEnv


def eval_genome(genome, config, max_time_steps=1000):
    env = TorcsEnv(path="torcs_env/quickrace.xml")
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    state = env.reset(relaunch=True, render=False, sampletrack=True)
    genome.fitness = 0.0
        
    for _ in range(max_time_steps):
        inputs = state
        action = net.activate(inputs)
        action = [float(a) for a in action]

        next_state, reward, done, _ = env.step(action)
        genome.fitness += reward

        if done:
            break

        state = next_state

    print(f"Genome: {genome.key}, Current Reward: {genome.fitness:.2f}")


def load_config(filename):
    config_path = os.path.join(os.path.dirname(__file__), filename)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    return config


def compute_population(interval, checkpoint_path, checkpoint_file_prefix):
    os.makedirs(checkpoint_path, exist_ok=True)
    
    if len(glob.glob(os.path.join(checkpoint_path, checkpoint_file_prefix + '*'))) == 0:
        config = load_config('config_file.txt')
        population = neat.Population(config)
    else:
        gen_num = []
        for file in glob.glob(os.path.join(checkpoint_path, checkpoint_file_prefix + '*')):
            gen_num.append(int(file[len(checkpoint_path + checkpoint_file_prefix):]))
        population = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_path, checkpoint_file_prefix + str(max(gen_num))))
    
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(interval, filename_prefix=checkpoint_path + checkpoint_file_prefix))

    return population