import neat
import os
import numpy as np

from torcs_env.gym_torcs import TorcsEnv

def eval_genome(genome, config):
    genome.fitness = 4.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = TorcsEnv(path="torcs_env/quickrace.xml")
    ob = env.reset(relaunch= True, render=False, sampletrack=True)

    for _ in range(1000): 
        inputs = ob 
        action = net.activate(inputs)
        action = np.array(action)
        ob, reward, done, _ = env.step(action)

        genome.fitness += reward
        if done:
            break


def load_config(filename):
    config_path = os.path.join(os.path.dirname(__file__), filename)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    return config