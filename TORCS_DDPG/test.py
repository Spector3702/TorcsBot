import numpy as np
import torch
import argparse

import torcs_env.gym_torcs as gymt
from agent.ddpg import Ddpg
from agent.ActorNetwork import ActorNetwork
from agent.CriticNetwork import CriticNetwork

FOLDER_NAME = "TORCS_DDPG"

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, required=True, help="spceify which device to use.")
args = parser.parse_args()


def test(device):
    print("Entered test function")
    
    env = gymt.TorcsEnv(path="torcs_env/quickrace.xml")
    
    insize = env.observation_space.shape[0]
    outsize = env.action_space.shape[0]
    
    valuenet = CriticNetwork(insize, outsize)
    policynet = ActorNetwork(insize)
    agent = Ddpg(valuenet, policynet, buffersize=1)

    agent.load_state_dict(torch.load(f'{FOLDER_NAME}/best_agent_dict', map_location=torch.device(device)))
    agent.to(device)

    state = env.reset(relaunch=True, render=True)

    while True:
        torch_state = agent._totorch(state, torch.float32).view(1, -1)
        action, value = agent.act(torch_state)
        action =  action.to("cpu").squeeze()  # can use only cpu
        action.clamp_(-1, 1)
        action = np.concatenate([action[:2], [-1]])

        next_state, reward, done, _ = env.step(action)
        agent.push(state, action, reward, next_state, done)

        if done:
            print("DONE")
            break

        state = next_state


if __name__ == '__main__':
    device = args.device
    test(device)