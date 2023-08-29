import matplotlib.pyplot as plt
import os

FOLDER_NAME = 'TORCS_NEAT'


def read_fitness_from_file(filename):
    with open(filename, 'r') as file:
        return [float(line.strip()) for line in file.readlines()]


def plot_average_fitness_per_generation(all_fitnesses, pop_size=2):
    # Calculate the average fitness for each generation across all genomes
    average_fitnesses = [sum(fitness) / len(fitness) for fitness in zip(*all_fitnesses)]
    
    plt.figure(figsize=(10,6))
    plt.plot(average_fitnesses, marker='o', linestyle='-')
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness per Generation')
    plt.grid(True)
    plt.show()

def plot_fitness_per_genome(all_fitnesses):
    # Plotting each genome's fitness evolution across generations
    plt.figure(figsize=(10,6))
    for fitness in all_fitnesses:
        plt.plot(fitness, marker='o', linestyle='-')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness per Genome')
    plt.grid(True)
    plt.show()

def main():
    # Assuming all genome fitness files are in the FOLDER_NAME directory
    fitness_path = f'{FOLDER_NAME}/fitnesses'
    fitness_files = [f for f in os.listdir(fitness_path)]
    
    all_fitnesses = []
    for f_file in fitness_files:
        full_path = os.path.join(fitness_path, f_file)
        fitnesses = read_fitness_from_file(full_path)
        all_fitnesses.append(fitnesses)
        
    # Plotting
    plot_average_fitness_per_generation(all_fitnesses)
    plot_fitness_per_genome(all_fitnesses)



if __name__ == '__main__':
    main()
