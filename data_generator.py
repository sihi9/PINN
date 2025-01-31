import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
import io

def simulate_heat_diffusion(plate_size, steps, alpha, source_pos):
    temperature = np.zeros((plate_size, plate_size))
    temperature[source_pos] = 100  # Apply the heat source
    output = np.zeros((steps, plate_size, plate_size))
    output[0, :, :] = temperature

    for t in range(1, steps):
        new_temperature = temperature.copy()
        for i in range(1, plate_size-1):
            for j in range(1, plate_size-1):
                new_temperature[i, j] = temperature[i, j] + alpha *  (
                        temperature[i+1, j] +
                        temperature[i-1, j] + 
                        temperature[i, j+1] + 
                        temperature[i, j-1] - 
                        4 * temperature[i, j])
    
        new_temperature[0, :] = new_temperature[-1, :] = 0
        new_temperature[:, 0] = new_temperature[:, -1] = 0
        temperature = new_temperature
        output[t, :, :] = temperature
    
    return output

def generate_data(n_samples, alpha_min, alpha_max, plate_size=32, steps=100, train=True):
    data = []
    alphas = []
    source_positions = []
    for _ in range(n_samples):
        alpha = np.random.uniform(alpha_min, alpha_max)
        if train:
            source_pos = (np.random.randint(1, plate_size - 1), np.random.randint(1, plate_size // 2))
        else:
            source_pos = (np.random.randint(1, plate_size - 1), np.random.randint(1, plate_size - 1))
        sample_data = simulate_heat_diffusion(plate_size, steps, alpha, source_pos)
        data.append(sample_data)
        alphas.append(alpha)
        source_positions.append(source_pos)
    return np.array(data), np.array(alphas), np.array(source_positions)

def save_data(data, alphas, source_positions, filename):
    np.savez(filename, data=data, alphas=alphas, source_positions=source_positions)

def load_data(filename):
    with np.load(filename) as data:
        return data['data'], data['alphas'], data['source_positions']
    
def plot_random_samples(data, alphas, n_samples=5, times=[0, 25, 50, 75], save_path='plots/summary_plot.png'):
    # Create the directory if it doesn't exist
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Choose random samples
    indices = np.random.choice(range(data.shape[0]), min(n_samples, data.shape[0]), replace=False)
    fig, axes = plt.subplots(len(indices), len(times), figsize=(20, 5 * len(indices)), squeeze=False)

    for idx, data_idx in enumerate(indices):
        for i, time in enumerate(times):
            ax = axes[idx, i]
            ax.imshow(data[data_idx][time], cmap='hot', interpolation='nearest')
            if i == 0:
                ax.set_ylabel(f'α={alphas[data_idx]:.3f}', fontsize=12, rotation=0, labelpad=40, verticalalignment='center')
    
    # Set time steps as x-labels at the bottom of the figure
    for i, time in enumerate(times):
        axes[-1, i].set_xlabel(f'Time step {time}', fontsize=12)
        axes[-1, i].xaxis.set_label_position('bottom')

    # Adjust layout to prevent overlapping and ensure everything fits
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.15, right=0.95, hspace=0.4, wspace=0.1)
    plt.savefig(save_path)
    plt.close()


def plot_random_samples_as_gifs(data, alphas, n_samples=5, save_dir='gifs'):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Choose up to 5 random samples to plot
    indices = np.random.choice(range(data.shape[0]), min(n_samples, data.shape[0]), replace=False)

    for idx, sample_idx in enumerate(indices):
        images = []
        for time_step in range(data.shape[1]):  # Assuming data.shape[1] is the number of timesteps
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(data[sample_idx][time_step], cmap='hot', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'Time step {time_step}', fontsize=12)

            # Save the plot to a temporary buffer.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            image = imageio.imread(buf)
            images.append(image)
            buf.close()

        # Save the images as a GIF
        gif_path = os.path.join(save_dir, f'sample_{sample_idx}_alpha_{alphas[sample_idx]:.3f}.gif')
        imageio.mimsave(gif_path, images, duration=0.5)  # duration controls the time between frames in seconds


def add_gaussian_noise(data, mean=0, std_dev=1):
    noise = np.random.normal(mean, std_dev, data.shape)
    noisy_data = data + noise
    return noisy_data

def add_pepper_noise(data, square_size=2, pepper_prob=0.01):
    noisy_data = np.copy(data)
    num_samples, num_timesteps, height, width = data.shape

    for sample_index in range(num_samples):
        num_squares = int(pepper_prob * (height * width) / (square_size * square_size))
        for _ in range(num_squares):
            t = np.random.randint(0, num_timesteps)
            i = np.random.randint(0, height - square_size + 1)
            j = np.random.randint(0, width - square_size + 1)
            noisy_data[sample_index, :, i:i + square_size, j:j + square_size] = 0

    return noisy_data

def main():
    n_samples_train = 80  # Number of training samples
    n_samples_val = 20    # Number of validation samples
    alpha_min = 0.05
    alpha_max = 0.2

    # Generate training data (left half start points)
    train_data, train_alphas, train_source_positions = generate_data(n_samples_train, alpha_min, alpha_max, train=True)
    # Generate validation data (start points all over the plate)
    val_data, val_alphas, val_source_positions = generate_data(n_samples_val, alpha_min, alpha_max, train=False)

    # Save the training data
    train_filename = 'heat_diffusion_train_data_clean.npz'
    save_data(train_data, train_alphas, train_source_positions, train_filename)
    # Save the validation data
    val_filename = 'heat_diffusion_val_data_clean.npz'
    save_data(val_data, val_alphas, val_source_positions, val_filename)


    # Load and plot one sample
    loaded_data, loaded_alphas, loaded_source_positions = load_data(train_filename)
    plot_random_samples(loaded_data, loaded_alphas, save_path= "plots/clean_summary.png")  
    #plot_random_samples_as_gifs(loaded_data, loaded_alphas, n_samples=1, save_dir="gifs/clean")


    # add noise
    noisy_train = add_gaussian_noise(train_data, mean=0, std_dev=0.01)
    noisy_train = add_pepper_noise(noisy_train, pepper_prob=0.01)

    noisy_val = add_gaussian_noise(val_data, mean=0, std_dev=0.01)
    noisy_val = add_pepper_noise(noisy_val, pepper_prob=0.01)

    # Save the noisy training data
    noisy_train_filename = 'heat_diffusion_train_data_noisy.npz'
    save_data(noisy_train, train_alphas, train_source_positions, noisy_train_filename)
    # Save the noisy validation data
    noisy_val_filename = 'heat_diffusion_val_data_noisy.npz'
    save_data(noisy_val, val_alphas, val_source_positions, noisy_val_filename)


    # Load and plot one sample
    loaded_data, loaded_alphas, loaded_source_positions = load_data(noisy_train_filename)
    plot_random_samples(loaded_data, loaded_alphas, save_path= "plots/noisy_summary.png")  
    #plot_random_samples_as_gifs(loaded_data, loaded_alphas, n_samples=1, save_dir="gifs/noisy")


if __name__ == "__main__":
    main()
