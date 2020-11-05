"""
Created on Wed 04 Nov 2020

@author: Przemyslaw Zielinski
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_reconstruction(writer, data, model, epoch,
                        coords=[0,1], coords_labels=['x','y']):

    dat_np = data.detach().numpy()
    rec_np = model(data).detach().numpy()

    fig, ax = plt.subplots()
    ax.scatter(dat_np.T[coords[0]], dat_np.T[coords[1]], label="data point")
    ax.scatter(rec_np.T[coords[0]], rec_np.T[coords[1]], label="reconstruction")
    ax.set_xlabel(f"{coords_labels[0]} coordinate")
    ax.set_ylabel(f"{coords_labels[1]} coordinate", rotation=90)
    ax.set_title(f"Reconstruction: epoch {epoch}")
    plt.legend()

    writer.add_figure(f'Reconstructions', fig, epoch)

def plot_slow_latent_correlation(writer, dataset, model, epoch):
    slow_map = dataset.system.slow_map

    data = dataset.data
    slow_var = slow_map(data.detach().numpy().T)
    lat_var = model.encoder(data).detach().numpy().T

    fig, ax = plt.subplots()
    ax.scatter(slow_var, lat_var)
    ax.set_xlabel('slow variable')
    ax.set_ylabel('latent variable', labelpad=0)
    ax.set_title(f"Performance: epoch {epoch}")
    fig.tight_layout()

    writer.add_figure(f'Latent performance', fig, epoch)

def write_logs(writer, log_msg, epoch):
    # msg = log_msg.split(', ',1)[1]
    writer.add_text('Logs', ', '.join(log_msg), epoch)
