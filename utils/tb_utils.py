"""
Created on Wed 04 Nov 2020

@author: Przemyslaw Zielinski
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_reconstruction(writer, trainer, epoch,
                        coords=[0,1], coords_labels=['x','y']):

    dat_t = trainer.dataset.data
    dat_np = dat_t.detach().numpy()
    rec_np = trainer.model(dat_t).detach().numpy()

    fig, ax = plt.subplots()
    ax.scatter(dat_np.T[coords[0]], dat_np.T[coords[1]], label="data point")
    ax.scatter(rec_np.T[coords[0]], rec_np.T[coords[1]], label="reconstruction")
    ax.set_xlabel(f"{coords_labels[0]}")
    ax.set_ylabel(f"{coords_labels[1]}", rotation=90)
    ax.set_title(f"Reconstruction: epoch {epoch}")
    plt.legend()

    writer.add_figure(f'Reconstructions', fig, epoch)

def plot_slow_latent_correlation(writer, trainer, epoch):
    slow_map = trainer.dataset.system.slow_map

    data = trainer.dataset.data
    slow_var = slow_map(data.detach().numpy().T)
    lat_var = trainer.model.encoder(data).detach().numpy().T

    fig, ax = plt.subplots()
    ax.scatter(slow_var, lat_var)
    ax.set_xlabel('slow variable')
    ax.set_ylabel('latent variable', labelpad=0)
    ax.set_title(f"Performance: epoch {epoch}")
    fig.tight_layout()

    writer.add_figure(f'Latent performance', fig, epoch)
