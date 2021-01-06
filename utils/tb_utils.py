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
    ax.set_title(f"Epoch: {epoch}")
    plt.legend()

    writer.add_figure(f'Reconstruction_{coords_labels[0]}_vs_{coords_labels[1]})',
                      fig, epoch)

def plot_slow_latent_correlation(writer, trainer, epoch, coord=0):
    slow_map = trainer.dataset.system.slow_map

    data = trainer.dataset.data
    slow_var = slow_map(data.detach().numpy().T)
    lat_coord = trainer.model.encoder(data).detach().numpy().T[coord]

    sdim = len(slow_var)
    fig, axs = plt.subplots(ncols=sdim, sharey='row',
                            figsize=(sdim*6, 5))

    for n, (ax, slow_coord) in enumerate(zip(axs, slow_var)):
        ax.scatter(slow_coord, lat_coord)
        ax.set_xlabel(f'slow {n}')
        if n == 0:
            ax.set_ylabel(f'latent {coord}', labelpad=0)
        # ax.set_title(f"Performance: epoch {epoch}")
    fig.tight_layout()

    writer.add_figure(f'Latent performance (coord={coord})', fig, epoch)
