import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import pandas
from matplotlib.gridspec import GridSpec

# vmax = None

def display_sessions(session_values, file_name=None, title=None, cmap=None, losses=None, total_epochs=None, vmax=None):
    # global vmax
    sessions = np.array([np.array(v[:1000]) for v in session_values[:1000]])

    cm = plt.get_cmap('tab20c')
    cm_b = plt.get_cmap('tab20b')
    all_colors = cm.colors + cm_b.colors
    # mixed_colors = cm.colors + cm_b.colors
    # all_colors = []
    # for i in range(1, len(cm.colors)):
    #     all_colors.append(mixed_colors[i])
    #     all_colors.append(mixed_colors[-i])
     # = np.pad(sessions, (len(sessions), max_session_len), 'constant', constant_values=(0, 0))
    if losses:
        fig = plt.figure(constrained_layout=True, figsize=[15, 8])

        gs = GridSpec(2, 3, figure=fig)
        ax = fig.add_subplot(gs[:, :-1])
        ax2 = fig.add_subplot(gs[0, -1])
        # fig, (ax, ax2) = plt.subplots(1,2, figsize=[15, 8])
        ax2.plot(losses)
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epochs')
        # ax.set_aspect()
        # ax2.set_aspect()
        ax2.set_xticks(list(range(1, total_epochs+1, int(total_epochs/10))))
        ax2.set_yticks([0,0.5, 1])
        ax2.set_title('Loss-Verlauf')
        ax2.set_yscale('log')

    else:
        fig, ax = plt.subplots()
    # possible_log_keys = np.unique(sessions.flatten())
    # possible_colors = [cm(1. * i / NUM_COLORS) if for i in range(NUM_COLORS)]  # the label locations

    if not vmax and losses:
        vmax = sessions.max()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('Sessions')
    if losses:
        ax.set_xlabel('Abweichung von Eingabe')
    else:
        ax.set_xlabel('Log-Keys')

    if not title:
        ax.set_title('Log-Keys durch Farben repräsentiert je Session')
    else:
        ax.set_title(title)

    v_option = {}
    cbar_options = {'ticks': []}
    if vmax:
        v_option = {'vmin': 0, 'vmax': vmax}
        cbar_options = {'ticks': [0, vmax]}

    if cmap:
        c = ax.pcolormesh(sessions, cmap=cmap, **v_option)
    else:
        cMap = col.ListedColormap(all_colors[:36])
        c = ax.pcolormesh(sessions, cmap=cMap, **v_option)
    cbar = plt.colorbar(c, ax=ax, drawedges=True, extendrect=True, **cbar_options)
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label('Log-Key-IDs', rotation=270, labelpad=10)
    if file_name:
        plt.savefig(file_name + '.png')
        plt.close('all')
    else:
        plt.show()

    return vmax