'''Visualize Tokult results.
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sugayutils.figure import makefig

matplotlib.use('TkAgg')


##
def show_residuals(data: np.ndarray, model: np.ndarray) -> None:
    '''Visualize data, model, and residual cubes at each channel.'''
    residual = data - model
    vmax = np.max([data, model, residual])
    vmin = np.min([data, model, residual])
    shape = data.shape

    fig = makefig(figsize=['large', 0.3])
    axs = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.90)

    im0 = axs[0].imshow(data[0], vmin=vmin, vmax=vmax)
    im1 = axs[1].imshow(model[0], vmin=vmin, vmax=vmax)
    im2 = axs[2].imshow(residual[0], vmin=vmin, vmax=vmax)

    titles = ['Data', 'Model', 'Residual']
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.remove_xyticklabels()

    ax_channel = fig.add_axes([0.20, 0.08, 0.60, 0.07])
    ax_next = fig.add_axes([0.85, 0.05, 0.05, 0.10])
    ax_previous = fig.add_axes([0.10, 0.05, 0.05, 0.10])

    s_time = Slider(ax_channel, 'Channels', 0, shape[0] - 1, valinit=0, valstep=1.0)
    s_time.label.set(position=(0.4, 0.04), va='top', ha='center')
    s_time.valtext.set(position=(0.6, 0.04), va='top', ha='center')
    ax_next.remove_frame()
    button_next = Button(ax_next, '>', color='None')
    ax_previous.remove_frame()
    button_previous = Button(ax_previous, '<', color='None')

    def update(val):
        pos = int(s_time.val)
        # ax.axis([pos, pos + 10, 20, 40])
        im0.set_data(data[pos])
        im1.set_data(model[pos])
        im2.set_data(residual[pos])
        fig.canvas.draw_idle()

    def to_next(event):
        s_time.set_val(s_time.val + 1.0)
        pos = int(s_time.val)
        im0.set_data(data[pos])
        im1.set_data(model[pos])
        im2.set_data(residual[pos])
        # ax.axis([pos, pos + 10, 20, 40])
        fig.canvas.draw_idle()

    def to_previous(event):
        s_time.set_val(s_time.val - 1.0)
        pos = int(s_time.val)
        im0.set_data(data[pos])
        im1.set_data(model[pos])
        im2.set_data(residual[pos])
        # ax.axis([pos, pos + 10, 20, 40])
        fig.canvas.draw_idle()

    s_time.on_changed(update)
    button_next.on_clicked(to_next)
    button_previous.on_clicked(to_previous)

    # Tk.mainloop()
    plt.show()
