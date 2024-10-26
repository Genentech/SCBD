import matplotlib.pyplot as plt
import seaborn as sns


N_COLS = 8


sns.set_context(context="paper", font_scale=2)


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def next_color(ax):
    return next(ax._get_lines.prop_cycler)["color"]


def plot_image(ax, x):
    x = x.detach().cpu().numpy()
    if len(x.shape) == 3:
        x = x.transpose((1, 2, 0))
    ax.imshow(x, cmap="gray")