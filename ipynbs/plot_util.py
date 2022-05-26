import matplotlib
#matplotlib.rcParams['font.size'] = 18.0
import matplotlib.pyplot as plt
#from matplotlib import figure

def draw_per_task_values(
        ax,
        values,
        norm,
        title,
        np_target_task_vecs,
        size_base=6.0,
        cmap=plt.cm.viridis,
        label_values=True,
        label_format='{:.3f}',
        task_vecs_set_to_mark_before=[],
        task_vecs_set_to_mark=[],
        draw_own_colorbar=False,
    ):
    for task_vecs, kwargs in task_vecs_set_to_mark_before:
        ax.scatter(
            x=task_vecs[:, 0],
            y=task_vecs[:, 1],
            **dict(dict(marker='x'), **kwargs),
        )
    if norm is None:
        norm = matplotlib.colors.Normalize(
            vmin=values.min(),
            vmax=values.max(),
        )
    plot_scatter = ax.scatter(
        x=np_target_task_vecs[:, 0],
        y=np_target_task_vecs[:, 1],
        c=values,
        cmap=cmap,
        norm=norm,
        s=size_base ** 2,
    )
    if label_values:
        for x, y, c in zip(np_target_task_vecs[:, 0], np_target_task_vecs[:, 1], values):
            ax.text(x=x+0.02, y=y+0.02, s=label_format.format(c))
    #if mark_source_tasks:
    #    ax.scatter(
    #        x=np_source_task_vecs[:, 0],
    #        y=np_source_task_vecs[:, 1],
    #        c='#444444',
    #        marker='x',
    #    )
    for task_vecs, kwargs in task_vecs_set_to_mark:
        ax.scatter(
            x=task_vecs[:, 0],
            y=task_vecs[:, 1],
            **dict(dict(marker='x'), **kwargs),
        )
    ax.set_title(title)
    if draw_own_colorbar:
        plt.colorbar(plot_scatter, ax=ax)
    

