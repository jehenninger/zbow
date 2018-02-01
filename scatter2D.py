def scatter_2d(data, color_data):
    from matplotlib import pyplot as plt

    scatter_window = plt.figure()

    scatter = plt.scatter(data[:, 0], data[:, 1],
                          s=2,
                          c=color_data,
                          marker='o',
                          alpha=1,
                          edgecolors='face',
                          figure=scatter_window)

    plt.show()

    return scatter_window
