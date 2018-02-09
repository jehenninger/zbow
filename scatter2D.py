def scatter_2d(data, color_data, position=None):
    from matplotlib import pyplot as plt
    import ternary

    # # old way
    # scatter_window = plt.figure()
    #
    # scatter = plt.scatter(data[:, 0], data[:, 1],
    #                       s=2,
    #                       c=color_data,
    #                       marker='o',
    #                       alpha=1,
    #                       edgecolors='face',
    #                       figure=scatter_window)
    #
    # plt.show()

    # return scatter_window

    # new way with library
    # scale = 1.0
    # figure, tern_plot = ternary.figure(scale=scale)
    # tern_plot.set_title("ternary plot", fontsize=18)
    # tern_plot.boundary(linewidth=2.0)
    # tern_plot.gridlines(multiple=0.1, color='grey')
    #
    # tern_plot.scatter(data, marker='o', color=color_data)
    # tern_plot.ticks(axis='lbr', linewidth=1, multiple=0.1)
    #
    # tern_plot.show()

    # vispy method

    from vispy import app, visuals, scene
    from vispy.color import Color, ColorArray
    import helper
    from PyQt5 import QtWidgets

    # build your visuals
    scatter_handle = scene.visuals.create_visual_node(visuals.MarkersVisual)

    # scatter_window = QtWidgets.QMainWindow()

    # The real-things : plot using scene
    # build canvas
    canvas = scene.SceneCanvas(title='zbow 2D scatter plot',
                               keys='interactive',
                               show=True,
                               bgcolor=Color([1, 1, 1, 1]),
                               position=position)
    # parent=scatter_window)  # defaults to black


    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'


    p1 = scatter_handle(parent=view.scene)
    # p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_gl_state('translucent', blend=True, depth_test=False)

    cell_color = ColorArray(color=color_data, alpha=1)
    # @BUG I want to use a different alpha here, but Vispy has a bug where you can see through the main canvas with alpha

    p1.set_data(pos=data,
                symbol='o',
                size=5,
                edge_width=0.1,
                edge_color=cell_color,
                face_color=cell_color)

    p1.symbol = visuals.marker_types[10]


