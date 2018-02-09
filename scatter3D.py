def scatter_3d(data, color_data, position=None):
    from vispy import app, visuals, scene
    from vispy.color import Color, ColorArray
    from vispy import gloo
    import helper

    # build your visuals
    scatter = scene.visuals.create_visual_node(visuals.MarkersVisual)

    # scatter_window = QtWidgets.QMainWindow()

    # The real-things : plot using scene
    # build canvas
    canvas = scene.SceneCanvas(title='zbow 3D scatter plot',
                               keys='interactive',
                               show=True,
                               bgcolor=Color([0, 0, 0, 1]),
                               position=position)
                               # parent=scatter_window)  # defaults to black


    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 5
    view.camera.distance = 25
    view.camera.elevation = 30
    view.camera.azimuth = 130

    gloo.set_state()
    # plot
    scene.visuals.XYZAxis(parent=view.scene)

    p1 = scatter(parent=view.scene)
    p1.set_gl_state('translucent')
    # p1.set_gl_state(blend=False, depth_test=True)

    cell_color = ColorArray(color=color_data, alpha=1)
    # @BUG I want to use a different alpha here, but Vispy has a bug where you can see through the main canvas with alpha

    p1.set_data(pos=data,
                symbol='o',
                size=5,
                edge_width=0.1,
                edge_color=cell_color,
                face_color=cell_color)

    p1.symbol = visuals.marker_types[10]






