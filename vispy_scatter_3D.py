def scatter(data):
    from vispy import app, visuals, scene

    # build your visuals
    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)

    # The real-things : plot using scene
    # build canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True)

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 5

    # plot
    a1 = scene.visuals.XYZAxis(parent=view.scene)
    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(data)
    p1.symbol = visuals.marker_types[10]

    # run
    app.run()



