
smpl_connectivity_dict = [[0, 1], [0, 2], [0, 3], [3, 6], [6, 9], [9, 14], [9, 13], [9, 12], [12, 15],
                                      [14, 17], [17, 19], [19, 21], [13, 16], [16, 18], [18, 20]
                , [2, 5], [5, 8], [1, 4], [4, 7]]

def draw3Dpose(pose_3d, pose_3d2,pose_3d3,pose_3d4, ax, lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):  # blue, orange
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.grid(False)
    for i in smpl_connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=2, c="blue")
        x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x2, y2, z2, lw=2, c="red")
        x3, y3, z3 = [np.array([pose_3d3[i[0], j], pose_3d3[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x3, y3, z3, lw=2, c='black')
        x4, y4, z4 = [np.array([pose_3d4[i[0], j], pose_3d4[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x4, y4, z4, lw=2, c='grey')

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def draw3Dpose_only(pose_3d, pose_3d2,pose_3d3,  lcolor="#3498db", rcolor="#e74c3c",
               add_labels=False):  # blue, orange
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    for i in smpl_connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=2, c="blue")
        x2, y2, z2 = [np.array([pose_3d2[i[0], j], pose_3d2[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x2, y2, z2, lw=2, c="red")
        x3, y3, z3 = [np.array([pose_3d3[i[0], j], pose_3d3[i[1], j]]) for j in range(3)]
        # ax = fig.add_subplot(111, projection='3d')
        ax.plot(x3, y3, z3, lw=2, c='black')

    RADIUS = 1  # space around the subject
    xroot, yroot, zroot = pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def draw3Dpose_frames(rgb ,ti_p,data_key_rgb,data_key_tip):
    # 绘制连贯的骨架
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    i = 0
    j = 0
    while i < ti_p.shape[0]:
        draw3Dpose(ti_p[i],rgb[i], data_key_rgb[i],data_key_tip[i], ax)
        plt.pause(0.3)
        # print(ax.lines)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        # ax.lines = []
        i += 1
        if i == ti_p.shape[0]:
            # i=0
            j += 1
        if j == 2:
            break

    plt.ioff()
    plt.show()
