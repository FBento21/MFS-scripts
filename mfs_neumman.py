import numpy as np
import matplotlib.pyplot as plt
from shapes import circle_parameterization, polar_parameterization, boundary_polygons, create_source_points, create_inner_points, create_plot_points

def phi(x):
    return -np.log(np.linalg.norm(x, axis=2).T)/(2*np.pi)

def grad_phi(x, normals):
    x_norm = np.linalg.norm(x, axis=2).T**2
    dot_product = normals * x
    components = np.sum(dot_product, axis=2).T
    return -components * 1/ (x_norm * (2*np.pi))

def boundary_condition(x):
    #return np.linalg.norm(x, axis=1)
    return np.ones(x.shape[0])

def mfs_matrix(boundary, source, normals):
    subtraction = boundary.reshape(1, boundary.shape[0], 2) - source.reshape(source.shape[0], 1, 2)
    matrix = grad_phi(subtraction, normals)

    b_vector = boundary_condition(boundary)
    alphas = np.linalg.lstsq(matrix, b_vector, rcond=None)[0]

    return alphas

def plot(alphas, plot_points):
    subtraction = plot_points.reshape(1, plot_points.shape[0], 2) - source.reshape(source.shape[0], 1, 2)
    matrix = phi(subtraction)

    u_approx = matrix@alphas
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    contour = ax.tricontourf(plot_points[:, 0], plot_points[:, 1], u_approx)
    fig.colorbar(contour)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(plot_points[:, 0], plot_points[:, 1], u_approx, cmap='jet', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    boundary = boundary_polygons(4, 30, True)
    boundary = circle_parameterization(1, 30)
    source, normals = create_source_points(boundary, 0.2, 2)
    # inner = create_inner_points(20, boundary)
    plot_points = create_plot_points(100, boundary)

    plt.scatter(boundary[:, 0], boundary[:, 1], label='Boundary points')
    plt.scatter(source[:, 0], source[:, 1], label='Source points')
    # plt.scatter(inner[:, 0], inner[:, 1], label='Inner points')
    plt.legend()

    plt.show()

    alphas = mfs_matrix(boundary, source, normals)
    plot(alphas, plot_points)


