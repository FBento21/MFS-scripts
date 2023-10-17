import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry import LinearRing

def circle_parameterization(r, n_boundary_points):
    theta = np.linspace(0, 2*np.pi, n_boundary_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    boundary = np.column_stack([x, y])
    return boundary

def polar_parameterization(parameterization, n_boundary_points):
    theta = np.linspace(0, 2*np.pi, n_boundary_points)
    r = parameterization(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    boundary = np.column_stack([x, y])
    return boundary

def boundary_polygons(n_vertices, n_boundary_points, coords_or_regular):
    if isinstance(coords_or_regular, np.ndarray):
        vertices = coords_or_regular
    elif coords_or_regular:
        angle = 2*np.pi/n_vertices
        vertices = []
        for i in range(n_vertices):
            x = np.cos(i*angle)
            y = np.sin(i*angle)
            vertices.append(np.array([x, y]))
        vertices = np.vstack(vertices)
    else:
        rng = np.random.default_rng()
        while True:
            vertices = rng.uniform(size=(n_vertices, 2))
            ls = LinearRing(vertices)
            if ls.is_simple and ls.is_ccw:
                print(ls.is_simple)
                break

    # Sectionate the boundary taking the norm of each segment into account
    modified_vertices = np.vstack([vertices, vertices[0]])
    lengths = []
    for i in range(1, modified_vertices.shape[0]):
        lengths.append(np.linalg.norm(modified_vertices[i] - modified_vertices[i-1]))
    perimeter = sum(lengths)
    # Sectionate each segment
    boundary = []
    for i in range(1, modified_vertices.shape[0]):
        segment_points = int(lengths[i-1] * n_boundary_points / perimeter)
        segment = np.linspace(modified_vertices[i-1], modified_vertices[i], segment_points)
        boundary.append(segment)
    boundary = np.vstack(boundary)

    return boundary

def create_source_points(boundary, eta, ratio_source):
    def normal(v1, v2):
        n = (v2 - v1)[::-1]
        n[0] = -n[0]
        return n

    normals = []
    modiefied_boundary_points = np.vstack([boundary[-1], boundary, boundary[0]])
    for i in range(1, modiefied_boundary_points.shape[0]-1):
        vi_m1 = modiefied_boundary_points[i-1]
        vi = modiefied_boundary_points[i]
        vi_p1 = modiefied_boundary_points[i+1]
        u_norm = (normal(vi_m1, vi) + normal(vi, vi_p1))/2
        norm = u_norm/np.linalg.norm(u_norm)
        normals.append(norm)
    normals = np.stack(normals)
    source = boundary - eta*normals
    source = source[::ratio_source]

    return source, -normals

def create_inner_points(n_inner_points,boundary, grid=True):
    if grid:
        # Create a lattice
        x_min = min(boundary[:, 0])
        x_max = max(boundary[:, 0])

        y_min = min(boundary[:, 1])
        y_max = max(boundary[:, 1])

        x_inner = np.linspace(x_min, x_max, n_inner_points)
        y_inner = np.linspace(y_min, y_max, n_inner_points)

        xx_inner, yy_inner = np.meshgrid(x_inner, y_inner)
        points_inner = np.stack([xx_inner, yy_inner], axis=2)
        points_inner = np.concatenate(points_inner, axis=0)
    else:
        pass

    polygon = Path(boundary)

    bool_points_inner = polygon.contains_points(points_inner)
    points_inner = points_inner[bool_points_inner]

    return points_inner

def create_plot_points(n_plot_points,boundary, grid=True):
    if grid:
        # Create a lattice
        x_min = min(boundary[:, 0])
        x_max = max(boundary[:, 0])

        y_min = min(boundary[:, 1])
        y_max = max(boundary[:, 1])

        x_plot = np.linspace(x_min, x_max, n_plot_points)
        y_plot = np.linspace(y_min, y_max, n_plot_points)

        xx_plot, yy_plot = np.meshgrid(x_plot, y_plot)
        points_plot = np.stack([xx_plot, yy_plot], axis=2)
        points_plot = np.concatenate(points_plot, axis=0)
    else:
        pass

    polygon = Path(boundary)

    bool_points_plot = polygon.contains_points(points_plot)
    points_plot = points_plot[bool_points_plot]

    return points_plot


if __name__ == '__main__':
    boundary = boundary_polygons(4, 200, True)
    source_points, normals = create_source_points(boundary, 0.1, 2)
    points_inner = create_inner_points(20, boundary)
    points_plot = create_plot_points(50, boundary)

    plt.scatter(boundary[:, 0], boundary[:, 1])
    plt.scatter(source_points[:, 0], source_points[:, 1])
    plt.scatter(points_inner[:, 0], points_inner[:, 1])
    plt.scatter(points_plot[:, 0], points_plot[:, 1])


    plt.show()
