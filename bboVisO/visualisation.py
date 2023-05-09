import numpy as np


def create_wavefront_balls(coords, radius, output_file):
    """
    Create a wavefront object (.obj) file with spheres at the specified coordinates.

    Args:
    coords (numpy.ndarray): An nx3 numpy array of coordinates.
    radius (float): The radius of the spheres.
    output_file (str): The name of the output file.
    """

    # Open the output file for writing
    with open(output_file, 'w') as f:

        # Define the sphere vertices and faces
        num_segments = 16
        sphere_vertices = []
        sphere_faces = []
        for j in range(num_segments):
            for k in range(num_segments):
                x = radius * np.sin(np.pi * j / num_segments) * np.cos(2 * np.pi * k / num_segments)
                y = radius * np.sin(np.pi * j / num_segments) * np.sin(2 * np.pi * k / num_segments)
                z = radius * np.cos(np.pi * j / num_segments)
                sphere_vertices.append([x, y, z])
                if j < num_segments - 1 and k < num_segments - 1:
                    sphere_faces.append(
                        [j * num_segments + k + 1, j * num_segments + k + 2, (j + 1) * num_segments + k + 2])
                    sphere_faces.append(
                        [j * num_segments + k + 1, (j + 1) * num_segments + k + 2, (j + 1) * num_segments + k + 1])

        # Iterate over each coordinate and write a sphere
        for i in range(coords.shape[0]):

            # Write the sphere's vertices
            center = coords[i, :]
            for j in range(num_segments):
                for k in range(num_segments):
                    x = center[0] + sphere_vertices[j * num_segments + k][0]
                    y = center[1] + sphere_vertices[j * num_segments + k][1]
                    z = center[2] + sphere_vertices[j * num_segments + k][2]
                    f.write('v {} {} {}\n'.format(x, y, z))

            # Write the sphere's faces
            sphere_offset = i * num_segments ** 2
            for face in sphere_faces:
                f.write(
                    'f {} {} {}\n'.format(face[0] + sphere_offset, face[1] + sphere_offset, face[2] + sphere_offset))


def create_wavefront_cylinders(start_points, end_points, radius, output_file):
    """
    Create a Wavefront OBJ file with cylinders connecting the given start and end points.

    Args:
    start_points (numpy.ndarray): An nx3 numpy array of coordinates representing the starting points.
    end_points (numpy.ndarray): An nx3 numpy array of coordinates representing the ending points.
    radius (float): The radius of the cylinders.
    output_file (str): The name of the output Wavefront OBJ file.
    """
    num_cylinders = len(start_points)

    with open(output_file, 'w') as f:
        vertex_count = 0
        for i in range(num_cylinders):
            start_point = start_points[i]
            end_point = end_points[i]
            vec = end_point - start_point
            vec_len = np.linalg.norm(vec)
            vec_normalized = vec / vec_len

            if vec_normalized[0] == 0:
                x_axis = np.array([1, 0, 0])
            else:
                x_axis = np.array([0, 1, 0])

            y_axis = np.cross(vec_normalized, x_axis)
            y_axis_normalized = y_axis / np.linalg.norm(y_axis)
            x_axis_normalized = np.cross(y_axis_normalized, vec_normalized)

            rotation_matrix = np.column_stack([x_axis_normalized, y_axis_normalized, vec_normalized])
            num_segments = 16
            angle_increment = 2 * np.pi / num_segments

            for j in range(num_segments):
                angle1 = j * angle_increment
                angle2 = (j + 1) * angle_increment

                vertex1 = start_point + radius * np.matmul(rotation_matrix,
                                                           np.array([np.cos(angle1), np.sin(angle1), 0]))
                vertex2 = start_point + radius * np.matmul(rotation_matrix,
                                                           np.array([np.cos(angle2), np.sin(angle2), 0]))
                vertex3 = end_point + radius * np.matmul(rotation_matrix, np.array([np.cos(angle2), np.sin(angle2), 0]))
                vertex4 = end_point + radius * np.matmul(rotation_matrix, np.array([np.cos(angle1), np.sin(angle1), 0]))

                f.write(f"v {vertex1[0]} {vertex1[1]} {vertex1[2]}\n")
                f.write(f"v {vertex2[0]} {vertex2[1]} {vertex2[2]}\n")
                f.write(f"v {vertex3[0]} {vertex3[1]} {vertex3[2]}\n")
                f.write(f"v {vertex4[0]} {vertex4[1]} {vertex4[2]}\n")

                f.write(f"f {vertex_count + 1} {vertex_count + 2} {vertex_count + 3}\n")
                f.write(f"f {vertex_count + 1} {vertex_count + 3} {vertex_count + 4}\n")

                vertex_count += 4

# def create_wavefront_cylinders(start_points, end_points, radius, output_file):
#     """
#     Create a Wavefront OBJ file with cylinders connecting the given start and end points.
#
#     Args:
#     start_points (numpy.ndarray): An nx3 numpy array of coordinates representing the starting points.
#     end_points (numpy.ndarray): An nx3 numpy array of coordinates representing the ending points.
#     radius (float): The radius of the cylinders.
#     output_file (str): The name of the output Wavefront OBJ file.
#     """
#     num_cylinders = len(start_points)
#
#     vertices = []
#     faces = []
#
#     for i in range(num_cylinders):
#         start_point = start_points[i]
#         end_point = end_points[i]
#         vec = end_point - start_point
#         vec_len = np.linalg.norm(vec)
#         vec_normalized = vec / vec_len
#
#         if vec_normalized[0] == 0:
#             x_axis = np.array([1, 0, 0])
#         else:
#             x_axis = np.array([0, 1, 0])
#
#         y_axis = np.cross(vec_normalized, x_axis)
#         y_axis_normalized = y_axis / np.linalg.norm(y_axis)
#         x_axis_normalized = np.cross(y_axis_normalized, vec_normalized)
#
#         rotation_matrix = np.column_stack([x_axis_normalized, y_axis_normalized, vec_normalized])
#         num_segments = 16
#         angle_increment = 2 * np.pi / num_segments
#
#         cylinder_vertices = []
#         for j in range(num_segments):
#             angle1 = j * angle_increment
#             angle2 = (j + 1) * angle_increment
#
#             vertex1 = start_point + radius * np.matmul(rotation_matrix, np.array([np.cos(angle1), np.sin(angle1), 0]))
#             vertex2 = start_point + radius * np.matmul(rotation_matrix, np.array([np.cos(angle2), np.sin(angle2), 0]))
#             vertex3 = end_point + radius * np.matmul(rotation_matrix, np.array([np.cos(angle2), np.sin(angle2), 0]))
#             vertex4 = end_point + radius * np.matmul(rotation_matrix, np.array([np.cos(angle1), np.sin(angle1), 0]))
#
#             cylinder_vertices.extend([vertex1, vertex2, vertex3, vertex4])
#             faces.extend([len(vertices) + 1, len(vertices) + 2, len(vertices) + 3])
#             faces.extend([len(vertices) + 1, len(vertices) + 3, len(vertices) + 4])
#
#             vertices.extend([vertex1, vertex2, vertex3, vertex4])
#
#         with open(output_file, 'w') as f:
#             for vertex in vertices:
#                 f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
#
#             for face in faces:
#                 f.write(f"f {face}\n")
