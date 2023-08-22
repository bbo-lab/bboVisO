import numpy as np
import os
from pathlib import Path
from calibcamlib.camerasystem import Camerasystem
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt


def create_wavefront_balls(coords, radius, output_file):
    """
    Create a wavefront object (.obj) file with spheres at the specified coordinates.

    Args:
    coords (numpy.ndarray): An nx3 numpy array of coordinates.
    radius (float): The radius of the spheres.
    output_file (str): The name of the output file.
    """

    os.makedirs(Path(output_file).parent.as_posix(), exist_ok=True)

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

    os.makedirs(Path(output_file).parent.as_posix(), exist_ok=True)

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


def plot_frames(analysis_frames, calibs, landmark_imcoords, landmarks, error_repro, offsets, oris, path, positions,
                readers, yml_path, opts):
    abserr = np.sqrt(np.sum(error_repro ** 2, axis=-1))
    aggr_error = np.nanmax(abserr, axis=-1)

    if opts["img_out_n"] == 0:
        img_out_n = len(analysis_frames)
    else:
        img_out_n = opts["img_out_n"]

    error_n = min(len(analysis_frames), img_out_n)
    error_frames = analysis_frames[np.argpartition(aggr_error, -error_n)[-error_n:]]

    for fr_idx in error_frames:
        plot_frame(yml_path.stem, path, readers, offsets, analysis_frames, calibs, landmark_imcoords,
                   landmarks, positions, oris, error_repro, fr_idx)


def plot_frame(name, path, readers, offsets, analysis_frames, calibs, landmark_imcoords, landmarks, positions,
               oris_flu, error_repro, fr_idx):
    cs = Camerasystem.from_calibs(calibs)

    a_idx = np.where(analysis_frames == fr_idx)[0][0]
    pos = positions[a_idx]
    ori_flu = R.from_rotvec(oris_flu[a_idx])

    dirs = ori_flu.inv().apply(landmarks - pos)
    imcoords_proj = cs.project(dirs)

    for cam_idx, (r, im_c_proj, im_c, o) in enumerate(zip(readers, imcoords_proj, landmark_imcoords, offsets)):
        plt.figure(dpi=600, figsize=(5, 3))
        cam_path = os.path.join(path, f"cam{cam_idx}")
        save_path = os.path.join(cam_path, name) + f"_{fr_idx}.png"
        try:
            im = r.get_data(fr_idx + o)
            plt.imshow(im)
            for i_lm, lm in enumerate(im_c[a_idx]):
                if not np.any(np.isnan(lm)) and not np.any(np.isnan(im_c_proj[i_lm])):
                    ph0 = plt.plot(*lm, "+", markersize=6, markeredgewidth=0.3)[0]
                    ph1 = plt.plot(*im_c_proj[i_lm], "x", markersize=6, markeredgewidth=0.3)[0]
                    ph1.set_color(ph0.get_color())
                    plt.plot(*np.vstack((lm, im_c_proj[i_lm])).T, "k--", markersize=1, markeredgewidth=0.1)
        except StopIteration:
            pass
        plt.title(f"Frame {fr_idx} ({fr_idx + o}), Cam {cam_idx}")
        # plt.show()

        if save_path is not None:
            os.makedirs(Path(save_path).parent.as_posix(), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
