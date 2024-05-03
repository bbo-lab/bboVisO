from bboVisO.geometry import align_positions_orientations


def align_results_by_landmarks(landmarks_real, landmarks, positions, oris_flu):
    positions_aligned, orientations_aligned = align_positions_orientations(landmarks_real, landmarks,
                                                                           [landmarks, positions], [oris_flu])
    landmarks = positions_aligned[0]
    positions = positions_aligned[1]
    oris_flu = orientations_aligned[0]
    return landmarks, positions, oris_flu


def align_results_by_positions(positions_real, landmarks, positions, oris_flu):
    positions_aligned, orientations_aligned = align_positions_orientations(positions_real, positions,
                                                                           [landmarks, positions], [oris_flu])
    landmarks = positions_aligned[0]
    positions = positions_aligned[1]
    oris_flu = orientations_aligned[0]
    return landmarks, positions, oris_flu


def align_results_by_perches(landmarks_real, landmarks, positions, oris_flu, fix_indices=(0, 1),
                             dir_indices=None):
    if dir_indices is not None:
        corresp_idxs = fix_indices + dir_indices
        assert len(corresp_idxs) == len(landmarks_real), \
            "fix_indices + dir_indices must correspond to landmarks_real in order"
        landmarks_ref = landmarks[corresp_idxs, ...]
    else:
        landmarks_ref = landmarks

    positions_aligned, orientations_aligned = align_positions_orientations(landmarks_real, landmarks_ref,
                                                                           [landmarks, positions],
                                                                           [oris_flu],
                                                                           fix_indices=fix_indices)
    landmarks = positions_aligned[0]
    positions = positions_aligned[1]
    oris_flu = orientations_aligned[0]
    return landmarks, positions, oris_flu
