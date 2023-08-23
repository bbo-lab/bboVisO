# bboVisO
Simple visual odometry for omnidirectional cameras

## Running
Run by supplying a config file:

    python -m bboVisO \
        --from_yml [path to config file] \
        --text_out [directory for output of text files]

and optionally

        --img_out [path for output of reprojected images, generation is slow]
        --img_out_n [number of reprojected images, starting with the one with highest max reprojection error]
and

        --check_mode  # Interpolated frames with high error, helps to identify labeleing errors in combination with img_out

## Output
In the directory of `--text_out` text files will be generated:

- `..._cam_orientations.txt` holds the camera system orientations in `fr_id X Y Z W` quaternion format
- `..._cam_positions.txt` holds the camera system positions in `fr_id X Y Z` format
- `..._landmarks.txt` holds the landmarks positions in `name X Y Z` format
- `..._error.txt` holds the reprojection erros in `fr_idx lm_1_error ... lm_n_error` format (nan is 0!)
- `..._maxerror.txt` holds the reprojection erros in `fr_idx nanmax(lm_1_error, ..., lm_n_error)` format
- `..._meannerror.txt` holds the reprojection erros in `fr_idx nanmean(lm_1_error, ..., lm_n_error)` format

## Config file

    calibration: [path to camera calibration file from bbo_calibcam],
    cam_offsets: [list of frame offsets, in the order of the calibration file],
    acm_labels: [path to labels from acm_traingui, in the order of the calibration file],
    landmark_coordinates: 
        landmarks:
            landmark_1: 
                position: [317857.32, 5597789.46, 531.46]
                optimize: false  # bboVisO will not optimize the coordinates
                far: false  # Feature is not far away, i.e. trajectory may pass it
                use: true  # Feature should be used
            landmark_2: 
                position: [317879.82, 5597769.70, 529.71]
                optimize: true
                far: false
                use: true
                comment: "Was labeled, but landmark coordinates are inaccurate. Will be optimized by bboVisO"
            landmark_3: 
                position: []
                optimize: true
                far: true
                use: true
                comment: "Was labeled, but landmark coordinates are unknown. Will be found by bboVisO"
            landmark_4: 
                position: [317882.84, 5597767.28, 528.22]
                optimize: true
                far: true
                use: false
                comment: "Could not be labeled and is not used"
            ...
        origin:
            position: [317840.81, 5597868.59, 535.61]
            info: "Will be subtracted from all landmarks before processing"
    videos: [optional, for visualization list of video paths, in the order of the calibration file]

Internal note: Instead of the above format, with the additional switch --tetherless, the battery_cycle yml file of the tetherless project may be supplied.
