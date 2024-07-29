import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import json

def create_animation_for_data_id(data_id, data_dir):
    file_path = os.path.join(data_dir, f'{data_id}.jsonl')
    if not os.path.exists(file_path):
        return 'File not found'

    # Read and process the JSONL data
    with open(file_path, 'r') as file:
        json_data = file.read().strip().split('\n')
    
    data = [json.loads(line) for line in json_data if line.strip()]
    
    if not data:
        return 'No valid data in file'

    full_df = pd.DataFrame(data)

    positions = []
    velocities = []
    accelerations = []
    times = []

    data = full_df.to_dict(orient='records')[-1]

    for sample in data["samples_ball"]:
        if "pos" in sample:
            positions.append(sample["pos"])
        if "vel" in sample:
            velocities.append(sample["vel"])
        if "acc" in sample:
            accelerations.append(sample["acc"])
        if "time" in sample:
            times.append(sample["time"])

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    times = np.array(times)

    # Calculate speed (magnitude of velocity)
    speeds = np.linalg.norm(velocities, axis=1)

    samples_bat = data['samples_bat']
    samples_ball = data['samples_ball']

    # Extracting handle, head, and ball positions with timestamps
    handle_positions = [(sample['time'], sample['handle']['pos']) for sample in samples_bat]
    head_positions = [(sample['time'], sample['head']['pos']) for sample in samples_bat]
    ball_positions = [(sample['time'], sample['pos']) for sample in samples_ball if 'pos' in sample]

    # Synchronize the data by timestamp
    timestamps = sorted(set([time for time, _ in handle_positions] + [time for time, _ in head_positions] + [time for time, _ in ball_positions]))
    handle_dict = {time: pos for time, pos in handle_positions}
    head_dict = {time: pos for time, pos in head_positions}
    ball_dict = {time: pos for time, pos in ball_positions}

    handle_positions_sync = [handle_dict.get(time, (np.nan, np.nan, np.nan)) for time in timestamps]
    head_positions_sync = [head_dict.get(time, (np.nan, np.nan, np.nan)) for time in timestamps]
    ball_positions_sync = [ball_dict.get(time, (np.nan, np.nan, np.nan)) for time in timestamps]

    # Splitting handle, head, and ball positions into x, y, z coordinates
    handle_x, handle_y, handle_z = zip(*handle_positions_sync)
    head_x, head_y, head_z = zip(*head_positions_sync)
    ball_x, ball_y, ball_z = zip(*ball_positions_sync)

    # Function to create and save the animation from a specific view
    def create_animation(view_angle, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([min(min(handle_x), min(head_x), min(ball_x)), max(max(handle_x), max(head_x), max(ball_x))])
        ax.set_ylim([min(min(handle_y), min(head_y), min(ball_y)), max(max(handle_y), max(head_y), max(ball_y))])
        ax.set_zlim([min(min(handle_z), min(head_z), min(ball_z)), max(max(handle_z), max(head_z), max(ball_z))])
        
        # Set the view angle
        ax.view_init(*view_angle)
        
        handle_line, = ax.plot([], [], [], color='r', label='Handle')
        head_line, = ax.plot([], [], [], color='b', label='Head')
        ball_line, = ax.plot([], [], [], color='g', label='Ball')
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        def init():
            handle_line.set_data([], [])
            handle_line.set_3d_properties([])
            head_line.set_data([], [])
            head_line.set_3d_properties([])
            ball_line.set_data([], [])
            ball_line.set_3d_properties([])
            time_text.set_text('')
            return handle_line, head_line, ball_line, time_text

        def update(frame):
            handle_line.set_data(handle_x[:frame], handle_y[:frame])
            handle_line.set_3d_properties(handle_z[:frame])
            head_line.set_data(head_x[:frame], head_y[:frame])
            head_line.set_3d_properties(head_z[:frame])
            ball_line.set_data(ball_x[:frame], ball_y[:frame])
            ball_line.set_3d_properties(ball_z[:frame])
            time_text.set_text(f'Time: {timestamps[frame]}')
            return handle_line, head_line, ball_line, time_text

        anim = FuncAnimation(fig, update, frames=len(timestamps), init_func=init, blit=True)

        writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(filename, writer=writer)

        plt.close(fig)

    # Define view angles and filenames
    view_angles = [(30, 30), (60, 30), (90, 30)]
    filenames = [f'videos/{data_id}_view{i+1}.mp4' for i in range(len(view_angles))]
    
    # Ensure the videos directory exists
    os.makedirs('videos', exist_ok=True)
    
    for view_angle, filename in zip(view_angles, filenames):
        create_animation(view_angle, filename)
    
    return None
