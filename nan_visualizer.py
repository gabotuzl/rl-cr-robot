from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy import VideoClip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ast

# Creating a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def position_fetcher(episode):
    with open (f'20_test_runs_new/position_data/test_run_{episode}.txt', 'r') as f:
        content = f.read()
    nested_list = ast.literal_eval(content)

    position_array = np.array(nested_list)
    
    return position_array

def mplfig_to_npimage(fig):
    """
    Converts a Matplotlib figure to a RGB numpy array (H x W x 3).
    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3]
    return image

# function to get frames
def make_frame(t):

    global count

    
    # clear
    ax.clear()
     
    # Scatter plot
    ax.scatter(position_data[count][0], position_data[count][1], position_data[count][2])
    ax.axes.set_zlim3d(bottom=-base_length,top=base_length)
    ax.axes.set_ylim3d(bottom=-base_length,top=base_length)
    ax.axes.set_xlim(-base_length/2,base_length)

    # Labeling axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    calculated_x = position_data[count][0][-1]
    calculated_y = position_data[count][1][-1]
    calculated_z = position_data[count][2][-1]

    x_point = round(calculated_x,3)
    y_point = round(calculated_y,3)
    z_point = round(calculated_z,3)
    annotation_text = f'Tip Pos: ({x_point}, {y_point}, {z_point})'
    ax.text(x_point, y_point, z_point, annotation_text, color='red')


    # Update counter
    count += 1
    # returning numpy imagedef make_frame(t):
    return mplfig_to_npimage(fig)
 
for episode in range(1,21):
    position_data = position_fetcher(episode)
    count=0
    base_length = 0.25

    # creating animation
    clip = VideoClip(make_frame, duration = 6.0)
    clip.write_videofile(f"20_test_runs_new/videos/test_video_{episode}.mp4", codec = "libx264", fps = 10)
