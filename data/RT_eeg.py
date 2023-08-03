import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from load_data import load_mat_data
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import mne

from electrodes_positions import (get_electrodes_positions, get_electrodes_coordinates, set_electrodes_montage)


PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"

def get_mat_filename(i_child: int, state: str):
    return "{i_child}_{state}.mat".format(i_child=i_child, state=state)

signal_filepath = str(Path(PATH_DATASET_MAT, get_mat_filename(1, "normal")))
raw, info, ch_names,eeg_signal = load_mat_data(signal_filepath)

electrodes_coordinates = get_electrodes_coordinates(ch_names)
#print(electrodes_coordinates)
dig_points = get_electrodes_positions(ch_names, electrodes_coordinates)
raw_with_dig_pts,info = set_electrodes_montage(ch_names, electrodes_coordinates,eeg_signal)

sfreq = 128  # Replace with your sampling frequency
num_channels = len(ch_names)
num_samples = 5000

def plot_realtime_eeg(edf_file):
    # Open the EDF file for reading
    f, info, ch_names, eeg_signal = load_mat_data(edf_file)
    #f = pyedflib.EdfReader(edf_file)

    # Get the number of channels and the sample rate
    num_channels = 19
    sample_rate = f.info['sfreq']

    # Initialize the plot
    fig, ax = plt.subplots()
    lines = ax.plot([], [])
    ax.set_xlim(0, 10)  # Adjust the x-axis limits as needed

    # Function to update the plot with new data
    def update_plot(frame):
        # Read the next chunk of data from the EDF file
        data = np.zeros((num_channels, frame_size))
        f.readsignals(data)

        # Update the plot with the new data
        for i, line in enumerate(lines):
            line.set_data(np.arange(frame_size) / sample_rate, data[i])

        return lines

    # Set the size of each frame (adjust as needed)
    #frame_size = int(sample_rate * 1)  # 1-second frame

    # Create an animation that calls the update_plot function every frame
    #anim = FuncAnimation(fig, update_plot, frames=1000, interval=1000)

    # Display the plot in a GUI
    #plt.show()

    # Close the EDF file
    #f.close()


# Create a Dash app
app = dash.Dash(__name__)

# Set up the layout of the app
app.layout = html.Div([
    dcc.Graph(id='eeg-plot'),
    dcc.Interval(
        id='interval-component',
        interval=1000 // sfreq,  # Update every 1/sampling frequency seconds
        n_intervals=0
    )
])

# Define a callback to update the EEG plot in real-time
@app.callback(Output('eeg-plot', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_eeg_plot(n):
    # Get the latest EEG data
    data, times = raw[:, -num_samples:]  # Update with new data from your acquisition system

    # Create the plot for each channel
    fig = go.Figure()
    for idx, ch_name in enumerate(ch_names):
        fig.add_trace(go.Scatter(x=times, y=data[idx], mode='lines', name=ch_name))

    # Update the layout
    fig.update_layout(title='Real-time EEG Data',
                      xaxis_title='Time (s)',
                      yaxis_title='Amplitude (uV)',
                      template='plotly_dark')

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

