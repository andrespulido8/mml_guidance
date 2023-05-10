import os
import rospkg
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

# Set the name of the input CSV file
guidance_mode = 'Information'
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/joined/"
csv_file = folder_path + guidance_mode + '_joined.csv'

is_plot = False

# Set the list of column names to include in the plots
include_data = {
    'err estimation x', 'err estimation y', 'err tracking x', 'err tracking y', 
    'entropy data', 'info_gain', 'eer time data'
    }
include_plot = {
    'entropy data', 'rail  nwu  pose stamped x', 'rail  nwu  pose stamped y',
    'takahe  nwu  pose stamped x', 'takahe  nwu  pose stamped y', 'xyTh estimate x',
    'xyTh estimate y', 'xyTh estimate yaw', 'err estimation x', 'err estimation y',
    'err tracking x', 'err tracking y', 'n eff particles data', 'eer time data',
    'desired state x', 'desired state y', 'info_gain'
    }

# Set the font sizes for the plot labels
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

# Set the figure size and resolution
fig_size = (8, 6)  # inches
dpi = 300

def crop_col(df_col):
    # Crop the column data to remove the first 20% of the data
    return df_col.dropna().iloc[int(0.3 * len(df_col.dropna())):]

def main():
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(csv_file, low_memory=False)

    # Initialize the x and y axis labels
    x_label = ''
    y_label = ''

    for col_name in df.columns:

        # Check if the column name ends with "rosbagTimestamp"
        if col_name.endswith('rosbagTimestamp'):

            # Set the x axis label to the full column name
            x_label = col_name
        elif any(col_name == word for word in include_plot):
            # Check if the column name contains any word in the include_plot set
            
            # Set the y axis label to the full column name
            y_label = col_name

            outdir = folder_path + "/figures/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if is_plot:
                # Create a new plot with the x and y data
                plt.figure(figsize=fig_size)

                # Plot the data
                t0 = df[x_label][0]
                plt.plot(crop_col(df[x_label]) - t0, crop_col(df[y_label]), linewidth=2)

                # Plot the vertical line
                #plt.axvline(x=df[x_label][indx], color='g', linestyle='-')

                # Add the legend and axis labels
                plt.xlabel("Time [s]", fontdict=font)
                plt.ylabel(y_label, fontdict=font)
                plt.savefig(outdir + y_label.replace(" ", "_") + '.png', dpi=dpi)
        
        if any(col_name == word for word in include_data):
            with open(outdir + 'rms.csv', 'a') as csvfile:
                row_list = [col_name]
                # Print the root mean square of the y data with two decimal places
                rms = round(np.sqrt(np.mean(df[y_label]**2)), 3)
                print("RMS of " + y_label + ": " + str(rms))
                row_list.append(rms)
                writer = csv.writer(csvfile)
                writer.writerow(row_list)

    # Print the percent of the total length that 'is update data' column is true
    with open(outdir + 'rms.csv', 'a') as csvfile:
        row_list = ['is update percent']
        perc = round(100 * np.sum(df['is update data'].dropna()) / len(df['is update data'].dropna()), 3)
        row_list.append(perc)
        writer = csv.writer(csvfile)
        writer.writerow(row_list)
    print("Percent of time 'is update data' is true: " + 
          str(perc) + "%")

    # Err estimation
    if guidance_mode != 'Lawnmower':
        # Err estimation
        plt.figure(figsize=fig_size)
        t0 = df['err estimation rosbagTimestamp'][0]
        plt.plot(crop_col(df['err estimation rosbagTimestamp']) - t0, crop_col(df['err estimation x']), linewidth=2, label='x error')
        plt.plot(crop_col(df['err estimation rosbagTimestamp']) - t0, crop_col(df['err estimation y']), linewidth=2, label='y error')
        plt.xlabel("Time [s]", fontdict=font)
        plt.ylabel("Estimation Error [m]", fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'estimation' + '.png', dpi=dpi)
        # Err tracking
        plt.figure(figsize=fig_size)
        t0 = df['err tracking rosbagTimestamp'][0]
        plt.plot(crop_col(df['err tracking rosbagTimestamp']) - t0, crop_col(df['err tracking x']), linewidth=2, label='x error')
        plt.plot(crop_col(df['err tracking rosbagTimestamp']) - t0, crop_col(df['err tracking y']), linewidth=2, label='y error')
        plt.xlabel("Time [s]", fontdict=font)
        plt.ylabel("Tracking Error [m]", fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'estimation' + '.png', dpi=dpi)
    else:
        # Err tracking for files where the err estimation data is not available
        plt.figure(figsize=fig_size)
        errx = crop_col(df['takahe  nwu  pose stamped x'] - df['rail  nwu  pose stamped x'])
        erry = crop_col(df['takahe  nwu  pose stamped y'] - df['rail  nwu  pose stamped y'])
        t0 = df['takahe  nwu  pose stamped rosbagTimestamp'][0]
        plt.plot(crop_col(df['takahe  nwu  pose stamped rosbagTimestamp']) - t0, errx, linewidth=2, label='x error')
        plt.plot(crop_col(df['takahe  nwu  pose stamped rosbagTimestamp']) - t0, erry, linewidth=2, label='y error')
        rms_x = round(np.sqrt(np.mean(errx**2)), 3)
        rms_y = round(np.sqrt(np.mean(erry**2)), 3)
        print("RMS of " + "err x" + ": " + str(rms_x))
        print("RMS of " + "err y" + ": " + str(rms_y))
        plt.xlabel("Time [s]", fontdict=font)
        plt.ylabel("Tracking Error [m]", fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'tracking' + '.png', dpi=dpi)
    # Road network
    plt.figure(figsize=fig_size)
    plt.plot(crop_col(df['rail  nwu  pose stamped x']), crop_col(df['rail  nwu  pose stamped y']), linewidth=2, label='turtlebot path')
    plt.plot(crop_col(df['takahe  nwu  pose stamped x']), crop_col(df['takahe  nwu  pose stamped y']), linewidth=2, label='drone path')
    plt.plot(crop_col(df['desired state x']), crop_col(df['desired state y']), linewidth=2, label='desired position')
    plt.xlabel("X position [m]", fontdict=font)
    plt.ylabel("Y position [m]", fontdict=font)
    plt.legend()
    plt.savefig(outdir + 'road' + '.png', dpi=dpi)

    # Entropy
    #plt.figure(figsize=fig_size)
    #plt.plot(crop_col(df['entropy rosbagTimestamp']) - t0, crop_col(df['entropy data']), linewidth=2)
    #plt.xlabel("Time [s]", fontdict=font)
         
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
