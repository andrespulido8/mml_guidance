import os
import rospkg
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Set the name of the input CSV file
filename = 'Information_2023-06-21-16-42-45_joined.csv'
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance") 
folder_path = package_dir + "/hardware_data/csv/joined/"
csv_file = folder_path + filename 

is_plot = False 
print_rms = True 

# Set the list of column names to include in the plots
include_data = {
    'err estimation norm':'$e_{estimation}$ [m]', 'err tracking norm':'$e_{tracking}$ [m]', 
    'entropy data':'Entropy', 'n eff particles data':'Effective Number of Particles',
    }
cropped_plot = { 
    'desired state x', 'rail nwu pose stamped position x', 'takahe nwu pose stamped position x', 
    'desired state y', 'rail nwu pose stamped position y', 'takahe nwu pose stamped position y', 
    }

# Set the font sizes for the plot labels
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
sns.set()
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

def crop_col(df_col):
    """Crop the column of a dataframe between begin and end percentage of the time"""
    #return df_col.dropna().iloc[int(begin* len(df_col.dropna())):int(end* len(df_col.dropna()))]
    beg = time_bounds_dict[df_col.name][0]
    end = time_bounds_dict[df_col.name][1]
    return df_col.dropna().iloc[beg:end]

def main():
    global time_bounds_dict
    for filename in os.listdir(folder_path):
        print("Filename: ", filename)
        csv_file = folder_path + filename 
        # Read the CSV file into a pandas dataframe
        df = pd.read_csv(csv_file, low_memory=False)

        # Initialize the x and y axis labels
        x_label = ''
        y_label = ''

        # Begin and end percentage of the time to plot 
        beg_end = np.array([0.0, 0.90])
        # Get min and max time from topic with most data 
        min_time = df['takahe nwu pose stamped rosbagTimestamp'].min() / 10e8
        max_time = df['takahe nwu pose stamped rosbagTimestamp'].max() / 10e8
        crop_time = (max_time - min_time) * beg_end 
        print("Time range before crop is: ", round(max_time - min_time, 2), "seconds" )
        time_bounds_dict = {}
        for col_name in df.columns:
            if col_name.endswith('rosbagTimestamp'):
                x_label = col_name
                df[col_name] = df[col_name] / 10e8 - min_time
                min_max_indx = [df.index[df[x_label] >= crop_time[0]].tolist()[0], 
                                df.index[df[x_label] >= crop_time[1]-0.6].tolist()[0]]
            time_bounds_dict[col_name] = min_max_indx 
            df[col_name] = crop_col(df[col_name])

        min_time = df['takahe nwu pose stamped rosbagTimestamp'].min()
        max_time = df['takahe nwu pose stamped rosbagTimestamp'].max()
        print("Time range: ", round(max_time - min_time, 2), "seconds" )
        # Crop the data between begin and end for zoomed in plot 
        zoomed_beg_end = np.array([0.7, 0.73])
        zoomed_time = (max_time - min_time) * zoomed_beg_end 

        # Occlusions
        occ_width = 0.75
        occ_center = np.array([-1.25, -1.05])
        # list of positions of the occlusions to plot in x and y
        occ_pos = [[occ_center[0] - occ_width / 2, occ_center[0] + occ_width / 2,
                    occ_center[0] + occ_width / 2, occ_center[0] - occ_width / 2,
                    occ_center[0] - occ_width / 2], [occ_center[1] - occ_width / 2,
                    occ_center[1] - occ_width / 2, occ_center[1] + occ_width / 2,
                    occ_center[1] + occ_width / 2, occ_center[1] - occ_width / 2]]

        # Indices where 'is update data' is false
        indx_not_upd = np.where(df['is update data'].dropna().astype(bool).to_numpy() == False)[0] #
        indx_occ = np.where((df['rail nwu pose stamped position x'].dropna().to_numpy() > occ_center[0] - occ_width / 2) &
                            (df['rail nwu pose stamped position x'].dropna().to_numpy() < occ_center[0] + occ_width / 2) &
                            (df['rail nwu pose stamped position y'].dropna().to_numpy() > occ_center[1] - occ_width / 2) &
                            (df['rail nwu pose stamped position y'].dropna().to_numpy() < occ_center[1] + occ_width / 2))[0] # print("Number of occlusions: ", len(indx_occ))

        for col_name in df.columns:
            if col_name.endswith('rosbagTimestamp'):
                # Set the x axis label to the full column name
                x_label = col_name
                df[x_label] = df[x_label] - min_time 

            elif any(col_name == word for word in include_data.keys()):
                # Set the y axis label to the full column name
                y_label = col_name

                outdir = folder_path + "/figures/" + filename + "/"
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                # Create a new plot with the x and y data
                fig_size = (8, 6)  # inches
                dpi = 300
                plt.figure(figsize=fig_size)

                # Plot the data
                plt.plot(df[x_label], df[y_label], linewidth=1)
                plt.scatter(df[x_label], df[y_label], marker='.', s=20)

                # vertical line when target lost
                for xi_u in df['is update rosbagTimestamp'][indx_not_upd]:
                    plt.axvline(x=xi_u, alpha=0.4, color='k', linestyle='-', linewidth=0.8)
                plt.axvline(x=df['is update rosbagTimestamp'][indx_not_upd[0]], alpha=0.4, 
                            color='k', linestyle='-', linewidth=0.8, label="No Update")
                # vertical line when there is occulsion (every 6th occulsion to avoid clutter) 
                for xi_o in df['rail nwu pose stamped rosbagTimestamp'][indx_occ[::6]]: 
                    plt.axvline(x=xi_o, alpha=0.1, color='r', linestyle='-', linewidth=0.8)
                plt.axvline(x=df['rail nwu pose stamped rosbagTimestamp'][indx_occ[0]], alpha=0.1, 
                            color='r', linestyle='-', linewidth=0.8, label="Occlusion")

                # Add the legend and axis labels
                plt.xlabel("Time [s]", fontdict=font)
                plt.ylabel(include_data[y_label], fontdict=font)
                plt.legend(loc='upper right', fontsize=16)
                plt.savefig(outdir + y_label.replace(" ", "_") + '.png', dpi=dpi)
                plt.show() if is_plot else plt.close()

            elif any(col_name == word for word in cropped_plot):
                # Get index of min and max time
                min_max_indx = [df.index[df[x_label] >= zoomed_time[0]].tolist()[0], 
                                df.index[df[x_label] >= zoomed_time[1]].tolist()[0]]
                #print("Max time: ", df[x_label][min_max_indx[0]], "for column: ", col_name)
                time_bounds_dict[col_name] = min_max_indx 

            if print_rms:
                if any(col_name == word for word in include_data):
                    with open(outdir + 'rms.csv', 'a') as csvfile:
                        row_list = [col_name]
                        # Print the root mean square of the y data with two decimal places
                        rms = round(np.sqrt(np.mean(df[y_label]**2)), 3)
                        print("RMS of " + y_label + ": " + str(rms))
                        row_list.append(rms)
                        writer = csv.writer(csvfile)
                        writer.writerow(row_list)

        if print_rms:
            # Print the percent of the total length that 'is update data' column is true
            with open(outdir + 'rms.csv', 'a') as csvfile:
                row_list = ['is update percent']
                perc = round(100 * np.sum(df['is update data'].dropna()) / len(df['is update data'].dropna()), 3)
                row_list.append(perc)
                writer = csv.writer(csvfile)
                writer.writerow(row_list)
            print("Percent of time 'is update data' is true: " + 
                str(perc) + "%")

        # zoomed in FOV
        fig_size = (8, 6)  # inches
        dpi = 800
        plt.figure(figsize=fig_size)
        alphas_robot = np.linspace(0.1, 0.8, len(crop_col(df['rail nwu pose stamped position x'])))
        colors_robot = cm.Blues(alphas_robot)
        plt.scatter(crop_col(df['rail nwu pose stamped position x']), crop_col(df['rail nwu pose stamped position y']), c=colors_robot, marker='.', label='Turtlebot')
        alphas_quad = np.linspace(0.1, 0.6, len(crop_col(df['takahe nwu pose stamped position x'])))
        colors_quad = cm.Oranges(alphas_quad)
        plt.scatter(crop_col(df['takahe nwu pose stamped position x']), crop_col(df['takahe nwu pose stamped position y']), c=colors_quad, marker='.', label='Quadcopter')
        alphas_fov = np.linspace(0.05, 0.55, len(crop_col(df['desired state x'])))
        colors_fov = cm.Greens(alphas_fov)
        plt.scatter(crop_col(df['desired state x']), crop_col(df['desired state y']), alpha=0.7, c=colors_fov, marker='s', s=4000)
        plt.plot(crop_col(df['desired state x']), crop_col(df['desired state y']), alpha=0.2, color='g', label='Reference Position')
        plt.xlabel("X position [m]", fontdict=font)
        plt.ylabel("Y position [m]", fontdict=font)
        plt.title("Field of View Road Network with beg and end: " + str(zoomed_beg_end), fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'zoomed_road' + '.png', dpi=dpi)
        plt.show() if is_plot else plt.close()
        # FOV
        fig_size = (8, 6)  # inches
        dpi = 800
        plt.figure(figsize=fig_size)
        plt.scatter(df['rail nwu pose stamped position x'], df['rail nwu pose stamped position y'], marker='.', label='Turtlebot')
        plt.scatter(df['takahe nwu pose stamped position x'], df['takahe nwu pose stamped position y'], marker='.', label='Quadcopter')
        plt.scatter(df['desired state x'], df['desired state y'], alpha=0.05, marker='s', s=4000)
        plt.plot(df['desired state x'], df['desired state y'], alpha=0.2, color='g', label='Reference Position')
        plt.plot(occ_pos[0], occ_pos[1], '--r', linewidth=2, label='Occlusion')
        plt.xlabel("X position [m]", fontdict=font)
        plt.ylabel("Y position [m]", fontdict=font)
        plt.title("Field of View Road Network", fontdict=font)
        plt.legend('upper right')
        plt.savefig(outdir + 'road' + '.png', dpi=dpi)
        plt.show() if is_plot else plt.close()


if __name__ == '__main__':
    main()
