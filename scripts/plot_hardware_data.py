import os
import rospkg
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sns.set()
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

# Set the name of the input CSV file
filename = 'Information-Ns25_2023-06-08-14-33-55_joined.csv'
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance") 
folder_path = package_dir + "/hardware_data/csv"
csv_file = folder_path + "/joined/" + filename 

is_plot = True 
print_rms = False

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

# Set the figure size and resolution
fig_size = (8, 6)  # inches
dpi = 300

def crop_col(df_col):
    """Crop the column of a dataframe between begin and end percentage of the time"""
    #return df_col.dropna().iloc[int(begin* len(df_col.dropna())):int(end* len(df_col.dropna()))]
    beg = time_bounds[df_col.name][0]
    end = time_bounds[df_col.name][1]
    return df_col.dropna().iloc[beg:end]

def main():
    global time_bounds
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(csv_file, low_memory=False)

    # Initialize the x and y axis labels
    x_label = ''
    y_label = ''

    # Occlusions
    occ_width = 0.75
    occ_center = [-1.25, -1.05]
    # list of positions of the occlusions to plot in x and y
    occ_pos = [[occ_center[0] - occ_width, occ_center[0] + occ_width, 
                occ_center[0] + occ_width, occ_center[0] - occ_width, 
                occ_center[0] - occ_width], [occ_center[1] - occ_width,
                occ_center[1] - occ_width, occ_center[1] + occ_width,
                occ_center[1] + occ_width, occ_center[1] - occ_width]]

    # Indices where 'is update data' is false
    indx_not_upd = np.where(df['is update data'].dropna().astype(bool).to_numpy() == False)[0] #
    indx_occ = np.where((df['rail nwu pose stamped position x'].dropna().to_numpy() > occ_center[0] - occ_width) &
                        (df['rail nwu pose stamped position x'].dropna().to_numpy() < occ_center[0] + occ_width) &
                        (df['rail nwu pose stamped position y'].dropna().to_numpy() > occ_center[1] - occ_width) &
                        (df['rail nwu pose stamped position y'].dropna().to_numpy() < occ_center[1] + occ_width))[0]
    print("Number of occlusions: ", len(indx_occ))

    # Begin and end percentage of the time to plot in zoomed in FOV
    beg = 0.28
    end = 0.31
    # Get min and max time from "takahe nwu pose stamped rosbagTimestamp"
    min_time = df['takahe nwu pose stamped rosbagTimestamp'].min() / 10e8
    max_time = df['takahe nwu pose stamped rosbagTimestamp'].max() / 10e8
    df['is update rosbagTimestamp'] = df['is update rosbagTimestamp'] / 10e8 - min_time 
    df['rail nwu pose stamped rosbagTimestamp'] = df['rail nwu pose stamped rosbagTimestamp'] / 10e8 - min_time
    print("Time range is: ", round(max_time - min_time, 2), "seconds" )
    beg_time = (max_time - min_time) * beg 
    end_time = (max_time - min_time) * end
    time_bounds = {}

    for col_name in df.columns:

        # Check if the column name ends with "rosbagTimestamp"
        if col_name.endswith('rosbagTimestamp'):

            # Set the x axis label to the full column name
            x_label = col_name
            if x_label != 'is update rosbagTimestamp' and x_label != 'rail nwu pose stamped rosbagTimestamp':
                df[x_label] = df[x_label] / 10e8 - min_time 

        elif any(col_name == word for word in include_data.keys()):
            # Check if the column name contains any word in the include_plot set
            
            # Set the y axis label to the full column name
            y_label = col_name

            outdir = folder_path + "/joined/figures/"
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if is_plot:
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
                #plt.show()
        elif any(col_name == word for word in cropped_plot):
            # Get index of min and max time
            min_indx = df.index[df[x_label] >= beg_time].tolist()[0]
            max_indx = df.index[df[x_label] >= end_time].tolist()[0] 
            #print("Max time: ", df[x_label][max_indx], "for column: ", col_name)
            time_bounds[col_name] = [min_indx, max_indx]
        
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

    # Print the percent of the total length that 'is update data' column is true
    with open(outdir + 'rms.csv', 'a') as csvfile:
        row_list = ['is update percent']
        perc = round(100 * np.sum(df['is update data'].dropna()) / len(df['is update data'].dropna()), 3)
        row_list.append(perc)
        writer = csv.writer(csvfile)
        writer.writerow(row_list)
    print("Percent of time 'is update data' is true: " + 
          str(perc) + "%")
        
    if is_plot:
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
        plt.title("Field of View Road Network with beg and end: " + str(beg) + " " + str(end), fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'zoomed_road' + '.png', dpi=dpi)
        plt.show()
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
        plt.legend()
        plt.savefig(outdir + 'road' + '.png', dpi=dpi)
        plt.show()

    ## All methods comparison 
    outdir_all = folder_path + "/all_runs/figures/"
    if not os.path.exists(outdir_all):
        os.mkdir(outdir_all)
    error_df = pd.DataFrame()
    entropy_df = pd.DataFrame()
    #cat_list = ['entropy data', 'n eff particles data']
    err_list = ['err estimation norm', 'err tracking norm']
    for filename in os.listdir(folder_path + "/all_runs/"):
        if filename.endswith(".csv"):
            first_word = filename.split("_")[0]

            # extract the first word from the file name
            file_df = pd.read_csv(folder_path + "/all_runs/" + filename, low_memory=False)

            for col in err_list:
                values = file_df[col]

                row = pd.DataFrame({
                    'Guidance Method': first_word,
                    'hue': col,
                    'Error [m]': values
                })
                error_df = pd.concat([error_df, row], ignore_index=True)

            values = file_df['entropy data']
            row = pd.DataFrame({
                'Guidance Method': first_word,
                'Entropy': values
            })
            entropy_df = pd.concat([entropy_df, row], ignore_index=True)

    rms_estimator = lambda x: np.sqrt(np.mean(np.square(x))) 

    dpi = 300
    # BOX
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x = "Guidance Method",
                y = "Error [m]",
                data = error_df,
                ax = ax,
                hue = "hue", 
                showfliers=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', fontsize=16, title_fontsize=20)
    sns.despine(right = True)
    plt.savefig(outdir_all + 'box_guidance' + '.png', dpi=dpi)

    # BAR
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x = "Guidance Method",
                y = "Error [m]",
                data = error_df,
                ax = ax,
                hue = "hue", 
                estimator = rms_estimator,
                capsize = 0.1,
                errorbar =  "sd")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', fontsize=16, title_fontsize=20)
    sns.despine(right = True)
    plt.savefig(outdir_all + 'bar_guidance' + '.png', dpi=dpi)
    #plt.show()

    # BAR entropy
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x = "Guidance Method",
                y = "Entropy",
                data = entropy_df,
                ax = ax,
                estimator = rms_estimator,
                capsize = 0.1,
                errorbar =  "sd")
    sns.despine(right = True)
    plt.savefig(outdir_all + 'bar_entropy' + '.png', dpi=dpi)
    #plt.show()

if __name__ == '__main__':
    main()
