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
filename = 'Information-Ns25_all_runs.csv'
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/csv/all_runs/"
csv_file = folder_path + filename 

is_plot = False 
print_rms = False

# Set the list of column names to include in the plots
include_data = {
    'err estimation norm', 'err tracking norm', 
    'entropy data', 'n eff particles data', 'eer time data'
    }
include_plot = { 
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

    # Indices where 'is update data' is false
    indx_not = np.where(df['is update data'].dropna().astype(bool).to_numpy() == False)[0] #
    beg = 0.28
    end = 0.31
    # Get min and max time from "takahe nwu pose stamped rosbagTimestamp"
    min_time = df['takahe nwu pose stamped rosbagTimestamp'].min() / 10e8
    max_time = df['takahe nwu pose stamped rosbagTimestamp'].max() / 10e8
    print("Time range is: ", max_time - min_time)
    beg_time = (max_time - min_time) * beg 
    end_time = (max_time - min_time) * end
    time_bounds = {}

    for col_name in df.columns:

        # Check if the column name ends with "rosbagTimestamp"
        if col_name.endswith('rosbagTimestamp'):

            # Set the x axis label to the full column name
            x_label = col_name
            df[x_label] = df[x_label] / 10e8 - min_time 

        elif any(col_name == word for word in include_data):
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
                plt.plot(df[x_label] - t0, df[y_label], linewidth=2)

                # vertical line when there is occlusions or target lost
                for xi in df['is update rosbagTimestamp'][indx_not] - t0:
                    plt.axvline(x=xi, alpha=0.3, color='r', linestyle='-', linewidth=0.8)
                plt.axvline(x=df['is update rosbagTimestamp'][indx_not[0]] - t0, alpha=0.3, 
                            color='r', linestyle='-', linewidth=0.8, label="Occlusion")

                # Add the legend and axis labels
                plt.xlabel("Time [s]", fontdict=font)
                plt.ylabel(y_label, fontdict=font)
                plt.legend()
                plt.savefig(outdir + y_label.replace(" ", "_") + '.png', dpi=dpi)
                #plt.show()
        elif any(col_name == word for word in include_plot):
            # Get index of min and max time
            min_indx = df.index[df[x_label] >= beg_time].tolist()[0]
            max_indx = df.index[df[x_label] >= end_time].tolist()[0] 
            print("Max time: ", df[x_label][max_indx], "for column: ", col_name)
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
        # FOV
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
        plt.scatter(crop_col(df['desired state x']), crop_col(df['desired state y']), alpha=0.7, c=colors_fov, marker='s', s=4000, label='Reference Position')
        plt.plot(crop_col(df['desired state x']), crop_col(df['desired state y']), alpha=0.2, color='g',)
        plt.xlabel("X position [m]", fontdict=font)
        plt.ylabel("Y position [m]", fontdict=font)
        plt.title("Field of View Road Network with beg and end: " + str(beg) + " " + str(end), fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'zoomed_road' + '.png', dpi=dpi)
        plt.show()

    ## All methods comparison 
    error_df = pd.DataFrame()
    entropy_df = pd.DataFrame()
    #cat_list = ['entropy data', 'n eff particles data']
    err_list = ['err estimation norm', 'err tracking norm']
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            first_word = filename.split("_")[0]

            # extract the first word from the file name
            file_df = pd.read_csv(folder_path + filename, low_memory=False)

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
    plt.savefig(outdir + 'box_guidance' + '.png', dpi=dpi)

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
    plt.savefig(outdir + 'bar_guidance' + '.png', dpi=dpi)
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
    plt.savefig(outdir + 'bar_entropy' + '.png', dpi=dpi)
    #plt.show()

if __name__ == '__main__':
    main()
