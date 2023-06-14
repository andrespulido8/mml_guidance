import os
import rospkg
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

# Set the name of the input CSV file
filename = 'Lawnmower_2023-06-07-16-41-58_joined.csv'
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/csv/joined/"
csv_file = folder_path + filename 

is_plot = False 

# Set the list of column names to include in the plots
include_data = {
    'err estimation norm', 'err tracking norm', 
    'entropy data', 'n eff particles data', 'eer time data'
    }
include_plot = include_data 

# Set the font sizes for the plot labels
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

# Set the figure size and resolution
fig_size = (8, 6)  # inches
dpi = 300

def crop_col(df_col, begin, end):
    # Crop the column data to remove the first 20% of the data
    return df_col.dropna().iloc[int(begin* len(df_col.dropna())):int(end* len(df_col.dropna()))]

def main():
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(csv_file, low_memory=False)

    # Initialize the x and y axis labels
    x_label = ''
    y_label = ''

    # Indices where 'is update data' is false
    indx_not = np.where(df['is update data'].dropna().astype(bool).to_numpy() == False)[0] #

    for col_name in df.columns:

        # Check if the column name ends with "rosbagTimestamp"
        if col_name.endswith('rosbagTimestamp'):

            # Set the x axis label to the full column name
            x_label = col_name

            #print("\nColumn name: ", col_name)
            #print("Data size: ", len(df[x_label].dropna()))
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
        # Err estimation
        plt.figure(figsize=fig_size)
        t0 = df['err estimation rosbagTimestamp'][0]
        plt.plot(df['err estimation rosbagTimestamp'] - t0, df['err estimation x'], linewidth=2, label='x error')
        plt.plot(df['err estimation rosbagTimestamp'] - t0, df['err estimation y'], linewidth=2, label='y error')
        for xi in df['is update rosbagTimestamp'][indx_not] - t0:
            plt.axvline(x=xi, alpha=0.3, color='r', linestyle='-', linewidth=0.8)
        plt.axvline(x=df['is update rosbagTimestamp'][indx_not[0]] - t0, alpha=0.3, color='r', linestyle='-', linewidth=0.8, label="Occlusion")
        plt.xlabel("Time [s]", fontdict=font)
        plt.ylabel("Estimation Error [m]", fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'estimation' + '.png', dpi=dpi)
        #plt.show()
        # Err tracking
        plt.figure(figsize=fig_size)
        t0 = df['err tracking rosbagTimestamp'][0]
        plt.plot(df['err tracking rosbagTimestamp'] - t0, df['err tracking x'], linewidth=2, label='x error')
        plt.plot(df['err tracking rosbagTimestamp'] - t0, df['err tracking y'], linewidth=2, label='y error')
        for xi in df['is update rosbagTimestamp'][indx_not] - t0:
            plt.axvline(x=xi, alpha=0.3, color='r', linestyle='-', linewidth=0.8)
        plt.axvline(x=df['is update rosbagTimestamp'][indx_not[0]] - t0, alpha=0.3, color='r', linestyle='-', linewidth=0.8, label="Occlusion")
        plt.xlabel("Time [s]", fontdict=font)
        plt.ylabel("Tracking Error [m]", fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'tracking' + '.png', dpi=dpi)
        #plt.show()
        # FOV
        plt.figure(figsize=fig_size)
        beg = 0.1
        end = 0.9
        plt.plot(crop_col(df['rail nwu pose stamped position x'], beg, end), crop_col(df['rail nwu pose stamped position y'], beg, end), linewidth=2, label='turtlebot path')
        plt.plot(crop_col(df['takahe nwu pose stamped position x'], beg, end), crop_col(df['takahe nwu pose stamped position y'], beg, end), linewidth=2, label='drone path')
        plt.scatter(crop_col(df['desired state x'], beg, end), crop_col(df['desired state y'], beg, end), alpha=0.2, c='g', marker='s', s=500, label='desired position') 
        plt.xlabel("X position [m]", fontdict=font)
        plt.ylabel("Y position [m]", fontdict=font)
        plt.title("Field of View Road Network", fontdict=font)
        plt.legend()
        plt.savefig(outdir + 'road' + '.png', dpi=dpi)
        #plt.show()


    method_data = {} 
    for filename in os.listdir(outdir):
        if filename.endswith(".csv"):

            # extract the first word from the file name
            method_data[filename] = pd.read_csv(filename, low_memory=False)

            #sns.boxplot(data=method_data, x='Method', y=['err estimation norm', 'err tracking norm', 'entropy data', 'n eff particles data', 'eer time data'])
            # Convert selected columns to numeric data
            numeric_cols = ['err estimation norm', 'err tracking norm', 'entropy data', 'n eff particles data', 'eer time data']


if __name__ == '__main__':
    main()
