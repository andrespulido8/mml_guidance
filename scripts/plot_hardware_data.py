import rospkg
import pandas as pd
import matplotlib.pyplot as plt

# Set the name of the input CSV file
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/joined/"
csv_file = folder_path + 'Information' + '_joined.csv'

# Set the list of column names to include in the plots
include_plot = {
    'entropy data', 'rail  nwu  pose stamped x', 'rail  nwu  pose stamped y',
    'takahe  nwu  pose stamped x', 'takahe  nwu  pose stamped y', 'xyTh estimate x',
    'xyTh estimate y', 'xyTh estimate yaw', }

# Set the font sizes for the plot labels
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

# Set the figure size and resolution
fig_size = (8, 6)  # inches
dpi = 300

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

        elif any(word in col_name for word in include_plot):
            # Check if the column name contains any word in the include_plot set

            # Create a new plot with the x and y data
            plt.figure(figsize=fig_size)
            # Set the y axis label to the full column name
            y_label = col_name

            # Plot the data
            indx = int(len(df[x_label]) * 0.2)
            # print how many values in the column are nans
            plt.plot(df[x_label], df[y_label], linewidth=2)

            # Plot the vertical line
            #plt.axvline(x=df[x_label][indx], color='g', linestyle='-')

            # Add the legend and axis labels
            # plt.legend()
            plt.xlabel(x_label, fontdict=font)
            plt.ylabel(y_label, fontdict=font)
         
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
