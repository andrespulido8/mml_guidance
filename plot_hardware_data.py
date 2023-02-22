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
    'entropy_data', 'rail_slash_nwu_slash_pose_stamped_x', 'rail_slash_nwu_slash_pose_stamped_y',
    'takahe_slash_nwu_slash_pose_stamped_x', 'takahe_slash_nwu_slash_pose_stamped_y', 'xyTh_estimate_x',
    'xyTh_estimate_y', 'xyTh_estimate_yaw', }

# Set the font sizes for the plot labels
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}

# Set the figure size and resolution
fig_size = (8, 6)  # inches
dpi = 300

# Read the CSV file into a pandas dataframe
df = pd.read_csv(csv_file)

# Initialize the x and y axis labels
x_label = ''
y_label = ''

# Iterate over the columns of the dataframe
for col_name in df.columns:
    
    # Check if the column name ends with "rosbagTimestamp"
    if col_name.endswith('rosbagTimestamp'):
        
        # Set the x axis label to the full column name
        x_label = col_name
        
    # Check if the column name contains any word in the include_plot set
    elif any(word in col_name for word in include_plot):
        
        # Set the y axis label to the full column name
        y_label = col_name
        
        # Create a new plot with the x and y data
        plt.figure(figsize=fig_size)
        plt.plot(df[x_label], df[y_label], linewidth=2)
        
        # Add the legend and axis labels
        #plt.legend()
        plt.xlabel(x_label, fontdict=font)
        plt.ylabel(y_label, fontdict=font)
        
        # Set the tick label font sizes
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Save the plot to a file
        fig_file = col_name + '.png'
        #plt.savefig(fig_file, dpi=dpi, bbox_inches='tight')
        
# Display the plot
plt.show()
