import os
import pandas as pd
import rospkg

# specify the folder containing the CSV files
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data"

# create an empty dictionary to store the dataframes for each group
dfs_by_group = {}

# loop through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # extract the first word from the file name
        first_word = filename.split("_")[0]
        # read the CSV file into a pandas dataframe
        df = pd.read_csv(os.path.join(folder_path, filename))
        # extract the topic name from the file name (-4 removes the ".csv" extension)

        last_words = '_'.join(filename.split("_")[4:])[0:-4] 
        if last_words == 'xyTh_estimate': 
            # loop through the columns in the dataframe
            columns_to_drop = []
            drop_col = False
            for i, col in enumerate(df.columns):
                # ignore columns that are followed by the "all_particle" column
                if i < len(df.columns) - 1 and df.columns[i+1] == "all_particle":
                    drop_col = True
                if drop_col:
                    columns_to_drop.append(col)

            # drop the columns that are followed by the "all_particle" column
            df.drop(columns_to_drop, axis=1, inplace=True)

        df.rename(columns={col: f"{last_words}_{col}" if i >= 0 else col for i, col in enumerate(df.columns)}, inplace=True)
        # add the dataframe to the dictionary for the appropriate group
        if first_word in dfs_by_group:
            dfs_by_group[first_word].append(df)
        else:
            dfs_by_group[first_word] = [df]

# loop through the dictionary and join the dataframes for each group
for group_name, group_dfs in dfs_by_group.items():
    # concatenate the dataframes for the group along the rows
    joined_df = pd.concat(group_dfs, axis=1)

    # write the joined dataframe to a CSV file
    outdir = folder_path + '/joined'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print("Creating file: " + f"{outdir}/{group_name}_joined.csv")
    joined_df.to_csv(f"{outdir}/{group_name}_joined.csv", index=False)
