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
        # add the dataframe to the dictionary for the appropriate group
        if first_word in dfs_by_group:
            dfs_by_group[first_word].append(df)
        else:
            dfs_by_group[first_word] = [df]

# loop through the dictionary and join the dataframes for each group
for group_name, group_dfs in dfs_by_group.items():
    # concatenate the dataframes for the group along the rows
    joined_df = pd.concat(group_dfs, axis=0, ignore_index=True)
    outdir = folder_path + '/joined'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    joined_df.to_csv(f"{outdir}/{group_name}_joined.csv", index=False)
    # keep only the "rosbagTimestamp" column
    #joined_df = joined_df.loc[:, ["rosbagTimestamp"]]
    # loop through the columns in the joined dataframe
    columns_to_drop = []
    print("columns: " + str(joined_df.columns))
    for i, col in enumerate(joined_df.columns):
        # skip the "rosbagTimestamp" column
        if col == "rosbagTimestamp":
            continue
        # ignore columns that are followed by the "all_particle" column
        if i < len(joined_df.columns) - 1 and joined_df.columns[i+1] == "all_particle":
            columns_to_drop.append(col)
        else:
            # extract the words from the file name to use as a prefix
            prefix = "_".join(group_name.split("_")[1:])
            # add the prefix to the column name
            joined_df.rename(columns={col: f"{prefix}_{col}"}, inplace=True)
        # extract the words from the file name to use as a prefix
        prefix = "_".join(group_name.split("_")[1:])
        # add the prefix to the column name
        joined_df.rename(columns={col: f"{prefix}_{col}"}, inplace=True)
        break
    # drop the columns that are followed by the "all_particle" column
    joined_df.drop(columns_to_drop, axis=1, inplace=True)

    # write the joined dataframe to a CSV file
    outdir = folder_path + '/joined'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print("Creating file: " + f"{outdir}/{group_name}_joined.csv")
    #joined_df.to_csv(f"{outdir}/{group_name}_joined.csv", index=False)
