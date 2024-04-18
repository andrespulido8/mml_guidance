import numpy as np
import pandas as pd
import os


def create_dataset():
    # output velocities if true, else output positions
    is_veloctities = True

    # split the path and file name. Path will be ~/mml_ws/src/mml_guidance
    path = os.path.expanduser("~/mml_ws/src/mml_guidance/hardware_data/")
    # print the files in the directory
    df = pd.read_csv(
        path + "training_data_2024-03-28-00-01-46__slash_robot0_slash_odom.csv"
    )

    # Select every 10th row from the 11th and 12th columns
    reduced_df = df.iloc[::10, 11:13]
    # select every 10th row from the columns named x and y
    # reduced_df = df.iloc[::10, df.columns.get_indexer(['x', 'y'])]
    print("first ten rows: ", reduced_df.head(10))

    if is_veloctities:
        df_time = df.iloc[::10, 0]

        # Calculate the velocities
        velocities = reduced_df.diff().div(df_time.diff() * 1e-9, axis=0).dropna()

        print("diff: ", reduced_df.diff().head(10))
        print("diff time: ", df_time.diff().head(10) * 1e-9)
        print("velocities: ", velocities.head(10))

    # Create a new DataFrame to hold the final dataset
    final_df = pd.DataFrame()

    # Starting from the 10th row
    for i in range(9, len(reduced_df) - 1):
        if is_veloctities:
            # Get the previous 9 x and y values
            prev_values = velocities.iloc[i - 8 : i + 1].values.flatten()
            # Get the next x and y value
            next_values = velocities.iloc[i].values
        else:
            # Get the previous 10 x and y values
            prev_values = reduced_df.iloc[i - 9 : i + 1].values.flatten()
            # Get the next x and y value
            next_values = reduced_df.iloc[i + 1].values

        # Combine the previous and next values and add them as a new row in the final DataFrame
        # use pandas concat instead of append
        final_df = pd.concat(
            [final_df, pd.DataFrame([np.concatenate([prev_values, next_values])])]
        )

    # Save the final DataFrame to a new CSV file
    out_name = "converted_velocities_training_data.csv"
    final_df.to_csv(path + out_name, index=False)
    print("Converted dataset saved to : ", path + out_name)


if __name__ == "__main__":
    create_dataset()
