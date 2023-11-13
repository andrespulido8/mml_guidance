import os
import rospkg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the name of the input CSV directory
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/csv"

# Set the font sizes for the plot labels
font = {"family": "serif", "weight": "normal", "size": 18}
sns.set()
sns.set_style("white")
sns.set_context("paper", font_scale=2)


def main():
    outdir_all = folder_path + "/all_runs/figures/"
    if not os.path.exists(outdir_all):
        os.mkdir(outdir_all)

    include_data = {
        "err estimation norm": "$ee$ $[m]$",
        "err tracking norm": "$te$ $[m]$",
        "entropy data": "Entropy",
        "n eff particles data": "Effective Number of Particles",
        "xyTh estimate cov det": r"$\det({\Sigma_k})$  $[m^2]$",
    }

    guidance_method = {
        "Information": "MMLEER",
        "Particles": "PFWM",
        "Lawnmower": "LAWN",
    }

    # empty dataframes to store the data for the bar plots
    error_df = pd.DataFrame()
    err_list = ["xyTh estimate cov det", "err estimation norm", "err tracking norm"]

    # empty dictionary to store the RMS values
    csv_dict = {}

    filenames = os.listdir(folder_path + "/all_runs/")
    filenames[0], filenames[2] = filenames[2], filenames[0]  # swap order of files
    for filename in filenames:
        if filename.endswith(".csv"):
            first_word = filename.split("_")[0]
            file_df = pd.read_csv(
                folder_path + "/all_runs/" + filename, low_memory=False
            )

            # Collect dataframes for error and entropy bar plots
            for col in err_list:
                values = file_df[col]

                row = pd.DataFrame(
                    {
                        "Error": include_data[col],
                        "hue": guidance_method[first_word],
                        "Guidance Method": values,
                    }
                )
                error_df = pd.concat([error_df, row], ignore_index=True)

            # Collect data for RMS values csv
            csv_dict[first_word] = {}
            for col_name in file_df.columns:
                if any(col_name == word for word in include_data):
                    rms = round(np.sqrt(np.mean(file_df[col_name] ** 2)), 3)
                    sd = round(np.std(file_df[col_name]), 3)  # standard deviation
                    csv_dict[first_word][col_name] = (rms, sd)
                elif col_name == "is update data":
                    # percent of the total length that 'is update data' column is true
                    perc = round(
                        100
                        * np.sum(file_df[col_name].dropna())
                        / len(file_df[col_name].dropna()),
                        3,
                    )
                    csv_dict[first_word][col_name] = perc

    csv_df = pd.DataFrame.from_dict(csv_dict, orient="index")
    csv_df.T.to_csv(outdir_all + "all_runs_rms.csv", index=True)
    print("RMS values: \n", csv_df.T)

    ## PLOTS
    rms_estimator = lambda x: np.sqrt(np.mean(np.square(x)))
    dpi = 300
    # BAR
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(
        y="Guidance Method",
        x="Error",
        data=error_df,
        ax=ax,
        hue="hue",
        estimator=rms_estimator,
        capsize=0.1,
        errorbar="sd",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Guidance Methods", fontsize=16, title_fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=font)
    ax.set_ylabel("Error value", fontdict=font)
    ax.set_xlabel("Error Type", fontdict=font)
    sns.despine(right=True)
    plt.savefig(outdir_all + "bar_errors_fix" + ".png", dpi=dpi)
    # plt.show()


if __name__ == "__main__":
    main()
