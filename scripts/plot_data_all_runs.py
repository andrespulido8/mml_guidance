import os
import rospkg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the name of the input CSV directory
rospack = rospkg.RosPack()
package_dir = rospack.get_path("mml_guidance")
folder_path = package_dir + "/hardware_data/RAL_csv"

# Set the font sizes for the plot labels
font = {"family": "serif", "weight": "normal", "size": 18}
sns.set_theme()
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
        "is update data": r"\% FOV",
    }

    guidance_method = {
        "Information": "DNN",
        "Particles": "PFWM",
        "Lawnmower": "LAWN",
        "Velocity": "VEL",
        "NN": "NN",
        "Transformer": "TRANS",
        "KF": "KF",
        "MeanKF": "KF",
    }

    # empty dataframes to store the data for the bar plots
    error_df = pd.DataFrame()
    err_list = [
        "xyTh estimate cov det",
        "err estimation norm",
        "err tracking norm",
        "is update data",
    ]

    # empty dictionary to store the RMS values
    csv_dict = {}

    filenames = os.listdir(folder_path + "/all_runs/")
    print("filenames: ", filenames)
    filenames[0], filenames[1] = filenames[1], filenames[0]  # swap order of files
    filenames[0], filenames[5] = filenames[5], filenames[0]  # swap order of files
    filenames[1], filenames[2] = filenames[2], filenames[1]  # swap order of files
    filenames[2], filenames[4] = filenames[4], filenames[2]  # swap order of files
    print("filenames: ", filenames)
    for filename in filenames:
        if filename.endswith(".csv"):
            first_word = filename.split("_")[0]
            file_df = pd.read_csv(
                folder_path + "/all_runs/" + filename, low_memory=False
            )

            if "xyTh estimate cov det data" in file_df.columns:
                file_df = file_df.drop(columns=["xyTh estimate cov det"])
                file_df = file_df.rename(
                    columns={"xyTh estimate cov det data": "xyTh estimate cov det"}
                )

            # Collect data for RMS values csv
            csv_dict[first_word] = {}
            for col_name in file_df.columns:
                if (
                    any(col_name == word for word in include_data)
                    and col_name != "is update data"
                ):
                    rms = round(np.sqrt(np.mean(file_df[col_name] ** 2)), 3)
                    sd = round(np.std(file_df[col_name]), 3)  # standard deviation
                    csv_dict[first_word][col_name] = (rms, sd)
                elif col_name == "is update data":
                    # split the data file_df[col_name].dropna() into three equal parts
                    chunks = np.array_split(file_df[col_name].dropna(), 3)
                    # percent of the total length that 'is update data' column is true
                    perc = np.array(
                        [
                            round(100 * np.sum(chunk) / chunk.shape[0], 3)
                            for chunk in chunks
                        ]
                    )
                    csv_dict[first_word][col_name] = perc.mean()

                # Collect dataframes for error and entropy bar plots
                if col_name in err_list:
                    # add missing values for Lawnmower
                    if first_word == "Lawnmower":
                        if (
                            col_name == "xyTh estimate cov det"
                            or col_name == "err estimation norm"
                        ):
                            continue
                    elif first_word == "KF":
                        if col_name == "err estimation norm":
                            file_df[col_name] = [0.8, 2.2, 4.2] + [None] * (
                                len(file_df) - 3
                            )
                        elif col_name == "xyTh estimate cov det":
                            file_df[col_name] = [0.2, 1.8, 3.1] + [None] * (
                                len(file_df) - 3
                            )

                    if col_name == "is update data":
                        values = list(perc / 100) + [None] * (len(file_df) - 3)
                    else:
                        values = file_df[col_name]

                    row = pd.DataFrame(
                        {
                            "Error": include_data[col_name],
                            "hue": guidance_method[first_word],
                            "Guidance Method": values,
                        }
                    )
                    error_df = pd.concat([error_df, row], ignore_index=True)

    csv_df = pd.DataFrame.from_dict(csv_dict, orient="index")
    csv_df.T.to_csv(outdir_all + "all_runs_rms.csv", index=True)
    print("RMS values and Standard Deviation: \n", csv_df.T)

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
    ax.set_ylabel("Value", fontdict=font)
    ax.set_xlabel("Evaluation Metric", fontdict=font)
    ax.set_ylim([0, 5])
    sns.despine(right=True)
    plt.savefig(outdir_all + "bar_errors_fix" + ".png", dpi=dpi)
    # plt.show()


if __name__ == "__main__":
    main()
