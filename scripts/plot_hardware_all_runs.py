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
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 16}
sns.set()
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

def main():
    outdir_all = folder_path + "/all_runs/figures/"
    if not os.path.exists(outdir_all):
        os.mkdir(outdir_all)

    include_data = {
        'err estimation norm':'$e_{estimation}$ [m]', 'err tracking norm':'$e_{tracking}$ [m]', 
        'entropy data':'Entropy', 'n eff particles data':'Effective Number of Particles',
        }

    # empty dataframes to store the data for the bar plots
    error_df = pd.DataFrame()
    entropy_df = pd.DataFrame()
    err_list = ['err estimation norm', 'err tracking norm']

    # empty dictionary to store the RMS values
    csv_dict = {}

    for filename in os.listdir(folder_path + "/all_runs/"):
        if filename.endswith(".csv"):
            first_word = filename.split("_")[0]

            # extract the first word from the file name
            file_df = pd.read_csv(folder_path + "/all_runs/" + filename, low_memory=False)

            # Collect dataframes for error and entropy bar plots
            # error 
            for col in err_list:
                values = file_df[col]

                row = pd.DataFrame({
                    'Guidance Method': first_word,
                    'hue': col,
                    'Error [m]': values
                })
                error_df = pd.concat([error_df, row], ignore_index=True)
            # entropy
            values = file_df['entropy data']
            row = pd.DataFrame({
                'Guidance Method': first_word,
                'Entropy': values
            })
            entropy_df = pd.concat([entropy_df, row], ignore_index=True)

            # Collect data for RMS values csv
            csv_dict[first_word] = {}
            for col_name in file_df.columns:
                if any(col_name == word for word in include_data):
                    # Calculate the RMS value for the column
                    rms = round(np.sqrt(np.mean(file_df[col_name] ** 2)), 3)
                    csv_dict[first_word][col_name] = rms

    csv_df = pd.DataFrame.from_dict(csv_dict, orient='index') 
    csv_df.T.to_csv(outdir_all + "all_runs_rms.csv", index=True)
    print("RMS values: \n", csv_df.T)
    
    ## PLOTS
    # Set the estimator for the error bar plot
    rms_estimator = lambda x: np.sqrt(np.mean(np.square(x))) 
    dpi = 300
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
