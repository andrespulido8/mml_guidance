#!/usr/bin/env python3
"""script that opens a csv file, converts each columns to a numpy array,
   takes the root mean square of each column and prints the result"""
import os
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import csv

def np_root_mean_square(array):
    """takes the root mean square of each column of a numpy array"""
    return np.sqrt(np.mean(array**2, axis=0))

def main():
    """main function"""
    with open('data/errors.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader_list = list(reader)
        names = reader_list[0]
        data = np.array(reader_list[1:]).astype(float)
        rms = np_root_mean_square(data)
        print("Data size: ", data.shape[0])
        print("Names: ", names)
        for i in range(len(names)):
            print(names[i], ": ", round(rms[i],2))
        plot_data(data, names)

    now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")  
    # copy the file errors.csv to a new file data_<date and time>.csv
    shutil.copy('data/errors.csv', 'data/data_' + now + '.csv')

    # ask for user input of a string to include better name of row
    row_name = input("Enter a name for the row: ")
    # write the rms values to a new row in the file rms.csv
    with open('data/rms.csv', 'a') as csvfile:
        row_list = [row_name + '_' + now]
        for i in range(len(names)):
            row_list.append(round(rms[i],2))
        writer = csv.writer(csvfile)
        writer.writerow(row_list)

def plot_data(data, names):
    """Plots each column of data in different subplots"""
    fig, ax = plt.subplots(2, 4, figsize=(12, 8))
    for i in range(data.shape[1]):
        ax[i//4, i%4].plot(data[:, i])
        ax[i//4, i%4].set_title(names[i])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #Working_directory = os.getcwd()
    #Print("working directory: ", working_directory)
    main()