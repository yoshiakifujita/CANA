"""
Filename: ims03_compare_control_sets_Dyn_FVS.py
Author: Yoshiaki Fujita
Date: 2025-09-21
Description:
    This script is to compare the minimum control sets identified with cana attractor_driver_nodes() amd the sets identified with feedback_vertex_set_driver_nodes() and visualize the comparison.
    from either structural graph or effective graphs of multiple models specified in an input csv.

Usage:
    python ims03_compare_control_sets_Dyn_FVS.py [options]

Options:

    -d Path to an input csv table specifying input models (target for comparison)
    -o Path to input / output directory from which input files are read / to which an output file is exported
    -n Number of models to be processed
    -s Start model ID which is described in the imput csv table

Example:
    python ims03_compare_control_sets_Dyn_FVS.py -n 10 -s 1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def read_minset_df(input_file):

    df = pd.read_csv(
        #"df_nodes_edges_sort.csv",
        input_file,
        dtype={
            "ID": str,
            "Network_Name": str,
            "nNodes": int,
            "nEdges": int,
            "nNodes_0.0": int,
            "nEdges_0.0": int,
            "num_min_sets_0.0": int,
            "len_min_sets_0.0": int,
            "min_sets_0.0": str,
            "nNodes_0.2": int,
            "nEdges_0.2": int,
            "num_min_sets_0.2": int,
            "len_min_sets_0.2": int,
            "min_sets_0.2": str,
            "nNodes_0.4": int,
            "nEdges_0.4": int,
            "num_min_sets_0.4": int,
            "len_min_sets_0.4": int,
            "min_sets_0.4": str,
            "nNodes_0.6": int,
            "nEdges_0.6": int,
            "num_min_sets_0.6": int,
            "len_min_sets_0.6": int,
            "min_sets_0.6": str,
            "nNodes_0.8": int,
            "nEdges_0.8": int,
            "num_min_sets_0.8": int,
            "len_min_sets_0.8": int,
            "min_sets_0.8": str
        }
    )
    return df

def df_read_csv(sfile):
    """
    Read input csv table.
    
    Args:
        sfile: Path to an input csv table specifying input models
    
    Returns:
        df: Pandas dataframe of the input table
    """

    df = pd.read_csv(
        #"df_nodes_edges_sort.csv",
        sfile,
        dtype={
            "ID": int,
            "Model_Name": str,
            "Model_File": str,
            "nNodes": int,
            "nEdges": int,
            "nInputNodes": int,
            "nAttractors": int,
            "nLoops": int
        }
    )

    return df

def create_visualization(sModel, e, y_fvs, y_dyn, max_y_lim, out_folder):
    """
    Create a visualization from a comparison.
    
    Args:
        sModel: Model name
        e: Number of edges of each effective graph
        y_fvs: Size of minimum control set identidied as FVS
        y_dyn: Size of minimum control set identidied with dynamic approach
        max_y_lim: Y-axis maximum for number of edges
        out_folder: Path to an export visualization
    Returns:
        out_path: Path to output visualization
    """

    x = np.array([0., 0.2, 0.4, 0.6, 0.8])
    
    fig, ax1 = plt.subplots()
    
    # Plot number of edges (red, left y-axis)
    line1, = ax1.plot(x, e, 's--', color='tab:red', label='number of edges')
    ax1.set_ylabel("Edges", color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xlabel("Effective graph threshold")
    ax1.set_ylim(0, e[0]+5)
    
    # Plot FVS and Dynamic sets (blue & green, right y-axis)
    ax2 = ax1.twinx()
    line2, = ax2.plot(x, y_fvs, 'o-', color='tab:blue', label='size of minimum sets: FVS')
    line3, = ax2.plot(x, y_dyn, 'd-', color='tab:green', label='size of minimum sets: Dynamic')
    
    ax2.set_ylabel("Size", color='black')  # neutral color since multiple series
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max_y_lim)
    
    # Title
    plt.title(sModel)
    
    # Combine legends from both axes
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    out_path = out_folder + "Dyn_FVS_Graph.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path

def compare_control_sets_FVS_vs_Dyn(df, ioPath, nProcess_Models, start_model):
    """
    Compare the minimum control sets identified as FVS and the sets identified with dynamic approach and visualize the comparison .
    
    Args:
        df: A table containing information about input models
        ioPath: Path to input / output directory from which input files are read / to which an output file is exported
        nProcess_Models: Number of edges of each effective graph
        start_model: Size of minimum control set identidied as FVS
    """
    
    for i in range(len(df)):
        m = i + (int(start_model) - 1)
        if "txt" in df.iloc[m]['Model_File']:
            sModel = df.iloc[m]['Model_Name']
            seek_folder = ioPath + sModel + "/"
            print("----- Model -----")
            print(seek_folder)

            filepath_Dyn = os.path.join(seek_folder, "Dyn_effective.csv")
            filepath_FSV = os.path.join(seek_folder, "FVS_effective.csv")
            if os.path.isfile(filepath_Dyn) and os.path.isfile(filepath_FSV):
                print("Start comparison")
                df_FVS = read_minset_df(filepath_FSV)
                df_Dyn = read_minset_df(filepath_Dyn)

                e = np.array([df_FVS.iloc[0]['nEdges_0.0'], df_FVS.iloc[0]['nEdges_0.2'], df_FVS.iloc[0]['nEdges_0.4'], 
                            df_FVS.iloc[0]['nEdges_0.6'], df_FVS.iloc[0]['nEdges_0.8']])
                y_fvs = np.array([df_FVS.iloc[0]['len_min_sets_0.0'], df_FVS.iloc[0]['len_min_sets_0.2'], df_FVS.iloc[0]['len_min_sets_0.4'], 
                                df_FVS.iloc[0]['len_min_sets_0.6'], df_FVS.iloc[0]['len_min_sets_0.8']])
                y_dyn = np.array([df_Dyn.iloc[0]['len_min_sets_0.0'], df_Dyn.iloc[0]['len_min_sets_0.2'], df_Dyn.iloc[0]['len_min_sets_0.4'], 
                                df_Dyn.iloc[0]['len_min_sets_0.6'], df_Dyn.iloc[0]['len_min_sets_0.8']])

                max_len_min_sets = max(y_fvs.max(), y_dyn.max())

                if max_len_min_sets > 10:
                    remainder = max_len_min_sets % 5
                    max_y_lim = max_len_min_sets + (5 - remainder)
                else:
                    max_y_lim = 10
            
                out_path = create_visualization(sModel, e, y_fvs, y_dyn, max_y_lim, seek_folder)
                print("Complete comparison, the result is exported:")
                print(out_path)
                print("")
                
            else:
                print("There are not minimum control set results for this model.")
                print("Skip comparison.")
                print("")

        if i == nProcess_Models-1: break

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Example script with arguments")
    
    # Add arguments
    parser.add_argument("-d", "--arg_d", type=str, default='df_cell_collective_models.csv', help="Input data frame summarize models")
    parser.add_argument("-o", "--arg_o", type=str, default='./results_by_model/', help="Input-Output directory path")
    parser.add_argument("-n", "--arg_n", type=int, default=None, help="Number of models to be processed")
    parser.add_argument("-s", "--arg_s", type=int, default=1, help="Start model ID")

    # Parse arguments
    args = parser.parse_args()

    df = df_read_csv(args.arg_d)

    compare_control_sets_FVS_vs_Dyn(df, args.arg_o, args.arg_n, args.arg_s)

if __name__ == "__main__":
    main()