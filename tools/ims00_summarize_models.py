"""
Filename: ims00_Summarize_models.py
Author: Yoshiaki Fujita
Date: 2025-09-21
Description:
    This script is to summarize models, their number of nodes, edges, input node, attractor, and loops. 

Usage:
    python ims00_Summarize_models.py [options]

Options:

    -d Path to an input csv table specifying input models
    -m Path to input models themselves
    -i input summary table file name
    -o output summary table file name
    -n Number of models to be processed
    -a Attractor calculation decision (it may take long time for complicated models).


Example:
    python ims00_Summarize_models.py -n 10 -s 1 -a skip
"""


import argparse

import os
import pandas as pd
from cana.boolean_network import BooleanNetwork


def read_input_csv(sfile):
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

def summarize_models(df, sModel_path, nProcess_Models, start_model, b_Attractor):
    """
    Summarize models.
    
    Args:
        df: A table containing information about input models
        sModel_path: Path to input models themselves
        nProcess_Models: Number of models to be processed
        start_model: Start model ID which is described in the table
        b_Attractor: Attractor calculation decidion flag
    Returns:
        df: Pandas dataframe of the input table
    """

    for i in range(len(df)):
        m = i + (start_model - 1)
        if "txt" in df.iloc[m]['Model_Name']:

            sfile = sModel_path + df.iloc[i]['Model_Name']
            bn = BooleanNetwork.from_file(sfile, type='cnet')
                
            SG = bn.structural_graph()

            df.iloc[i, df.columns.get_loc('nNodes')] = SG.number_of_nodes()
            df.iloc[i, df.columns.get_loc('nEdges')] = SG.number_of_edges()
            if b_Attractor==False:
                df.iloc[i, df.columns.get_loc('n_Attractors')] = len(bn.attractors())
            
        if i==nProcess_Models-1: break

    return df


def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Example script with arguments")
    
    # Add arguments
    parser.add_argument("-d", "--arg_d", type=str, default='./', help="Input / output directory path")
    parser.add_argument("-m", "--arg_m", type=str, default='./cana/datasets/cell_collective/', help="Input model directory path")
    parser.add_argument("-i", "--arg_i", type=str, default='df_cell_collective_models.csv', help="Input file name")
    parser.add_argument("-o", "--arg_o", type=str, default='df_cell_collective_models.csv', help="Output file name")
    parser.add_argument("-n", "--arg_n", type=int, default=50, help="Number of models to be processed")
    parser.add_argument("-s", "--arg_s", type=int, default=None, help="start model")
    parser.add_argument("-a", "--arg_a", type=bool, default=False, help="If you want to skip attracor calculation, please set True")

    # Parse arguments
    args = parser.parse_args()
    sInput_file = args.arg_d + args.arg_i
    sOutput_file = args.arg_d + args.arg_o
    
    df = read_input_csv(sInput_file)

    df_summary = summarize_models(df, args.arg_m, args.arg_n, args.arg_s, args.arg_a)

    df_summary.to_csv(sOutput_file, index=False)

if __name__ == "__main__":
    main()