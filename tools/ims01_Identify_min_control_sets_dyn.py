"""
Filename: ms01_Identify_min_control_sets_dyn.py
Author: Yoshiaki Fujita
Date: 2025-09-21
Description:
    This script is to identify the minimum control sets with cana attractor_driver_nodes, 
    from either structural graph or effective graphs of multiple models specified in an input csv.

Usage:
    python ms01_Identify_min_control_sets_dyn.py [options]

Options:

    -d Path to an input csv table specifying input models
    -m Path to input models themselves
    -o Path to output directory under which a directory for each model is created
    -n Number of models to be processed
    -x Mininum number of driver nodes to search
    -y Maximum number of driver nodes to search
    -s Start model ID which is described in the imput csv table
    -g Structure of graph from which minimum control sets are identified, either structural or effective

Example:
    python ms01_Identify_min_control_sets_dyn.py -n 10 -s 1 -g EG
"""

# Imports
import numpy as np
import pandas as pd
import cana
import networkx as nx
from cana.boolean_network import BooleanNetwork
from datetime import datetime
import argparse
import itertools
import ast
import os

def prune_edge(bn, target_idx, regulator_idx):
    """
    Prune a regulator edge (regulator_idx -> target_idx) from a CANA BooleanNetwork object.
    
    Args:
        bn:: Boolean network object
        target_idx: Index of the target node in bn.logic
        regulator_idx: Index of the regulator node to remove
    
    Returns:
        Updated BooleanNetwork object (in place)
    """
    node_logic = bn.logic[target_idx]
    inputs = node_logic['in']
    outputs = node_logic['out']

    if regulator_idx not in inputs:
        print(f"No edge {regulator_idx}->{target_idx} to prune.")
        return bn
    
    # Find position of regulator in inputs
    pos = inputs.index(regulator_idx)
    remaining_inputs = [i for i in inputs if i != regulator_idx]
    num_remaining = len(remaining_inputs)
    
    # New truth table length = 2^(num_remaining)
    new_out = []
    
    # Iterate over all assignments of remaining inputs
    for assignment in itertools.product([0,1], repeat=num_remaining):
        # Collect outputs where removed regulator is 0 or 1
        outs_for_both = []
        for reg_val in [0,1]:
            # Reconstruct full assignment including regulator
            full_assignment = list(assignment)
            full_assignment.insert(pos, reg_val)
            # Convert assignment -> index
            idx = int("".join(map(str,full_assignment)), 2)
            outs_for_both.append(outputs[idx])
        
        # Decide new output (here we take OR, i.e. if either regulator state gave 1)
        # Other choices: AND, majority, or just take first. This is a design choice!
        new_val = 1 if any(outs_for_both) else 0
        new_out.append(new_val)
    
    # Update logic
    node_logic['in'] = remaining_inputs
    node_logic['out'] = new_out
    bn.logic[target_idx] = node_logic
    return bn

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

def df_tem_create(threshold_lists):
    """
    Create output pandas dataframe template.
    
    Args:
        threshold_lists: Threshold lists for effective graph 
    
    Returns:
        df_tmp: Pandas dataframe template
    """
    fixed_cols = ['Model','nNodes_SG', 'nEdges_SG']
    dynamic_cols =  [f"{letter}{v}" for v in threshold_lists for letter in ['nNodes_', 'nEdges_', 'num_min_sets_','len_min_sets_','min_sets_']]

    all_columns = fixed_cols + dynamic_cols

    # Define dtypes (optional, but safer)
    dtypes = {col: float for col in all_columns}
    dtypes['Model'] = str
    dtypes['nNodes_SG'] = float
    dtypes['nEdges_SG'] = float

    # Create empty DataFrame
    df_tmp = pd.DataFrame({col: pd.Series(dtype=dtypes[col]) for col in all_columns})
    return df_tmp

def export_results(threshold_lists, sModel, nNodes, nEdges, result_list, output_file):
    """
    Export identified minimum control set results.
    
    Args:
        threshold_lists: Threshold lists for effective graph 
        sModel: Model name
        nNodes: Number of nodes of structural graph of the model
        nEdges: Number of edges of structural graph of the model
        result_list: Identified minimum set results
        output_file: Path to export the results
    """
    df_out = df_tem_create(threshold_lists)

    row_dict = {'Model': sModel, 'nNodes_SG': nNodes, 'nEdges_SG': nEdges}
    p=0

    for v in threshold_lists:
        row_dict[f"nNodes_{v}"] = result_list[p][0]
        row_dict[f"nEdges_{v}"] = result_list[p][1]    
        row_dict[f"num_min_sets_{v}"] = result_list[p][2]
        row_dict[f"len_min_sets_{v}"] = result_list[p][3]
        row_dict[f"min_sets_{v}"] = result_list[p][4]
        p=p+1
            
    df_out = pd.concat([df_out, pd.DataFrame([row_dict])], ignore_index=True)
        
    df_out.to_csv(output_file, index=False)

def create_output_directory(df, sOut_Path, nProcess_Models, start_model):
    """
    Create directories to store identified minimum control set result by a model.
    
    Args:
        df: A table containing information about input models
        sOut_Path: Path to output directory under which a directory for each model is created
        nProcess_Models: Number of models to be processed
        start_model: Start model ID which is described in the table
    """

    for i in range(len(df)):
        #try:
        m = i + (start_model - 1)
        sModel = df.iloc[m]['Model_Name']    
        new_folder_parth = sOut_Path + sModel

        os.makedirs(new_folder_parth, exist_ok=True)

        if i==nProcess_Models-1: break
            
def identify_min_control_sets_SG(df, sModel_Path, sOut_Path, nProcess_Models, min_nodes, max_nodes, start_model):
    """
    Identify minimum control sets from structural graphs of specific models
    
    Args:
        df: A table containing information about input models
        sModel_Path: Path to input models themselves
        sOut_Path: Path to output directory under which a directory for each model exists
        nProcess_Models: Number of models to be processed
        min_nodes: Mininum number of driver nodes to search
        max_nodes: Maximum number of driver nodes to search
        start_model: Start model ID which is described in the table
    """
    threshold_lists = [0.0, 0.2, 0.4, 0.6, 0.8] # To follow the same output format in EG case
    sOutFileName = "/Dyn_structural.csv"
    
    for i in range(len(df)):
        #try:
        m = i + (start_model - 1)

        # Create boolean network model       
        sfile = sModel_Path + df.iloc[m]['Model_File']
        print("----- Model -----")
        print(sfile)
        print("Start identification from structual graph")
        
        bn = BooleanNetwork.from_file(sfile, type='cnet')
        SG = bn.structural_graph()
        nNodes_SG = SG.number_of_nodes()
        nEdges_SG = SG.number_of_edges()

        print("Graph Structure")
        print(f"Number of nodes: {nNodes_SG}")
        print(f"Number of edges: {nEdges_SG}")
        
        # Identify minimum control sets
        min_sets = bn.attractor_driver_nodes(min_dvs=min_nodes, max_dvs=max_nodes)

        # Organize results for output
        result_list = []
        if len(min_sets)>0:
            if len(min_sets[0])>0: # Get results in an expected manner
                tmp_result = [nNodes_SG, nEdges_SG, len(min_sets), len(min_sets[0]), min_sets]
                result_list.append(tmp_result)
                print("Complete identification") 
                print(f"Size of identified set: {len(min_sets[0])}") 
            else:
                tmp_result = [nNodes_SG, nEdges_SG, len(min_sets), 0, min_sets]
                print("Fail identification with unexpected result.")
        else:
            tmp_result = [nNodes_SG, nEdges_SG, 0, 0, min_sets]
            print("Fail identification with unexpected result.")

        tmp_result = [0, 0, 0, 0, '[]']
        for r in range(len(threshold_lists)):
            result_list.append(tmp_result)

        # Export results
        sModel = df.iloc[m]['Model_Name']      
        new_folder_parth = sOut_Path + sModel
        output_file = new_folder_parth + sOutFileName

        export_results(threshold_lists, sModel, nNodes_SG, nEdges_SG, result_list, output_file)

        if i==nProcess_Models-1: break

def identify_min_control_sets_EG(df, sModel_Path, sOut_Path, nProcess_Models, min_nodes, max_nodes, start_model):
    """
    Identify minimum control sets from effective graphs of specific models
    
    Args:
        df: A table containing information about input models
        sModel_Path: Path to input models themselves
        sOut_Path: Path to output directory under which a directory for each model exists
        nProcess_Models: Number of models to be processed
        min_nodes: Mininum number of driver nodes to search
        max_nodes: Maximum number of driver nodes to search
        start_model: Start model ID which is described in the table
    """
    threshold_lists = [0.0, 0.2, 0.4, 0.6, 0.8]

    sOutFileName = "/Dyn_effective.csv"

    for i in range(len(df)):

        #try:
        m = i + (start_model - 1)

        # Create boolean network model       
        sfile = sModel_Path + df.iloc[m]['Model_File']
        print("----- Model -----")
        print(sfile)

        bn = BooleanNetwork.from_file(sfile, type='cnet')
        SG = bn.structural_graph()
        nNodes_SG = SG.number_of_nodes()
        nEdges_SG = SG.number_of_edges()
        
        # Generate effective graph to remove edges by each threshold
        EG = bn.effective_graph(threshold=None)
        edges_dict = {(u, v): attr for u, v, attr in EG.edges(data=True)}

        # Create FVS min sets list 
        # This version does not implement FVS heuristics
        #f_str = df.iloc[i]['min_sets_nodes']
        #fvs_nodes = set(ast.literal_eval(f_str))  # safely converts string to set of ints

        min_dvs = min_nodes
        max_dvs = max_nodes
        result_list=[]

        # Identify minimum control set from graphs whose edges are filtered by effective connectivity threshold
        print("Start identification from effective graphs")
        for k in range(len(threshold_lists)):
            
            # Generate boolean network object from which edges lower than threshold are removed.
            bn_e = BooleanNetwork.from_file(sfile, type='cnet')   
            for (regulator_node, target_node), attr in edges_dict.items():
                weight = attr.get('weight', None)  # get weight, default None if missing
                if weight < threshold_lists[k]:
                    bn_e = prune_edge(bn_e, target_node, regulator_node)
                    
            SG_e = bn_e.structural_graph()
            nNodes_e = SG_e.number_of_nodes()
            nEdges_e = SG_e.number_of_edges()

            print(f"Edge weight threshold: {threshold_lists[k]}")
            print("Graph Structure")
            print(f"Number of nodes: {nNodes_e}")
            print(f"Number of edges: {nEdges_e}")      

            # Identify minimum control sets
            min_sets = bn_e.attractor_driver_nodes(min_dvs=min_dvs, max_dvs=max_dvs)

            if len(min_sets)>0:
                if len(min_sets[0])>0:
                    tmp_result = [nNodes_e, nEdges_e, len(min_sets), len(min_sets[0]), min_sets]
                    result_list.append(tmp_result)
                    min_dvs = len(min_sets[0])
                    max_dvs = min_dvs + 1
                    b_break=False
                else:
                    tmp_result = [nNodes_e, nEdges_e, len(min_sets), 0, min_sets]
                    result_list.append(tmp_result)
                    b_break=True
            else:
                tmp_result = [nNodes_e, nEdges_e, 0, 0, min_sets]
                result_list.append(tmp_result)
                b_break=True
            
            if b_break==True:
                print("Stop identification due to unexpected result.")
                tmp_result = [0, 0, 0, 0, '[]']
                for r in range(0, (5-len(result_list))):
                    result_list.append(tmp_result)
                break

            print("Complete identification") 
            print(f"Size of identified set: {len(min_sets[0])}") 
        
        # Export results
        sModel = df.iloc[m]['Model_Name']     
        new_folder_parth = sOut_Path + sModel
        output_file = new_folder_parth + sOutFileName

        export_results(threshold_lists, sModel, nNodes_SG, nEdges_SG, result_list, output_file)

        if i==nProcess_Models-1: break

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="Example script with arguments")
    
    # Add arguments
    parser.add_argument("-d", "--arg_d", type=str, default='df_cell_collective_models.csv', help="Input data frame summarize models")
    parser.add_argument("-m", "--arg_m", type=str, default='./cana/datasets/cell_collective/', help="BN models source directory")
    parser.add_argument("-o", "--arg_o", type=str, default='./results_by_model/', help="Input-Output directory path")
    parser.add_argument("-n", "--arg_n", type=int, default=50, help="Number of models to be processed")
    parser.add_argument("-x", "--arg_x", type=int, default=1, help="minimum number of sets")
    parser.add_argument("-y", "--arg_y", type=int, default=6, help="maxmum number of sets")
    parser.add_argument("-s", "--arg_s", type=int, default=1, help="start model")
    parser.add_argument("-g", "--arg_g", type=str, default="SG", help="Identify the sets with Interaction Graph or Effective graphs")
    

    # Parse arguments
    args = parser.parse_args()

    df = df_read_csv(args.arg_d)

    create_output_directory(df, args.arg_o, args.arg_n, args.arg_s)

    if args.arg_g == 'SG':
        print(f"Start identification from SG of {args.arg_n} models")
        print("")
        identify_min_control_sets_SG(df, args.arg_m, args.arg_o, args.arg_n, args.arg_x, args.arg_y, args.arg_s)
        print("")
        print(f"Finish identification from SG of {args.arg_n} models")
    elif args.arg_g == 'EG':
        print(f"Start identification from EG of {args.arg_n} models")
        print("")
        identify_min_control_sets_EG(df, args.arg_m, args.arg_o, args.arg_n, args.arg_x, args.arg_y, args.arg_s)
        print("")
        print(f"Finish identification from EG of {args.arg_n} models")
    else:
        print("Wrong setting")
        print("Please set 'SG' or 'EG' with -g irgument")
        print("Terminate processing")       

if __name__ == "__main__":
    main()
