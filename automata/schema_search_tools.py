from cana.boolean_node import BooleanNode
import pandas as pd

# list of useful functions for the schema search


def concatenate(outputlist):
    """
    Mostly for converting output lists into one string for easy comparison with other strings.
    
    Example:
    outputlist = ['0', '1', '0', '1', '1', '0', '1', '0']
    concatenate(outputlist)
    '01011010'
    """
    return "".join(outputlist)


def check_spread_randomness(nodes, iterations=10000):
    """
    Check the spread randomness of the rules in the rule generator. This is to see if generated rules sample the possibility space uniformly.

    Parameters
    ----------
    nodes : generator
        Generator of nodes.
    iterations : int
        Number of iterations to run.

    Returns
    -------
    list
        List of counts.

    How to interpret the results:
    - If the values are close to 0.5, then the rules are spread randomly.
    - If the values are close to 1 or 0, then the rules are not spread randomly.
    - Numbers that are 1 or 0 are fixed values.

    Example:
    check_spread_randomness(nodes, iterations=10000)
    """

    rules = []
    for i in range(iterations):
        rule = next(nodes)
        string = concatenate(rule.outputs)
        rules.append(string)

    # checking the no of 0 and 1 for each index in the string to see if the rules are spread randomply across the space
    count = [0] * 128
    for rule in rules:
        for index, letter in enumerate(rule):
            if letter == "1":
                count[index] += 1

    mean_count = [item / iterations for item in count]
    print(mean_count)  # normalized count of 1s and 0s for each index

    filtered_count = [num for num in mean_count if 0 < num < 1]
    average = sum(filtered_count) / len(filtered_count)
    print(
        f"Total average probability for generated values for rows without fixed values: {average}."
    )  # average of the absolute value of the normalized count of 1s and 0s for each index
    # return count


def check_for_duplicates(nodes, iterations=10000):
    """
    Check for duplicates in the rules generated.
    """

    rules = []
    for i in range(iterations):
        rule = next(nodes)
        string = concatenate(rule.outputs)
        rules.append(string)
    print(f"Total rules generated:{len(rules)}")
    print(f"Unique rules generated:{len(set(rules))}")


def annihilation_generation_rules(output_list):
    """
    This function takes in a list of outputs from an automata and returns a dataframe with the rules that are annihilation and generation rules.

    Args:
    output_list: list of outputs from an automata

    Returns:
    annihilation_generation_rules: dataframe with the rules that are annihilation and generation rules.

    Method:
    Using BooleanNode from CANA, it creates a look-up-table from a list of outputs.
    It generates new lookup tables for annihilation(using logic [RULE & (NOT X_4)]) and generation(using logic [NOT RULE & (X_4)]).
    The rows that output 1 in the new schematas are the annihilations and generations.
    We combine the two dataframes to get the final dataframe. We reassign the annihilation output to 0.
    """

    node = BooleanNode.from_output_list(
        output_list
    )  # creating a node from the output list
    lut = node.look_up_table()

    annihilation_outputs_lut = (  # generates an LUT which is RULE & (NOT X_4), where X_4 is the middle input. the result is 1 for all the rules that annihilate and 0 for all the others.
        ((lut["Out:"] == "0") & (lut["In:"].str[3] == "1"))
        .apply(lambda x: "1" if x else "0")
        .tolist()
    )
    annihilation = BooleanNode.from_output_list(annihilation_outputs_lut)
    temp = annihilation.schemata_look_up_table()  # generating a new schemata from the new LUT to identify the rules that are annihilation
    annihilation_rules = temp[
        temp["Output"] == 1
    ]  # filtering the rules that are annihilation
    annihilation_rules.loc[:, "Output"] = (
        0  # reassigning the output to 0 since it is an annihilation rule
    )

    generation_outputs = (  # generates an LUT which is NOT RULE & (X_4), where X_4 is the middle input. the result is 1 for all the rules that generate and 0 for all the others.
        ((lut["Out:"] == "1") & (lut["In:"].str[3] == "0"))
        .apply(lambda x: "1" if x else "0")
        .tolist()
    )
    generation = BooleanNode.from_output_list(generation_outputs)

    temp = generation.schemata_look_up_table()  # generating a new schemata from the new LUT to identify the rules that are generation
    generation_rules = temp[
        temp["Output"] == 1
    ]  # filtering the rules that are generation
    generation_rules.loc[:, "Output"] = (
        1  # reassigning the output to 1 since it is a generation rule
    )

    # combining the two dataframes to get the final dataframe
    annihilation_generation_rules = pd.concat([annihilation_rules, generation_rules])

    # converting it into a list
    annihilation_generation_rules = annihilation_generation_rules.values.tolist()

    return annihilation_generation_rules