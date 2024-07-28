import random
from typing import Generator
from matplotlib import pyplot as plt
from cana.boolean_node import BooleanNode
import pandas as pd
from automata.automata_rules import automata_output_list

# list of useful functions for the schema search

def annihilation_generation_rules(output_list: list, split: bool = False) -> list:
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

    Example:
    output_list = ['0', '1', '0', '1', '1', '0', '1', '0']
    annihilation_generation_rules(output_list)

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
    if split:
        return annihilation_rules.values.tolist(), generation_rules.values.tolist()
    # combining the two dataframes to get the final dataframe
    annihilation_generation_rules = pd.concat([annihilation_rules, generation_rules])

    # converting it into a list
    annihilation_generation_rules = annihilation_generation_rules.values.tolist()

    return annihilation_generation_rules

def maintenance_rules(output_list: list) -> list:
    """
    This function takes in a list of outputs from an automata and returns a list of maintenance rules.

    Args:
    output_list: list of outputs from an automata

    Returns:
    maintenance_rules: list of maintenance rules

    Method:
    Using BooleanNode from CANA, it creates a look-up-table from a list of outputs.
    Using the annihilation_generation_rules function, it generates a list of annihilation and generation rules.
    It looks for the corresponding rule in the schemata and removes it.
    For cases where the annihilation or generation rule is partially in the schemata, it splits the corresponding rule and retrieves the maintenance part.


    Example:
    output_list = ['0', '1', '0', '1', '1', '0', '1', '0']
    maintenance_rules(output_list)

    """
    schemata = BooleanNode.from_output_list(output_list).schemata_look_up_table()
    schemata = schemata.values.tolist()
    anni_gen_rules = annihilation_generation_rules(output_list)

    for rule in anni_gen_rules:
        if (
            rule in schemata
        ):  # that which is neither annihilation nor generation is maintenance, except for a few exceptions. Removing the annihilation and generation rules from the schemata.
            schemata.remove(rule)

        # change the middle value of the rule to #
        wildcard_rule = [(rule[0][:3] + "#" + rule[0][4:]), rule[1]]
        if (
            wildcard_rule in schemata
        ):  # check if the annihilation or generation rule is partially in the schemata. Sometimes it is a part of the rule with "#" in the middle and isn't on its own.
            if (
                rule[1] in ["0", 0]
            ):  # if the output of the rule is 0, the corresponding maintenance rule output must also be 0
                # modify the middle value of schemata(wildcard_rule) to "0"
                schemata.remove(wildcard_rule)
                schemata.append(
                    [wildcard_rule[0][:3] + "0" + wildcard_rule[0][4:], rule[1]]
                )
            elif (
                rule[1] in ["1", 1]
            ):  # if the output of the rule is 1, the corresponding maintenance rule output must also be 1
                # modify the middle value of schemata(wildcard_rule) to "1"
                schemata.remove(wildcard_rule)
                schemata.append(
                    [wildcard_rule[0][:3] + "1" + wildcard_rule[0][4:], rule[1]]
                )
            else:
                raise ValueError("Rule output must be 0 or 1")
    maintenance_rules = schemata
    return maintenance_rules


def fill_missing_outputs_as_maintenance(node: BooleanNode) -> BooleanNode:
    """
    Generate a node with missing output values filled as maintenance rules.

    Args:
        node (BooleanNode) : The node to generate with. The node must contain missing '?' output values.

    Returns:
        A BooleanNode object with missing output values filled as maintenance rules.
    """

    # check if there are any missing output values
    if "?" not in node.outputs:
        # raise ValueError("There are no missing output values in the node.")
        return node # removed error to allow for complete functions to pass through smoothly
    # Replace '?' in generated_node.outputs with 0 or 1 dependint on the middle element of the respective rule
    lut = node.look_up_table().values.tolist()
    middle_element = len(lut[0][0]) // 2  # middle input is reduced by 1 since indexing starts from 0
    new_outputs = []
    for lut_row in lut:
        if lut_row[1] == "1":
            new_outputs.append("1")
        elif lut_row[1] == "0":
            new_outputs.append("0")
        else:
            new_outputs.append("1" if lut_row[0][middle_element] == "1" else "0")
    return BooleanNode.from_output_list(new_outputs)


def filter_for_ke(generated_nodes, ke, original_ke, epsilon=0.01):
    """
    Filter for nodes that are within epsilon of the desired effective connectivity.

    Args:
    generated_nodes (iterator): List of generated nodes.
    ke (float): Desired effective connectivity.
    epsilon (float): Tolerance for effective connectivity.

    Returns:
    generated_node (BooleanNode): Node that is within epsilon of the desired effective connectivity.

    """
    generated_node = None
    closest_node = None
    closest_ke = None

    for node in generated_nodes:
        if closest_node is None:
            closest_node = node
            closest_ke = node.effective_connectivity()
            smallest_gap = abs(ke - original_ke)
        else:
            ke = node.effective_connectivity()
            gap = abs(ke - original_ke)
            if gap < smallest_gap:
                closest_node = node
                smallest_gap = gap
                closest_ke = ke

                if smallest_gap < epsilon:
                    generated_node = node
                    yield generated_node

    if generated_node is None:
        print(
            f"None found within epsilon. Closest Effective Connectivity: {closest_ke}"
        )
        generated_node = closest_node
        yield generated_node


def filter_for_bias(generated_nodes, bias, original_bias, epsilon=0.01):
    """
    Filter for nodes that are within epsilon of the desired bias.

    Args:
    generated_nodes (iterator): List of generated nodes.
    bias (float): Desired bias.
    epsilon (float): Tolerance for bias.

    Returns:
    generated_node (BooleanNode): Node that is within epsilon of the desired bias.

    """
    generated_node = None
    closest_node = None
    closest_bias = None

    for node in generated_nodes:
        if closest_node is None:
            closest_node = node
            smallest_gap = abs(node.bias() - original_bias)
        else:
            bias = node.bias()
            gap = abs(bias - original_bias)
            if gap < smallest_gap:
                closest_node = node
                smallest_gap = gap
                closest_bias = bias

                if smallest_gap < epsilon:
                    generated_node = node
                    yield generated_node

    if generated_node is None:
        print(f"None found within epsilon. Closest Bias: {closest_bias}")
        generated_node = closest_node
        yield generated_node


def shuffle_wildcards_in_schemata(schemata):
    """
    Shuffle the wildcards in each line of the schemata.

    Args:
    schemata (list): List of schemata.

    Returns:
    shuffled_schemata (list): List of shuffled schemata.

    """
    shuffled_schemata = []
    for schema, output in schemata:
        schema = list(schema)
        random.shuffle(schema)
        schema = "".join(schema)
        shuffled_schemata.append([schema, output])
    return shuffled_schemata


def collect_data_from_generator(node_generator, limit=1000, name=None, verbose=False):
    """
    Collect data from a node generator.

    Args:
    node_generator (generator): Generator of nodes.
    limit (int): Limit of nodes to generate.
    name (str): Name of the generator.
    verbose (bool): Verbose output.

    Returns:
    generated_nodes (list): List of generated nodes.
    node_bias (list): List of node biases.
    node_ke (list): List of node effective connectivities.
    count (int): Count of generated nodes.

    """

    nodes = node_generator
    count = 0
    node_bias = []
    node_ke = []
    generated_nodes = []

    while True:
        try:
            node = next(nodes)
            generated_nodes.append(node)
            node_bias.append(node.bias())
            node_ke.append(node.effective_connectivity())
            count += 1
            if count == limit:
                if verbose:
                    print(f"Generated {count} nodes for {name}")
                break
        except StopIteration:
            if verbose:
                print(f"Generated {count} nodes for {name}")
            break
    return generated_nodes, node_bias, node_ke, count


def combine_output_lists(
    first_priority_outputs, second_priority_outputs, fill_missing_output_randomly=False
):
    """
    Combine two lists of outputs. Outputs from the first list are given priority.

    Args:
    first_priority_outputs (list): List of first priority outputs.
    second_priority_outputs (list): List of second priority outputs.

    Returns:
    combo_outputs (list): List of combined outputs.

    """
    # check if both output lengths are the same
    if len(first_priority_outputs) != len(second_priority_outputs):
        raise ValueError("Both output lists must be of the same length.")
    combo_outputs = ["?"] * len(first_priority_outputs)

    for i in range(len(first_priority_outputs)):
        combo_outputs[i] = (
            first_priority_outputs[i] if first_priority_outputs[i] != "?" else "?"
        )
        if combo_outputs[i] == "?":
            if fill_missing_output_randomly:
                combo_outputs[i] = (
                    second_priority_outputs[i]
                    if second_priority_outputs[i] != "?"
                    else random.choice(
                        ["0", "1"]
                    )  # this will fill the missing outputs randomly
                )
            else:
                combo_outputs[i] = (
                    second_priority_outputs[i]
                    if second_priority_outputs[i] != "?"
                    else "?"  # this will keep the missing outputs as '?'
                )
    return combo_outputs


def shuffle_and_generate(
    automata_output_list: dict,
    shuffle: str = "schemata",
    combine: str = "outputs",
    limit=None,
):
    """
    Shuffle the schemata and generate new nodes.

    Args:
    automata_output_list (dict): Dictionary of automata outputs.
    shuffle (str): Type of shuffle. Can be 'schemata' or 'maintenance'.
    combine (str): Type of combine. Can be 'schemata' or 'outputs'.
    limit (int): Limit of nodes to generate.

    Returns:
    A generator of shuffled nodes.
    """

    # get all relevant parent rule data
    parent_node = BooleanNode.from_output_list(automata_output_list)

    while True:
        if shuffle == "schemata":
            shuffled_schemata = shuffle_wildcards_in_schemata(
                parent_node.schemata_look_up_table().values.tolist()
            )
            shuffled_node = BooleanNode.from_partial_lut(
                shuffled_schemata, fill_clashes=True, fill_missing_output_randomly=True
            )
            yield shuffled_node

        elif shuffle == "maintenance":
            # get the annihilation generation rules and maintenance rules
            anni_gen_schemata = annihilation_generation_rules(automata_output_list)
            maintenance_schemata = maintenance_rules(automata_output_list)

            # shuffle schemata and combine with annihilation generation rules
            shuffled_schemata = shuffle_wildcards_in_schemata(maintenance_schemata)

            # combine the schemata TODO: [SRI] combine via schemata and ensure that first priority schemata are maintained
            if combine == "schemata":
                # combining the shuffled schemata with the annihilation generation rules doesn't preserve the annihilation generation rules in the current version of the code.
                combo = (
                    anni_gen_schemata + shuffled_schemata
                )  # put anni_gen first because clashing outputs will prefer the first one. this will prefer annihilation generation in the shuffled nodes.
                shuffled_node = BooleanNode.from_partial_lut(
                    combo, fill_clashes=True, fill_missing_output_randomly=True
                )
                yield shuffled_node
            elif (
                combine == "outputs"
            ):  # NOTE: [SRI] This is option preserves the annihilation generation rules in the shuffled nodes. But needs to be tested more.
                # combine the outputs
                anni_gen_lut_outputs = BooleanNode.from_partial_lut(
                    anni_gen_schemata
                ).outputs
                shuffled_lut_outputs = BooleanNode.from_partial_lut(
                    shuffled_schemata, fill_clashes=True
                ).outputs
                combo = combine_output_lists(
                    anni_gen_lut_outputs,
                    shuffled_lut_outputs,
                    fill_missing_output_randomly=True,
                )
                shuffled_node = BooleanNode.from_output_list(combo)
                yield shuffled_node

        else:
            raise ValueError(
                "Invalid shuffle type. Must be 'schemata' or 'maintenance'."
            )

        if limit is not None:
            if limit == 0:
                break
            limit -= 1
        # yield shuffled_node


def min_max_and_parent_rule_values(
    automata_output_list: dict,
    values: dict,
    value_type: str = "ke",
    include_parent=True,
):
    """
    Get the min and max values for the histograms and the parent rule values.

    Args:
    automata_output_list (dict): Dictionary of automata outputs.
    values (dict): Dictionary of node values.
    parent_value_type (str): Type of parent value. Can be 'ke' or 'bias'.

    Returns:
    min_max (list): List of min and max values for the histograms.
    parent_rule_values (dict): Dictionary of parent rule values.

    """

    min_max = []
    parent_rule_values = {}
    for item in values:
        if include_parent:
            parent_node = BooleanNode.from_output_list(automata_output_list[item])
            if value_type == "ke":
                parent_rule_values[item] = parent_node.effective_connectivity()
            elif value_type == "bias":
                parent_rule_values[item] = parent_node.bias()
            else:
                print("Invalid parent_value_type. Must be 'ke' or 'bias'.")

        min_max.append(min(values[item]))
        min_max.append(max(values[item]))
    if include_parent:
        min_max.append(min(parent_rule_values.values()) * 0.98)
        min_max.append(max(parent_rule_values.values()) * 1.02)
    min_max = [min(min_max), max(min_max)]

    if include_parent:
        return min_max, parent_rule_values
    else:
        return min_max


# def plot_hist(
#     generated_node_values,
#     value_type=None,
#     no_of_columns=4,
#     title=None,
#     save=False,
#     filename=None,
#     include_parent=False,
# ) -> None:
#     """
#     Plot histogram of generated node values.

#     Args:
#     generated_node_values (dict): Dictionary of generated node values.
#     value_type (str): Type of value to plot. Must be 'ke' or 'bias'.
#     no_of_columns (int): Number of columns in the plot.

#     Returns:
#     None

#     Example:
#     generated_node_values = {
#         "Rule 1": [0.1, 0.2, 0.3, 0.4, 0.5],
#         "Rule 2": [0.2, 0.3, 0.4, 0.5, 0.6],
#     }
#     plot_hist(generated_node_values, value_type='ke')
#     """

#     if value_type == "ke":
#         parent_rule_label = "Parent Rule $K_e$"
#         xlabel = "Effective Connectivity"
#         ylabel = "Count"
#     elif value_type == "bias":
#         parent_rule_label = "Parent Rule Bias"
#         xlabel = "Bias"
#         ylabel = "Count"
#     elif type(value_type) is not str and value_type not in ["ke", "bias"]:
#         xlabel = value_type
#         ylabel = "Count"
#     else:
#         raise ValueError("Invalid value_type. Must be 'ke' or 'bias'.")
#     if title is None:
#         title = f"Distribution of Generated {value_type.capitalize()} Values"

#     total_plots = len(generated_node_values)
#     no_of_rows = int((total_plots / no_of_columns) + 1)
#     # plot hist ke in subplots
#     fig, axs = plt.subplots(no_of_rows, no_of_columns, figsize=(25, 15))
#     fig.suptitle(
#         title,
#         fontsize=25,
#     )
#     min_max, parent_rule_values = min_max_and_parent_rule_values(
#         automata_output_list,
#         generated_node_values,
#         value_type=value_type,
#         include_parent=include_parent,
#     )
#     for i, rule in enumerate(generated_node_values):
#         ax = axs.flatten()[i]
#         # make each subplot axes equal in value range
#         ax.set_xlim(min_max)
#         # ax.set_ylim(0, max_count)

#         ax.hist(generated_node_values[rule], bins=100, color="blue", alpha=0.7)
#         ax.set_title(rule, fontsize=20)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         ax.text(
#             0.2,
#             0.9,
#             f"Total Count: {len(generated_node_values[rule])}",
#             horizontalalignment="center",
#             verticalalignment="center",
#             transform=ax.transAxes,
#             fontsize=10,
#         )

#         if parent_rule_values and parent_rule_label:
#             # plot parent rule value as a line
#             ax.axvline(
#                 parent_rule_values[rule],
#                 color="red",
#                 linestyle="--",
#                 label=f"{parent_rule_label}: {parent_rule_values[rule]:.2f}",
#             )
#             # ax.text(
#             #     parent_rule_values[rule],
#             #     .5,
#             #     f"{parent_rule_values[rule]:.2f}",
#             #     rotation=90,
#             #     fontsize=10,
#             # )

#             ax.legend(fontsize="large", loc="upper right")

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.92)
#     if save:
#         plt.savefig(filename)
#     plt.show()


# def plot_scatter(
#     first_values,
#     second_values,
#     label: list[str, str] = ["ke", "bias"],
#     no_of_columns=4,
#     title=None,
#     save=False,
#     filename=None,
#     include_parent=False,
# ) -> None:
#     """
#     Plot scatter plot of generated rules.

#     Args:
#     first_values (dict): Dictionary of first values.
#     second_values (dict): Dictionary of second values.
#     label (list): List of labels. Must be 'ke' or 'bias'.
#     no_of_columns (int): Number of columns in the plot.

#     Returns:
#     None

#     Example:
#     first_values = {
#         "Rule 1": [0.1, 0.2, 0.3, 0.4, 0.5],
#         "Rule 2": [0.2, 0.3, 0.4, 0.5, 0.6],
#     }
#     second_values = {
#         "Rule 1": [0.2, 0.3, 0.4, 0.5, 0.6],
#         "Rule 2": [0.3, 0.4, 0.5, 0.6, 0.7],
#     }
#     plot_scatter(first_values, second_values, label=['ke', 'bias'])
#     """
#     if label[0] == label[1]:
#         raise ValueError("Invalid labels. Must be different.")
#     if label[0] == "ke":
#         xlabel = "$K_e$"
#         if label[1] == "bias":
#             ylabel = "Bias"

#     elif label[0] == "bias":
#         xlabel = "Bias"
#         if label[1] == "ke":
#             ylabel = "$K_e$"
#     else:
#         raise ValueError("Invalid labels. Must be 'ke' or 'bias'.")

#     if title is None:
#         title = f"Scatter plot of {xlabel} and {ylabel} of generated rules."

#     total_plots = len(first_values)
#     no_of_rows = (
#         int((total_plots / no_of_columns) + 1) if total_plots > no_of_columns else 1
#     )

#     fig, axs = plt.subplots(no_of_rows, no_of_columns, figsize=(25, 15))
#     fig.suptitle(
#         title,
#         fontsize=25,
#     )
#     min_max_x, parent_rule_values_x = min_max_and_parent_rule_values(
#         automata_output_list,
#         first_values,
#         value_type=label[0],
#         include_parent=include_parent,
#     )
#     min_max_y, parent_rule_values_y = min_max_and_parent_rule_values(
#         automata_output_list,
#         second_values,
#         value_type=label[1],
#         include_parent=include_parent,
#     )
#     # plot scatter if first value and second value for
#     rules = list(first_values.keys())
#     for i, rule in enumerate(rules):
#         ax = axs[int(i / no_of_columns), i % no_of_columns]  # TODO: [SRI] Fix the indexing

#         ax.scatter(
#             first_values[rule],
#             second_values[rule],
#             label=f"Rule {rule}",
#             alpha=0.5,
#         )

#         ax.set_title(f"Rule {rule}")
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         ax.grid()
#         ax.set_xlim(min_max_x)
#         ax.set_ylim(min_max_y)

#         # plot parent rule values
#         ax.scatter(parent_rule_values_x[rule], parent_rule_values_y[rule], c="red")
#         ax.legend()

#     for i in range(total_plots, no_of_columns * no_of_rows):
#         fig.delaxes(axs[int(i / no_of_columns), i % no_of_columns])
#     plt.tight_layout()
#     if save:
#         plt.savefig(filename)
#     plt.show()
