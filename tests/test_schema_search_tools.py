# tests for schema_search_tools.py
import pytest
from cana.boolean_node import BooleanNode
from automata.schema_search_tools import annihilation_generation_rules, maintenance_rules, fill_missing_outputs_as_maintenance
from automata.automata_rules import automata_output_list, annihilation_generation

def test_annihilation_generation_maintenance_rules():
    """
    Splitting a rule into their annihilation_generation and maintenance. Then combining them again to check if we get the original rule.
    """
    anni_gen = {}
    maintenance = {}
    for item in automata_output_list:
        n = BooleanNode.from_output_list(automata_output_list[item])
        anni_gen[item] = n.get_annihilation_generation_rules()
        assert sorted(anni_gen[item]) == sorted(annihilation_generation[item]), (
            f"{item}: Annihilation, generation rules are not correct."
        )

        maintenance[item] = maintenance_rules(automata_output_list[item])

        combo = []
        combo = anni_gen[item] + maintenance[item]
        combo_node = BooleanNode.from_partial_lut(combo)
        parent_node = BooleanNode.from_output_list(automata_output_list[item])

        assert combo_node.outputs == parent_node.outputs, "Maintenance rules combined with annihilation and generation rules do not return the original rule. "


def test_missing_outputs_as_maintenance():
    """
    Test the fill_missing_outputs_as_maintenance function.
    """
    for item in automata_output_list:
        node1 = BooleanNode.from_output_list(["0", "?", "1", "?", "0", "0", "1", "?"])
        expected1 = ["0", "0", "1", "1", "0", "0", "1", "1"]
        assert fill_missing_outputs_as_maintenance(node1).outputs == expected1

        # Test case: No missing output values
        node3 = BooleanNode.from_output_list(["1", "0", "1", "0", "1", "1", "0", "1"])
        # with pytest.raises(ValueError, match="There are no missing output values in the node."):
        #     fill_missing_outputs_as_maintenance(node3)
        expected3 = ["1", "0", "1", "0", "1", "1", "0", "1"]
        assert fill_missing_outputs_as_maintenance(node3).outputs == expected3

        # Test case: All missing output values
        node4 = BooleanNode.from_output_list(["?", "?", "?", "?", "?", "?", "?", "?"])
        expected4 = ["0", "0", "1", "1", "0", "0", "1", "1"]
        assert fill_missing_outputs_as_maintenance(node4).outputs == expected4