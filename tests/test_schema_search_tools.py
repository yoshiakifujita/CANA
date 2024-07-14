# tests for schema_search_tools.py
from cana.boolean_node import BooleanNode
from automata.schema_search_tools import annihilation_generation_rules, maintenance_rules
from automata.automata_rules import automata_output_list, annihilation_generation


def test_annihilation_generation_maintenance_rules():
    """
    Splitting a rule into their annihilation_generation and maintenance. Then combining them again to check if we get the original rule. 
    """
    anni_gen = {}
    maintenance = {}
    for item in automata_output_list:
        anni_gen[item] = annihilation_generation_rules(automata_output_list[item])
        assert sorted(anni_gen[item]) == sorted(annihilation_generation[item]), ("Annihilation, generation rules are not correct.")

        maintenance[item] = maintenance_rules(automata_output_list[item])

        combo = []
        combo = anni_gen[item] + maintenance[item]
        combo_node = BooleanNode.from_partial_lut(combo)
        parent_node = BooleanNode.from_output_list(automata_output_list[item])

        assert combo_node.outputs == parent_node.outputs, ("Maintenance rules combined with annihilation and generation rules do not return the original rule. "  )