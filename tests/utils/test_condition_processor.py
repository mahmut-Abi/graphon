from graphon.runtime import VariablePool
from graphon.utils.condition.entities import Condition
from graphon.utils.condition.processor import ConditionProcessor


def test_number_formatting():
    condition_processor = ConditionProcessor()
    variable_pool = VariablePool()
    variable_pool.add(["test_node_id", "zone"], 0)
    variable_pool.add(["test_node_id", "one"], 1)
    variable_pool.add(["test_node_id", "one_one"], 1.1)

    assert (
        condition_processor.process_conditions(
            variable_pool=variable_pool,
            conditions=[
                Condition(
                    variable_selector=["test_node_id", "zone"],
                    comparison_operator="≤",
                    value="0.95",
                )
            ],
            operator="or",
        ).final_result
        is True
    )

    assert (
        condition_processor.process_conditions(
            variable_pool=variable_pool,
            conditions=[
                Condition(
                    variable_selector=["test_node_id", "one"],
                    comparison_operator="≥",
                    value="0.95",
                )
            ],
            operator="or",
        ).final_result
        is True
    )

    assert (
        condition_processor.process_conditions(
            variable_pool=variable_pool,
            conditions=[
                Condition(
                    variable_selector=["test_node_id", "one_one"],
                    comparison_operator="≥",
                    value="0.95",
                )
            ],
            operator="or",
        ).final_result
        is True
    )

    assert (
        condition_processor.process_conditions(
            variable_pool=variable_pool,
            conditions=[
                Condition(
                    variable_selector=["test_node_id", "one_one"],
                    comparison_operator=">",
                    value="0",
                )
            ],
            operator="or",
        ).final_result
        is True
    )
