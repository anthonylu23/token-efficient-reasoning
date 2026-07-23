"""Controlled task-family implementations."""

from .arithmetic import ArithmeticFamily
from .graph_traversal import GraphTraversalFamily

FAMILIES = {
    ArithmeticFamily.name: ArithmeticFamily(),
    GraphTraversalFamily.name: GraphTraversalFamily(),
}

__all__ = ["FAMILIES", "ArithmeticFamily", "GraphTraversalFamily"]
