"""
database.py

Stores all Tree objects produced during the PruneNet pipeline.

The TreeDatabase acts as a central repository that is passed
between segmentation, IWCP, localization, path planning,
and exporting.
"""

from typing import List
from .tree import Tree
import pandas as pd
from dataclasses import asdict
class TreeDatabase:
    """
    Stores every detected tree in the current pipeline run.
    """

    def __init__(self):

        self.trees: List[Tree] = []

    # ==========================================================
    # Basic Operations
    # ==========================================================

    def add_tree(self, tree: Tree):
        """
        Add a Tree object to the database.
        """
        self.trees.append(tree)

    def add_trees(self, trees: List[Tree]):
        """
        Add multiple Tree objects.
        """
        self.trees.extend(trees)

    def remove_tree(self, tree_id: int):
        """
        Remove a tree by its ID.
        """
        self.trees = [t for t in self.trees if t.tree_id != tree_id]

    def clear(self):
        """
        Remove all trees.
        """
        self.trees.clear()

    # ==========================================================
    # Getters
    # ==========================================================

    def get_all(self) -> List[Tree]:
        """
        Return all trees.
        """
        return self.trees

    def get_tree(self, tree_id: int):
        """
        Return a tree by its ID.
        """
        for tree in self.trees:
            if tree.tree_id == tree_id:
                return tree
        return None

    def get_pruned(self) -> List[Tree]:
        """
        Return only trees requiring pruning.
        """
        return [tree for tree in self.trees if tree.needs_pruning]

    def get_unpruned(self) -> List[Tree]:
        """
        Return trees that do not require pruning.
        """
        return [tree for tree in self.trees if not tree.needs_pruning]

    # ==========================================================
    # Statistics
    # ==========================================================

    def total_trees(self) -> int:
        return len(self.trees)

    def total_pruned(self) -> int:
        return len(self.get_pruned())

    def total_unpruned(self) -> int:
        return len(self.get_unpruned())

    # ==========================================================
    # Iteration Support
    # ==========================================================

    def __iter__(self):
        return iter(self.trees)

    def __len__(self):
        return len(self.trees)

    def __repr__(self):
        return (
            f"TreeDatabase("
            f"total={self.total_trees()}, "
            f"pruned={self.total_pruned()}, "
            f"healthy={self.total_unpruned()})"
        )

    # ==========================================================
    # Export
    # ==========================================================

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all Tree objects into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per tree containing all stored attributes.
        """

        records = []

        for tree in self.trees:

            tree_dict = asdict(tree)

            # Large arrays are not suitable for Excel
            tree_dict.pop("mask", None)

            tree_dict.pop("metadata", None)

            records.append(tree_dict)

        return pd.DataFrame(records)