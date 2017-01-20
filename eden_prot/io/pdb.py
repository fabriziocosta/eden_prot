#!/usr/bin/env python
"""Provides io for proteins."""

from Bio.PDB import PDBParser


def load(fname):
    """load.

    load structure from PDB database.
    """
    return PDBParser().get_structure('X', fname)
