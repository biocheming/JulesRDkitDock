import unittest
import numpy as np
import math # For math.exp
from rdkit import Chem
from rdkit.Chem import AllChem # For conformer generation for test protein
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from grid import (
    define_grid_around_binding_site, populate_energy_grid, 
    VDW_RADII_SUM_DEFAULTS, DEFAULT_VDW_SUM,
    WEIGHT_GAUSS1, WEIGHT_REPLUSION, C1_GAUSS # Import new constants
)
from atom_typer import assign_atom_types

class TestGrid(unittest.TestCase):

    def setUp(self):
        """Set up simple protein and ligand for tests."""
        # Ligand: Methane
        self.ligand_mol = Chem.MolFromSmiles("C")
        self.ligand_mol = Chem.AddHs(self.ligand_mol)
        assign_atom_types(self.ligand_mol) 
        
        conf_l = Chem.Conformer(self.ligand_mol.GetNumAtoms())
        conf_l.SetAtomPosition(0, np.array([1.0, 1.0, 1.0])) 
        if self.ligand_mol.GetNumAtoms() > 1: # Simple H positions if any
            for i in range(1, self.ligand_mol.GetNumAtoms()):
                 conf_l.SetAtomPosition(i, np.array([1.0 + (i*0.5), 1.0 + (i*0.5), 1.0 + (i*0.5)]))
        self.ligand_mol.AddConformer(conf_l, assignId=True)

        # Protein for define_grid test: Alanine residue
        self.protein_alanine_mol = Chem.MolFromSmiles("NCC(=O)N") 
        self.protein_alanine_mol = Chem.AddHs(self.protein_alanine_mol)
        if AllChem.EmbedMolecule(self.protein_alanine_mol, AllChem.ETKDG()) == -1:
            print("Warning: Embedding failed for protein_alanine_mol in setUp.") # Should not happen for Ala-Gly
        AllChem.UFFOptimizeMolecule(self.protein_alanine_mol) 
        assign_atom_types(self.protein_alanine_mol)
        
        if self.protein_alanine_mol.GetNumConformers() == 0:
            print("Warning: Protein (alanine) conformer generation failed in setUp. Creating dummy conformer.")
            conf_p_ala = Chem.Conformer(self.protein_alanine_mol.GetNumAtoms())
            for i in range(self.protein_alanine_mol.GetNumAtoms()):
                conf_p_ala.SetAtomPosition(i, np.array([float(i)*0.5, 0.0, 0.0]))
            self.protein_alanine_mol.AddConformer(conf_p_ala, assignId=True)


    def test_define_grid_around_binding_site(self):
        """Test the grid definition logic."""
        padding = 2.0
        spacing = 1.0
        
        # The define_grid_around_binding_site expects a ligand molecule.
        grid_definition = define_grid_around_binding_site(
            self.protein_alanine_mol, self.ligand_mol, padding=padding, spacing=spacing
        )

        # Expected ligand center (methane C atom at [1,1,1], Hs might shift if averaged)
        lig_coords = self.ligand_mol.GetConformer(0).GetPositions()
        expected_lig_center = lig_coords.mean(axis=0)
        
        self.assertIsNotNone(grid_definition)
        self.assertEqual(grid_definition['spacing'], spacing)
        
        # Assert that grid_definition['center'] is close to the ligand center.
        np.testing.assert_array_almost_equal(
            np.array(grid_definition['center']), expected_lig_center, decimal=3,
            err_msg="Grid center is not close to ligand center."
        )

        # Assert that grid_definition['dimensions'] are integers
        dims = grid_definition['dimensions']
        self.assertTrue(all(isinstance(d, (int, np.integer)) for d in dims), "Dimensions should be integers.")
        self.assertEqual(len(dims), 3, "Dimensions should be 3D.")

        # Assert origin calculation: origin = center - (adjusted_span / 2.0)
        # adjusted_span = (dims - 1) * spacing
        adjusted_span = (np.array(dims) - 1) * spacing
        expected_origin = np.array(grid_definition['center']) - (adjusted_span / 2.0)
        np.testing.assert_array_almost_equal(
            np.array(grid_definition['origin']), expected_origin, decimal=3,
            err_msg="Grid origin is not calculated correctly."
        )
        
        # Check if dimensions reflect ligand size + padding / spacing
        # Ligand size (methane is small, ~0 for C atom, Hs extend a bit)
        lig_min = lig_coords.min(axis=0)
        lig_max = lig_coords.max(axis=0)
        lig_span = lig_max - lig_min
        
        # Expected number of points based on ligand span + 2*padding
        expected_num_points = np.maximum(np.ceil((lig_span + 2 * padding) / spacing).astype(int) + 1, 1)
        np.testing.assert_array_equal(np.array(dims), expected_num_points,
                                      "Grid dimensions do not correctly reflect ligand size and padding.")


    def test_populate_energy_grid_simple(self):
        """Test energy grid population with a single protein atom (no Hs for simplicity)."""
        # Protein: Single Carbon atom [C], type 'C_H', at origin (0,0,0)
        protein_single_C = Chem.MolFromSmiles("[C]") 
        protein_single_C.GetAtomWithIdx(0).SetProp("atom_type", "C_H") # Manual type
        
        conf_p_C = Chem.Conformer(protein_single_C.GetNumAtoms())
        conf_p_C.SetAtomPosition(0, np.array([0.0, 0.0, 0.0])) 
        protein_single_C.AddConformer(conf_p_C, assignId=True)

        # Grid: 3x3x3, spacing 1.0, origin (-1,-1,-1).
        # Grid center index (1,1,1) corresponds to world (0,0,0).
        grid_def = {
            "center": (0.0, 0.0, 0.0), # Not used by populate, but for context
            "dimensions": (3, 3, 3),
            "spacing": 1.0,
            "origin": (-1.0, -1.0, -1.0)
        }
        atom_types_to_probe = ['C_H', 'O_A'] # Probe C_H (idx 0), O_A (idx 1)

        energy_grid, _ = populate_energy_grid(protein_single_C, grid_def, atom_types_to_probe)

        center_grid_idx = (1, 1, 1) # Corresponds to world coords (0,0,0) - protein atom location

        # For C_H probe at (0,0,0) interacting with protein C_H at (0,0,0)
        # distance = 0. Grid code uses: WEIGHT_REPLUSION * d_ij_ideal
        d_ij_CH_CH = VDW_RADII_SUM_DEFAULTS.get('C_H-C_H', DEFAULT_VDW_SUM) # 3.5
        expected_energy_ch_probe = WEIGHT_REPLUSION * d_ij_CH_CH # 0.8 * 3.5 = 2.8
        
        self.assertAlmostEqual(energy_grid[center_grid_idx[0], center_grid_idx[1], center_grid_idx[2], 0], 
                               expected_energy_ch_probe, delta=1e-3, 
                               msg="Energy for C_H probe at protein C location is incorrect.")

        # For O_A probe at (0,0,0) interacting with protein C_H at (0,0,0)
        # distance = 0. Grid code uses: WEIGHT_REPLUSION * d_ij_ideal
        d_ij_CH_OA = VDW_RADII_SUM_DEFAULTS.get('C_H-O_A', DEFAULT_VDW_SUM) # 3.4 (or O_A-C_H)
        expected_energy_oa_probe = WEIGHT_REPLUSION * d_ij_CH_OA # 0.8 * 3.4 = 2.72
        
        self.assertAlmostEqual(energy_grid[center_grid_idx[0], center_grid_idx[1], center_grid_idx[2], 1], 
                               expected_energy_oa_probe, delta=1e-3,
                               msg="Energy for O_A probe at protein C location is incorrect.")

    def test_input_errors_define_grid(self):
        """Test error handling for define_grid_around_binding_site."""
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided."):
            define_grid_around_binding_site(None, self.ligand_mol)
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided."):
            define_grid_around_binding_site(self.protein_alanine_mol, None)
        
        mol_no_conf = Chem.MolFromSmiles("C") 
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must have conformers."):
            define_grid_around_binding_site(self.protein_alanine_mol, mol_no_conf)

    def test_input_errors_populate_grid(self):
        """Test error handling for populate_energy_grid."""
        grid_def = define_grid_around_binding_site(self.protein_alanine_mol, self.ligand_mol)
        atom_types = ['C_H']
        with self.assertRaisesRegex(ValueError, "Protein molecule must be provided."):
            populate_energy_grid(None, grid_def, atom_types)

        mol_no_conf = Chem.MolFromSmiles("C")
        with self.assertRaisesRegex(ValueError, "Protein molecule must have a conformer."):
            populate_energy_grid(mol_no_conf, grid_def, atom_types)

if __name__ == '__main__':
    unittest.main()
```
