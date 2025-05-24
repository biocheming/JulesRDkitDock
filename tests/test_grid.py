import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem # For conformer generation for test protein
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from grid import define_grid_around_binding_site, populate_energy_grid, VDW_RADII_SUM_DEFAULTS, DEFAULT_VDW_SUM
from atom_typer import assign_atom_types

class TestGrid(unittest.TestCase):

    def setUp(self):
        """Set up simple protein and ligand for tests."""
        # Ligand: Methane
        self.ligand_mol = Chem.MolFromSmiles("C")
        self.ligand_mol = Chem.AddHs(self.ligand_mol)
        assign_atom_types(self.ligand_mol) # C: C_H, Hs: H
        
        conf_l = Chem.Conformer(self.ligand_mol.GetNumAtoms())
        conf_l.SetAtomPosition(0, np.array([1.0, 1.0, 1.0])) # Carbon atom
        # Add H positions if needed for more detailed tests, simple for now
        if self.ligand_mol.GetNumAtoms() > 1:
            conf_l.SetAtomPosition(1, np.array([1.0, 1.0, 2.09])) 
        self.ligand_mol.AddConformer(conf_l, assignId=True)

        # Protein: Alanine residue (simplified as N-C-C for backbone)
        # Using SMILES for a small peptide like structure
        self.protein_mol = Chem.MolFromSmiles("NCC(=O)N") # Ala-Gly like fragment
        self.protein_mol = Chem.AddHs(self.protein_mol)
        # Generate a 3D conformer for the protein
        AllChem.EmbedMolecule(self.protein_mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(self.protein_mol) # Basic optimization
        assign_atom_types(self.protein_mol)
        
        # Ensure protein has a conformer
        if self.protein_mol.GetNumConformers() == 0:
            # Fallback if embedding failed, create a dummy conformer
            print("Warning: Protein conformer generation failed in setUp. Creating dummy protein conformer.")
            conf_p = Chem.Conformer(self.protein_mol.GetNumAtoms())
            for i in range(self.protein_mol.GetNumAtoms()):
                conf_p.SetAtomPosition(i, np.array([float(i)*0.5, 0.0, 0.0]))
            self.protein_mol.AddConformer(conf_p, assignId=True)


    def test_define_grid_around_binding_site(self):
        """Test the grid definition logic."""
        padding = 2.0
        spacing = 1.0
        
        # The define_grid_around_binding_site expects a ligand molecule,
        # it will internally calculate the ligand center.
        grid_definition = define_grid_around_binding_site(
            self.protein_mol, self.ligand_mol, padding=padding, spacing=spacing
        )

        # Expected ligand center (methane C atom at [1,1,1], Hs might shift slightly if averaged)
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
        """Test energy grid population with a single protein atom."""
        # Protein: Single Carbon atom ('C_H') at origin (0,0,0)
        protein_single_C = Chem.MolFromSmiles("C")
        protein_single_C = Chem.AddHs(protein_single_C) # CH4
        assign_atom_types(protein_single_C) # C is C_H
        
        conf_p_C = Chem.Conformer(protein_single_C.GetNumAtoms())
        conf_p_C.SetAtomPosition(0, np.array([0.0, 0.0, 0.0])) # Carbon atom at origin
        protein_single_C.AddConformer(conf_p_C)

        # Grid: 3x3x3, spacing 1.0, origin (-1,-1,-1).
        # This means grid center index (1,1,1) corresponds to world (0,0,0).
        grid_def = {
            "center": (0.0, 0.0, 0.0), # Not used by populate, but for context
            "dimensions": (3, 3, 3),
            "spacing": 1.0,
            "origin": (-1.0, -1.0, -1.0)
        }
        atom_types_to_probe = ['C_H', 'O_A'] # Probe C_H (idx 0), O_A (idx 1)

        energy_grid, _ = populate_energy_grid(protein_single_C, grid_def, atom_types_to_probe)

        center_grid_idx = (1, 1, 1) # Corresponds to world coords (0,0,0)

        # Protein C atom is 'C_H'. Its Hydrogens are 'H'.
        # At grid point (0,0,0), protein C is at (0,0,0). Distance = 0.
        # populate_energy_grid has: if distance < 1e-4: interaction_energy = 100.0
        # This applies to the interaction of the probe with the protein's Carbon atom.
        # The hydrogens on the protein CH4 are NOT at (0,0,0).
        
        # Expected energy for C_H probe at (0,0,0) interacting with protein C_H at (0,0,0)
        # This should be the high penalty for overlap with the Carbon atom.
        # The hydrogens of the protein CH4 will also contribute attractive/repulsive terms.
        
        # Let's calculate manually for the C_H probe at grid point (1,1,1) [world (0,0,0)]
        # Protein C_H atom at (0,0,0). Probe C_H at (0,0,0). dist=0. Energy = 100.0 (from C-C overlap)
        # Protein H atoms are around (0,0,0), e.g. at (0.63, 0.63, 0.63) assuming tetrahedral.
        # Distance from probe at (0,0,0) to a protein H at (0.63,0.63,0.63) is ~1.09A.
        # d_ij for C_H(probe) - H(protein) is VDW_RADII_SUM_DEFAULTS.get('H-C_H', 2.8) = 2.8
        # For this pair: dist (1.09) < d_ij_ideal (2.8).
        # gauss1 = exp(-(((1.09-2.8)/0.5)**2)) = exp(-((-1.71/0.5)**2)) = exp(-(-3.42)**2) = exp(-11.69) ~ 0
        # gauss2 = exp(-(((1.09-2.8)/2.0)**2)) = exp(-((-1.71/2.0)**2)) = exp(-(-0.855)**2) = exp(-0.731) ~ 0.48
        # Interaction = -0.5*gauss1 + 1.0*(1-gauss2) = ~0 + 1*(1-0.48) = 0.52
        # There are 4 such H atoms. Total from Hs = 4 * 0.52 = 2.08
        # Total for C_H probe = 100.0 (from C) + ~2.08 (from Hs) = ~102.08
        
        self.assertAlmostEqual(energy_grid[center_grid_idx[0], center_grid_idx[1], center_grid_idx[2], 0], 
                               102.08, delta=1.0, msg="Energy for C_H probe at protein C location is incorrect.")

        # Expected energy for O_A probe at (0,0,0) interacting with protein C_H at (0,0,0)
        # Again, 100.0 from overlap with Carbon.
        # d_ij for O_A(probe) - H(protein) is VDW_RADII_SUM_DEFAULTS.get('H-O_A', 2.6) = 2.6
        # dist (1.09) < d_ij_ideal (2.6)
        # gauss1 = exp(-(((1.09-2.6)/0.5)**2)) = exp(-((-1.51/0.5)**2)) = exp(-(-3.02)**2) = exp(-9.12) ~ 0
        # gauss2 = exp(-(((1.09-2.6)/2.0)**2)) = exp(-((-1.51/2.0)**2)) = exp(-(-0.755)**2) = exp(-0.57) ~ 0.565
        # Interaction = -0.5*gauss1 + 1.0*(1-gauss2) = ~0 + 1*(1-0.565) = 0.435
        # Total from Hs = 4 * 0.435 = 1.74
        # Total for O_A probe = 100.0 (from C) + ~1.74 (from Hs) = ~101.74
        self.assertAlmostEqual(energy_grid[center_grid_idx[0], center_grid_idx[1], center_grid_idx[2], 1], 
                               101.74, delta=1.0, msg="Energy for O_A probe at protein C location is incorrect.")

    def test_input_errors_define_grid(self):
        """Test error handling for define_grid_around_binding_site."""
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided."):
            define_grid_around_binding_site(None, self.ligand_mol)
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided."):
            define_grid_around_binding_site(self.protein_mol, None)
        
        mol_no_conf = Chem.MolFromSmiles("C") # No conformer
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must have conformers."):
            define_grid_around_binding_site(self.protein_mol, mol_no_conf)

    def test_input_errors_populate_grid(self):
        """Test error handling for populate_energy_grid."""
        grid_def = define_grid_around_binding_site(self.protein_mol, self.ligand_mol)
        atom_types = ['C_H']
        with self.assertRaisesRegex(ValueError, "Protein molecule must be provided."):
            populate_energy_grid(None, grid_def, atom_types)

        mol_no_conf = Chem.MolFromSmiles("C")
        with self.assertRaisesRegex(ValueError, "Protein molecule must have a conformer."):
            populate_energy_grid(mol_no_conf, grid_def, atom_types)

if __name__ == '__main__':
    unittest.main()
```
