import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import docking
import grid
import atom_typer
import scoring # Though not directly tested, it's used by docking
import conformer_generation

class TestDocking(unittest.TestCase):

    def setUp(self):
        """Set up a simple environment for docking tests."""
        # Protein: Two Carbon atoms
        self.protein_mol = Chem.MolFromSmiles("CC") # Ethane
        self.protein_mol = Chem.AddHs(self.protein_mol)
        atom_typer.assign_atom_types(self.protein_mol) # All C_H, H
        
        conf_p = Chem.Conformer(self.protein_mol.GetNumAtoms())
        conf_p.SetAtomPosition(0, np.array([0.0, 0.0, 0.0])) # C1
        conf_p.SetAtomPosition(1, np.array([1.5, 0.0, 0.0])) # C2
        # Add H positions (simplified)
        for i in range(2, self.protein_mol.GetNumAtoms()):
            conf_p.SetAtomPosition(i, np.array([0.0 + (i-2)*0.3, 1.0, 0.0]))
        self.protein_mol.AddConformer(conf_p, assignId=True)

        # Ligand: Methane (single C_H atom for simplicity in grid tests)
        self.ligand_mol = Chem.MolFromSmiles("C")
        self.ligand_mol = Chem.AddHs(self.ligand_mol) # CH4
        atom_typer.assign_atom_types(self.ligand_mol) # C is C_H, Hs are H
        
        # Generate one conformer for the ligand
        self.ligand_mol = conformer_generation.generate_conformers(self.ligand_mol, num_conformers=1)
        if self.ligand_mol.GetNumConformers() == 0:
            # Fallback if conformer gen failed (should not for methane)
            conf_l = Chem.Conformer(self.ligand_mol.GetNumAtoms())
            conf_l.SetAtomPosition(0, np.array([5.0, 5.0, 5.0])) # Place it somewhere
            self.ligand_mol.AddConformer(conf_l, assignId=True)
            print("Warning: setUp using fallback ligand conformer.")


        # Grid Definition and Energy Grid
        self.atom_types_in_grid = ['C_H', 'O_A', 'UNK'] # Probe types
        
        # Define a simple grid around where we'll place the ligand C atom for testing get_grid_score
        # Ligand C atom for test_get_grid_score will be at (0,0,0) world
        # Grid: 5x5x5, spacing 1.0, origin (-2.5, -2.5, -2.5) -> center point (2,2,2) is at (0,0,0) world
        self.grid_definition = {
            "center": (0.0, 0.0, 0.0), 
            "dimensions": (5, 5, 5),
            "spacing": 1.0,
            "origin": (-2.5, -2.5, -2.5) 
        }
        
        # Dummy energy grid
        self.energy_grid = np.zeros((5,5,5,len(self.atom_types_in_grid)), dtype=np.float32)
        # Make grid point (2,2,2) (world 0,0,0) attractive for C_H probe
        self.attractive_score_val = -5.0
        c_h_probe_index = self.atom_types_in_grid.index('C_H')
        self.energy_grid[2, 2, 2, c_h_probe_index] = self.attractive_score_val
        
        # For perform_docking tests, use a grid defined around the protein
        # This protein is at [0,0,0] and [1.5,0,0]. Ligand is CH4.
        # Let define_grid_around_binding_site use a dummy ligand pos for defining its center
        dummy_lig_for_grid_def = Chem.MolFromSmiles("C") # Single atom
        dummy_lig_for_grid_def.AddConformer(Chem.Conformer(1))
        dummy_lig_for_grid_def.GetConformer(0).SetAtomPosition(0, np.array([0.75, 0.0, 0.0])) # Center between protein atoms

        self.docking_grid_def = grid.define_grid_around_binding_site(
            self.protein_mol, dummy_lig_for_grid_def, padding=3.0, spacing=1.0
        )
        self.docking_energy_grid, _ = grid.populate_energy_grid(
            self.protein_mol, self.docking_grid_def, self.atom_types_in_grid
        )


    def test_get_grid_score(self):
        """Test get_grid_score with an atom at an attractive grid point."""
        test_ligand_conf = Chem.Conformer(self.ligand_mol.GetNumAtoms())
        # Place the first atom (Carbon of CH4) at world (0,0,0)
        # This corresponds to grid index (2,2,2) in self.energy_grid
        test_ligand_conf.SetAtomPosition(0, np.array([0.0, 0.0, 0.0]))
        # Other atoms (hydrogens) are further away, their contribution should be small or zero if grid is sparse
        for i in range(1, self.ligand_mol.GetNumAtoms()): # Position Hs far from C for this specific test
            test_ligand_conf.SetAtomPosition(i, np.array([10.0 + i, 10.0, 10.0]))


        score = docking.get_grid_score(
            self.ligand_mol, # Topology for atom types
            test_ligand_conf, 
            self.grid_definition, 
            self.energy_grid, 
            self.atom_types_in_grid
        )
        # Expected score is primarily from the Carbon atom at the attractive point.
        # Hydrogens are far, should hit zero-energy parts of grid or be out of bounds (high penalty).
        # If Hs are out of bounds, score will be num_Hs * penalty_oob + attractive_score.
        # To simplify, let's test with a single-atom ligand for this.
        
        single_atom_lig = Chem.MolFromSmiles("[C]") # Just Carbon, no Hs
        atom_typer.assign_atom_types(single_atom_lig) # C_H
        single_atom_conf = Chem.Conformer(1)
        single_atom_conf.SetAtomPosition(0, np.array([0.0,0.0,0.0])) # At attractive point

        score_single_atom = docking.get_grid_score(
            single_atom_lig, single_atom_conf, self.grid_definition, self.energy_grid, self.atom_types_in_grid
        )
        self.assertAlmostEqual(score_single_atom, self.attractive_score_val, delta=0.1,
                               msg="Grid score for atom at attractive point is incorrect.")

    def test_get_grid_score_out_of_bounds(self):
        """Test get_grid_score with an atom placed out of grid bounds."""
        test_ligand_conf = Chem.Conformer(self.ligand_mol.GetNumAtoms())
        # Place the first atom (Carbon) far outside the grid
        test_ligand_conf.SetAtomPosition(0, np.array([100.0, 100.0, 100.0]))
        
        # Create a single-atom ligand to avoid complexities with other atoms
        single_atom_lig = Chem.MolFromSmiles("[C]")
        atom_typer.assign_atom_types(single_atom_lig)
        single_atom_conf = Chem.Conformer(1)
        single_atom_conf.SetAtomPosition(0, np.array([100.0, 100.0, 100.0]))


        score = docking.get_grid_score(
            single_atom_lig, 
            single_atom_conf, 
            self.grid_definition, 
            self.energy_grid, 
            self.atom_types_in_grid
        )
        # Expecting penalty_out_of_bounds from docking.get_grid_score (default 1000.0)
        self.assertAlmostEqual(score, 1000.0, delta=0.1,
                               msg="Grid score for out-of-bounds atom is not the defined penalty.")

    def test_perform_docking_smoke_test(self):
        """Basic smoke test for perform_docking without MMFF94s."""
        # Ensure ligand has at least one conformer from setUp
        self.assertGreater(self.ligand_mol.GetNumConformers(), 0, "Ligand needs conformers for docking smoke test.")

        results = docking.perform_docking(
            self.protein_mol, self.ligand_mol, 
            self.docking_energy_grid, self.docking_grid_def, self.atom_types_in_grid,
            n_orientations=1, n_local_search_steps=1, top_n_poses=1, 
            refine_top_poses_with_mmff94s=False
        )
        self.assertIsInstance(results, list, "perform_docking should return a list.")
        if results: # If any pose was found
            self.assertIsInstance(results[0], Chem.Mol, "Result items should be RDKit Mol objects.")
            self.assertTrue(results[0].HasProp("VinardoScore"), "Result molecule should have 'VinardoScore' property.")

    def test_perform_docking_with_mmff94s_refinement_smoke(self):
        """Basic smoke test for perform_docking with MMFF94s refinement."""
        self.assertGreater(self.ligand_mol.GetNumConformers(), 0, "Ligand needs conformers for docking refinement smoke test.")

        results = docking.perform_docking(
            self.protein_mol, self.ligand_mol, 
            self.docking_energy_grid, self.docking_grid_def, self.atom_types_in_grid,
            n_orientations=1, n_local_search_steps=1, top_n_poses=1, 
            refine_top_poses_with_mmff94s=True, mmff94s_max_iterations=5 # Few iterations for speed
        )
        self.assertIsInstance(results, list, "perform_docking with refinement should return a list.")
        if results:
            self.assertIsInstance(results[0], Chem.Mol, "Refined result items should be RDKit Mol objects.")
            # Check for one of the score properties that refinement might add
            has_refined_score = results[0].HasProp("VinardoScore_MMFF94s_Refined") or \
                                results[0].HasProp("VinardoScore_Initial") # if rescoring failed but original kept
            self.assertTrue(has_refined_score, "Refined molecule should have a refinement-related score property.")
            self.assertTrue(results[0].HasProp("PoseRank_MMFF94s_Refined"), "Refined molecule should have refined rank.")


    def test_input_handling_perform_docking(self):
        """Test input error handling for perform_docking."""
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided."):
            docking.perform_docking(None, self.ligand_mol, self.docking_energy_grid, self.docking_grid_def, self.atom_types_in_grid)
        
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided."):
            docking.perform_docking(self.protein_mol, None, self.docking_energy_grid, self.docking_grid_def, self.atom_types_in_grid)

        lig_no_confs = Chem.MolFromSmiles("C") # Methane without conformers
        atom_typer.assign_atom_types(lig_no_confs)
        results_no_confs = docking.perform_docking(self.protein_mol, lig_no_confs, self.docking_energy_grid, self.docking_grid_def, self.atom_types_in_grid)
        self.assertEqual(results_no_confs, [], "Docking with ligand having no conformers should return empty list.")
        
        prot_no_confs = Chem.MolFromSmiles("CC")
        atom_typer.assign_atom_types(prot_no_confs)
        with self.assertRaisesRegex(ValueError, "Protein molecule must have a conformer."):
            docking.perform_docking(prot_no_confs, self.ligand_mol, self.docking_energy_grid, self.docking_grid_def, self.atom_types_in_grid)


if __name__ == '__main__':
    unittest.main()

```
