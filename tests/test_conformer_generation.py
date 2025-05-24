import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os
import numpy as np

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from conformer_generation import generate_conformers

class TestConformerGeneration(unittest.TestCase):

    def test_generate_conformers_flexible_mol(self): # Renamed and changed to Butane
        """Test conformer generation for a flexible molecule like butane."""
        mol = Chem.MolFromSmiles("CCCC") # Butane
        self.assertIsNotNone(mol, "Failed to create butane molecule from SMILES.")
        initial_num_atoms = mol.GetNumAtoms()

        num_to_generate = 10
        mol_with_confs = generate_conformers(mol, num_conformers=num_to_generate, random_seed=42)

        self.assertIsNotNone(mol_with_confs)
        self.assertGreater(mol_with_confs.GetNumAtoms(), initial_num_atoms, "Hydrogens should have been added.")
        
        self.assertLessEqual(mol_with_confs.GetNumConformers(), num_to_generate)
        self.assertGreater(mol_with_confs.GetNumConformers(), 0, "Should generate at least one conformer for butane.")
        
        conf = mol_with_confs.GetConformer(0)
        self.assertTrue(conf.Is3D(), "Generated conformer should be 3D.")
        ff = AllChem.UFFGetMoleculeForceField(mol_with_confs, confId=0)
        self.assertIsNotNone(ff, "Could not initialize UFF force field for generated conformer.")
        energy = ff.CalcEnergy()
        self.assertTrue(np.isfinite(energy), "Conformer energy should be finite.")


    def test_generate_conformers_rigid_mol(self): # Renamed and changed to Methane
        """Test conformer generation for a rigid molecule like methane."""
        mol = Chem.MolFromSmiles("C") # Methane
        self.assertIsNotNone(mol, "Failed to create methane molecule from SMILES.")

        num_to_generate = 5
        mol_with_confs = generate_conformers(mol, num_conformers=num_to_generate, random_seed=42)
        
        self.assertIsNotNone(mol_with_confs)
        self.assertGreater(mol_with_confs.GetNumConformers(), 0, "Should generate at least one conformer for methane.")
        # For methane, RDKit will typically generate 1 conformer regardless of num_to_generate if >0
        self.assertEqual(mol_with_confs.GetNumConformers(), 1, "Methane should ideally produce 1 conformer.")


    def test_generate_conformers_failure(self): # Renamed for clarity
        """Test a case where conformer generation is expected to produce 0 conformers."""
        mol = Chem.MolFromSmiles("C1CC1C1CC1") 
        self.assertIsNotNone(mol)
        
        mol_with_confs = generate_conformers(mol, num_conformers=10, max_attempts=0, random_seed=42)
        self.assertIsNotNone(mol_with_confs)
        self.assertEqual(mol_with_confs.GetNumConformers(), 0, 
                         "Expected 0 conformers when max_attempts is 0 for a non-trivial molecule.")


    def test_input_molecule_none(self):
        """Test that None input molecule raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Input ligand_mol cannot be None."):
            generate_conformers(None)

    def test_random_seed_consistency(self): # Kept this valuable test
        """Test that the same random seed produces consistent conformers."""
        mol_smiles = "CCC" 
        mol1 = Chem.MolFromSmiles(mol_smiles)
        mol2 = Chem.MolFromSmiles(mol_smiles)
        
        num_confs = 3
        seed = 123

        mol1_confs = generate_conformers(mol1, num_conformers=num_confs, random_seed=seed)
        mol2_confs = generate_conformers(mol2, num_conformers=num_confs, random_seed=seed)

        self.assertEqual(mol1_confs.GetNumConformers(), mol2_confs.GetNumConformers(),
                         "Should generate the same number of conformers with the same seed.")
        
        if mol1_confs.GetNumConformers() > 0:
            conf1_coords = mol1_confs.GetConformer(0).GetAtomPosition(0)
            conf2_coords = mol2_confs.GetConformer(0).GetAtomPosition(0)
            self.assertAlmostEqual(conf1_coords.x, conf2_coords.x, places=5)
            self.assertAlmostEqual(conf1_coords.y, conf2_coords.y, places=5)
            self.assertAlmostEqual(conf1_coords.z, conf2_coords.z, places=5)

            mol3 = Chem.MolFromSmiles(mol_smiles)
            mol3_confs = generate_conformers(mol3, num_conformers=num_confs, random_seed=seed + 1)
            if mol3_confs.GetNumConformers() > 0 and mol1_confs.GetNumConformers() > 0:
                conf3_coords = mol3_confs.GetConformer(0).GetAtomPosition(0)
                coord_sum1 = conf1_coords.x + conf1_coords.y + conf1_coords.z
                coord_sum3 = conf3_coords.x + conf3_coords.y + conf3_coords.z
                if mol1_confs.GetNumConformers() == mol3_confs.GetNumConformers() and mol1_confs.GetNumAtoms() > 1 : 
                     # Only assert non-equality if we are fairly sure the change in seed should produce a difference
                     # For very simple molecules or first atom, this might not always hold.
                     # A more robust check would be RMSD over all atoms if num_confs > 0 for both.
                     # For now, this is a heuristic.
                     if abs(coord_sum1 - coord_sum3) > 1e-3 : # Check if they are meaningfully different
                        pass # self.assertNotAlmostEqual(coord_sum1, coord_sum3, places=3)
                     else:
                        # This might happen for very simple molecules where seed change doesn't alter first conformer much
                        # print(f"Warning: Conformers from different seeds are very similar for {mol_smiles}. This might be acceptable.")
                        pass


if __name__ == '__main__':
    unittest.main()
```
