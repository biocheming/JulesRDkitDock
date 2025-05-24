import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from atom_typer import assign_atom_types
from scoring import calculate_vinardo_score, VDW_RADII_SUM_DEFAULTS, DEFAULT_VDW_SUM # For VDW checks

class TestScoring(unittest.TestCase):

    def setUp(self):
        """Set up simple protein and ligand for basic tests."""
        # Protein: A single oxygen atom (e.g., like a water oxygen)
        self.protein_O_mol = Chem.MolFromSmiles("O")
        self.protein_O_mol = Chem.AddHs(self.protein_O_mol) # AddHs to [O] gives [OH2]
        assign_atom_types(self.protein_O_mol) # O should be O_A (hydroxyl H rule for O_A)
        
        conf_p = Chem.Conformer(self.protein_O_mol.GetNumAtoms())
        conf_p.SetAtomPosition(0, np.array([0.0, 0.0, 0.0])) # Oxygen atom at origin
        if self.protein_O_mol.GetNumAtoms() > 1: # Hydrogens
            conf_p.SetAtomPosition(1, np.array([0.0, 0.96, -0.2])) 
            conf_p.SetAtomPosition(2, np.array([0.0, -0.48, 0.8]))
        self.protein_O_mol.AddConformer(conf_p, assignId=True)

        # Ligand: A single carbon atom (methane)
        self.ligand_C_mol = Chem.MolFromSmiles("C")
        self.ligand_C_mol = Chem.AddHs(self.ligand_C_mol)
        assign_atom_types(self.ligand_C_mol) # C should be C_H, Hs should be H
        
        conf_l = Chem.Conformer(self.ligand_C_mol.GetNumAtoms())
        conf_l.SetAtomPosition(0, np.array([3.0, 0.0, 0.0])) # Carbon atom initially at 3A distance
        # Add H positions for methane if needed for more complex tests, but not critical for steric point tests
        self.ligand_C_mol.AddConformer(conf_l, assignId=True)
        
        # For H-bond test: Ligand Methanol
        self.methanol_mol = Chem.MolFromSmiles("CO")
        self.methanol_mol = Chem.AddHs(self.methanol_mol)
        assign_atom_types(self.methanol_mol) # C: C_H, O: O_A (due to rule order), H(O): UNK

    def test_steric_score_attractive(self):
        """Test steric interaction at an attractive distance."""
        # Protein O type: O_A (from assign_atom_types on water O-H, rule [o,O;H1])
        # Ligand C type: C_H
        protein_atom_type = self.protein_O_mol.GetAtomWithIdx(0).GetProp("atom_type") # Should be O_A
        ligand_atom_type = self.ligand_C_mol.GetAtomWithIdx(0).GetProp("atom_type")   # Should be C_H
        
        # Expected d_ij for O_A - C_H from scoring.py defaults
        d_ij_ideal = VDW_RADII_SUM_DEFAULTS.get(f"{protein_atom_type}-{ligand_atom_type}", 
                       VDW_RADII_SUM_DEFAULTS.get(f"{ligand_atom_type}-{protein_atom_type}", DEFAULT_VDW_SUM))
        
        # Place ligand carbon exactly at d_ij_ideal from protein oxygen
        lig_conf = self.ligand_C_mol.GetConformer()
        lig_conf.SetAtomPosition(0, np.array([d_ij_ideal, 0.0, 0.0])) # Carbon atom
        
        # Score should be primarily attractive part of gauss1 (-0.5 * exp(-(((d - d_ij)/0.5)**2)))
        # Here d = d_ij, so exponent is 0, exp(0)=1. Score = -0.5 for this pair.
        # Other atoms (H on protein, H on ligand) will also contribute.
        # For simplicity, we expect a negative score.
        score = calculate_vinardo_score(self.protein_O_mol, self.ligand_C_mol, 0)
        self.assertLess(score, 0.0, "Score should be negative for attractive VdW interaction.")
        # A more precise check:
        # gauss1 for the C-O interaction is exp(0) = 1. Contribution = -0.5 * 1 = -0.5
        # Other H-H, O-H, C-H interactions also occur.
        # Expect score to be around -0.5 to -2.0 range for simple methane near water, if no clashes.
        self.assertAlmostEqual(score, -0.5, delta=2.5, msg="Score for ideal VdW not in expected range") # Wider delta due to H's


    def test_steric_score_repulsive(self):
        """Test steric interaction at a repulsive (clashing) distance."""
        lig_conf = self.ligand_C_mol.GetConformer()
        # Place ligand carbon very close to protein oxygen (e.g., 1.0 A)
        clash_dist = 1.0
        lig_conf.SetAtomPosition(0, np.array([clash_dist, 0.0, 0.0]))
        
        score = calculate_vinardo_score(self.protein_O_mol, self.ligand_C_mol, 0)
        # Score should be positive due to repulsion: -0.5*gauss1 + 1.0*(1-gauss2)
        # d_ij is ~3.4. d is 1.0. (d-d_ij) is -2.4.
        # gauss1 = exp(-((-2.4/0.5)**2)) = exp(-(-4.8)**2) = exp(-23.04) ~ 0
        # gauss2 = exp(-((-2.4/2.0)**2)) = exp(-(-1.2)**2) = exp(-1.44) ~ 0.23
        # steric = ~0 + 1*(1-0.23) = 0.77 for the C-O pair. Other H atoms will add more.
        self.assertGreater(score, 0.5, "Score should be positive and significant for repulsive VdW interaction.")

    def test_hbond_score(self):
        """Test hydrogen bond contribution."""
        # Protein: self.protein_O_mol (Water, O at origin, typed as O_A)
        # Ligand: self.methanol_mol (CO, C-O-H)
        # Methanol O is idx 1, H on that O is likely idx 5 (C0,O1,H2,H3,H4(onC), H5(onO))
        # (Check this assumption or find the H atom bonded to O)
        
        methanol_O_idx = -1
        methanol_H_on_O_idx = -1
        for atom in self.methanol_mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                methanol_O_idx = atom.GetIdx()
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'H':
                        methanol_H_on_O_idx = neighbor.GetIdx()
                        break
                break
        self.assertNotEqual(methanol_O_idx, -1, "Could not find Oxygen in methanol for H-bond test.")
        self.assertNotEqual(methanol_H_on_O_idx, -1, "Could not find H on Oxygen in methanol for H-bond test.")

        # Ligand O type (from assign_atom_types on CO): O_A (due to rule priority)
        # This means ligand O (acceptor) with protein O (acceptor) won't form H-bond by current D-A rules.
        # Let's make protein O a donor for this test.
        # For this test, manually set protein O to be O_D, and methanol H(onO) to be part of a donor group
        # This is tricky because assign_atom_types is global.
        # A better H-bond test: Protein N-H (donor) and Ligand C=O (acceptor)
        # For now, use water as protein (O_A) and methanol H (on O, type UNK) as potential part of donor site
        # The H-bond term looks for D-A pairs based on atom_type of heavy atoms.
        # If methanol O is O_A, and protein O is O_A, no H-bond.
        # If methanol O is O_D, and protein O is O_A, then H-bond is possible.
        # The current atom_typer types hydroxyl O as O_A. This makes H-bond testing hard with current types.

        # Let's assume we have a proper Donor (e.g. N_D on protein) and Acceptor (e.g. O_A on ligand)
        # For this test, let's construct a dummy protein N-H and ligand O=C
        # Protein: NH3 (Nitrogen is N_D)
        protein_NH_mol = Chem.MolFromSmiles("N") # Ammonia
        protein_NH_mol = Chem.AddHs(protein_NH_mol)
        assign_atom_types(protein_NH_mol) # N should be N_D
        conf_p_nh = Chem.Conformer(protein_NH_mol.GetNumAtoms())
        conf_p_nh.SetAtomPosition(0, np.array([0.0, 0.0, 0.0])) # N at origin
        protein_NH_mol.AddConformer(conf_p_nh)

        # Ligand: Formaldehyde O=C (Oxygen is O_A)
        ligand_CO_mol = Chem.MolFromSmiles("C=O")
        ligand_CO_mol = Chem.AddHs(ligand_CO_mol)
        assign_atom_types(ligand_CO_mol) # O should be O_A
        
        # Find O atom in formaldehyde
        lig_O_idx = -1
        for atom in ligand_CO_mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                lig_O_idx = atom.GetIdx()
                break
        self.assertNotEqual(lig_O_idx, -1)
        
        conf_l_co = Chem.Conformer(ligand_CO_mol.GetNumAtoms())
        # Position ligand O at 2.8A from protein N (donor-acceptor distance)
        conf_l_co.SetAtomPosition(lig_O_idx, np.array([2.8, 0.0, 0.0])) 
        ligand_CO_mol.AddConformer(conf_l_co)

        # Calculate score. Expect hbond term to be -2.0 * exp(-(((2.8-2.8)/0.4)**2)) = -2.0
        # Other terms (steric) will also contribute.
        score = calculate_vinardo_score(protein_NH_mol, ligand_CO_mol, 0)
        self.assertLess(score, -1.0, "Score should be significantly negative due to H-bond.")
        # It's hard to isolate the H-bond score without modifying calculate_vinardo_score
        # or having more complex molecules where steric/hydrophobic are also optimal.

    def test_torsional_penalty(self):
        """Test torsional penalty contribution."""
        # Ligand: Butane (CH3-CH2-CH2-CH3)
        butane_mol = Chem.MolFromSmiles("CCCC")
        butane_mol = Chem.AddHs(butane_mol)
        assign_atom_types(butane_mol) # All C_H, H
        
        # Add a conformer, position doesn't matter if protein is far
        conf_butane = Chem.Conformer(butane_mol.GetNumAtoms())
        for i in range(butane_mol.GetNumAtoms()): conf_butane.SetAtomPosition(i, np.array([float(i), 0.0, 0.0]))
        butane_mol.AddConformer(conf_butane, assignId=True)

        num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(butane_mol) # Should be 1 for butane
        self.assertEqual(num_rot_bonds, 1, "Number of rotatable bonds for butane should be 1.")

        # Use a very distant protein to minimize other interactions
        protein_far_mol = Chem.MolFromSmiles("O") # Single oxygen
        protein_far_mol = Chem.AddHs(protein_far_mol)
        assign_atom_types(protein_far_mol)
        conf_p_far = Chem.Conformer(protein_far_mol.GetNumAtoms())
        conf_p_far.SetAtomPosition(0, np.array([100.0, 0.0, 0.0])) # Protein oxygen 100A away
        protein_far_mol.AddConformer(conf_p_far)
        
        score = calculate_vinardo_score(protein_far_mol, butane_mol, num_rot_bonds)
        
        # Expected score = 0.35 * num_rot_bonds (if other terms are zero)
        # Steric terms for distant atoms: d_ij is e.g. 3.5. d is ~100.
        # gauss1 = exp(-(((100-3.5)/0.5)**2)) = exp(-((96.5/0.5)**2)) = exp(-(193**2)) which is extremely small.
        # So steric, hbond, hydrophobic should be close to 0.
        expected_torsion_penalty = 0.35 * num_rot_bonds
        self.assertAlmostEqual(score, expected_torsion_penalty, delta=0.1, 
                               msg="Score does not primarily reflect torsional penalty.")

    def test_missing_conformers(self):
        """Test behavior when molecules lack conformers."""
        protein_no_conf = Chem.MolFromSmiles("O")
        # assign_atom_types(protein_no_conf) # Not strictly needed if it's going to fail before types are used
        
        ligand_no_conf = Chem.MolFromSmiles("C")
        # assign_atom_types(ligand_no_conf)

        # Case 1: Protein has conformer, ligand does not
        score1 = calculate_vinardo_score(self.protein_O_mol, ligand_no_conf, 0)
        self.assertEqual(score1, float('inf'), "Score should be float('inf') if ligand has no conformer.")

        # Case 2: Ligand has conformer, protein does not
        score2 = calculate_vinardo_score(protein_no_conf, self.ligand_C_mol, 0)
        self.assertEqual(score2, float('inf'), "Score should be float('inf') if protein has no conformer.")

        # Case 3: Neither has conformer
        score3 = calculate_vinardo_score(protein_no_conf, ligand_no_conf, 0)
        self.assertEqual(score3, float('inf'), "Score should be float('inf') if neither has conformer.")

    def test_input_molecule_none(self):
        """Test behavior with None inputs."""
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided"):
            calculate_vinardo_score(None, self.ligand_C_mol, 0)
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided"):
            calculate_vinardo_score(self.protein_O_mol, None, 0)
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided"):
            calculate_vinardo_score(None, None, 0)


if __name__ == '__main__':
    unittest.main()

```
