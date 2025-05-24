import unittest
import numpy as np
import math # For math.exp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from atom_typer import assign_atom_types
from scoring import (
    calculate_vinardo_score, VDW_RADII_SUM_DEFAULTS, DEFAULT_VDW_SUM,
    WEIGHT_GAUSS1, WEIGHT_REPLUSION, C1_GAUSS,
    WEIGHT_HBOND, IDEAL_HB_DIST, C_HB_DIST,
    WEIGHT_HYDROPHOBIC, IDEAL_HP_DIST, C_HP_DIST, # Assuming C_HP_DIST is also in scoring
    WEIGHT_ROT
)

class TestScoring(unittest.TestCase):

    def setUp(self):
        """Set up simple protein and ligand for basic tests."""
        # Protein: Single Oxygen atom, no Hs for simplicity in steric tests
        self.protein_O_atom = Chem.MolFromSmiles("[O]")
        # No AddHs here to keep it as a single atom
        self.protein_O_atom.GetAtomWithIdx(0).SetProp("atom_type", "O_A") # Manual type
        conf_p = Chem.Conformer(self.protein_O_atom.GetNumAtoms())
        conf_p.SetAtomPosition(0, np.array([0.0, 0.0, 0.0]))
        self.protein_O_atom.AddConformer(conf_p, assignId=True)

        # Ligand: Single Carbon atom, no Hs for simplicity in steric tests
        self.ligand_C_atom = Chem.MolFromSmiles("[C]")
        # No AddHs
        self.ligand_C_atom.GetAtomWithIdx(0).SetProp("atom_type", "C_H") # Manual type
        conf_l = Chem.Conformer(self.ligand_C_atom.GetNumAtoms())
        conf_l.SetAtomPosition(0, np.array([3.0, 0.0, 0.0]))
        self.ligand_C_atom.AddConformer(conf_l, assignId=True)
        
        # For H-bond test: Protein N-H (Ammonia N_D) and Ligand C=O (Formaldehyde O_A)
        self.protein_NH_mol = Chem.MolFromSmiles("N") 
        self.protein_NH_mol = Chem.AddHs(self.protein_NH_mol)
        assign_atom_types(self.protein_NH_mol) # N should be N_D, Hs are H
        conf_p_nh = Chem.Conformer(self.protein_NH_mol.GetNumAtoms())
        conf_p_nh.SetAtomPosition(0, np.array([0.0, 0.0, 0.0])) # N at origin
        # Position one H for ideal H-bond direction (e.g., along x-axis)
        if self.protein_NH_mol.GetNumAtoms() > 1:
             # Find an H bonded to N
            for atom in self.protein_NH_mol.GetAtoms():
                if atom.GetSymbol() == 'H':
                    conf_p_nh.SetAtomPosition(atom.GetIdx(), np.array([-1.0, 0.0, 0.0])) # Simplistic H position
                    break # Position one H, others can be arbitrary for this test focus
        self.protein_NH_mol.AddConformer(conf_p_nh)

        self.ligand_CO_mol = Chem.MolFromSmiles("C=O")
        self.ligand_CO_mol = Chem.AddHs(self.ligand_CO_mol)
        assign_atom_types(self.ligand_CO_mol) # O should be O_A
        self.lig_O_idx = -1
        for atom in self.ligand_CO_mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                self.lig_O_idx = atom.GetIdx()
                break
        conf_l_co = Chem.Conformer(self.ligand_CO_mol.GetNumAtoms())
        conf_l_co.SetAtomPosition(self.lig_O_idx, np.array([IDEAL_HB_DIST, 0.0, 0.0])) 
        self.ligand_CO_mol.AddConformer(conf_l_co)


    def test_steric_score_attractive(self):
        """Test steric interaction at an ideal attractive distance."""
        protein_atom_type = self.protein_O_atom.GetAtomWithIdx(0).GetProp("atom_type") # O_A
        ligand_atom_type = self.ligand_C_atom.GetAtomWithIdx(0).GetProp("atom_type")   # C_H
        
        d_ij_ideal = VDW_RADII_SUM_DEFAULTS.get(f"{protein_atom_type}-{ligand_atom_type}", 
                       VDW_RADII_SUM_DEFAULTS.get(f"{ligand_atom_type}-{protein_atom_type}", DEFAULT_VDW_SUM))
        
        lig_conf = self.ligand_C_atom.GetConformer()
        lig_conf.SetAtomPosition(0, np.array([d_ij_ideal, 0.0, 0.0]))
        
        # Expected score for this single pair: WEIGHT_GAUSS1 * exp(0) = WEIGHT_GAUSS1
        # Repulsion is 0 because d == d_ij_ideal.
        # Other terms (H-bond, hydrophobic, torsion) are 0 for these single atoms.
        score = calculate_vinardo_score(self.protein_O_atom, self.ligand_C_atom, 0)
        self.assertAlmostEqual(score, WEIGHT_GAUSS1, delta=1e-3,
                               msg="Score at ideal VdW distance should be WEIGHT_GAUSS1.")


    def test_steric_score_repulsive(self):
        """Test steric interaction at a repulsive (clashing) distance."""
        protein_atom_type = self.protein_O_atom.GetAtomWithIdx(0).GetProp("atom_type") # O_A
        ligand_atom_type = self.ligand_C_atom.GetAtomWithIdx(0).GetProp("atom_type")   # C_H
        d_ij_ideal = VDW_RADII_SUM_DEFAULTS.get(f"{protein_atom_type}-{ligand_atom_type}", 
                       VDW_RADII_SUM_DEFAULTS.get(f"{ligand_atom_type}-{protein_atom_type}", DEFAULT_VDW_SUM))

        clash_dist = d_ij_ideal / 2.0 
        # clash_dist = 1.0 # Fixed close distance
        
        lig_conf = self.ligand_C_atom.GetConformer()
        lig_conf.SetAtomPosition(0, np.array([clash_dist, 0.0, 0.0]))
        
        # Expected score for this single pair:
        # gauss1_contrib = WEIGHT_GAUSS1 * exp(-(((clash_dist - d_ij_ideal) / C1_GAUSS)**2))
        # repulsion_contrib = WEIGHT_REPLUSION * (d_ij_ideal - clash_dist)
        # expected_score = gauss1_contrib + repulsion_contrib
        
        d_diff = clash_dist - d_ij_ideal
        gauss1_exp_term = math.exp(-((d_diff / C1_GAUSS)**2))
        gauss1_contrib = WEIGHT_GAUSS1 * gauss1_exp_term
        repulsion_contrib = WEIGHT_REPLUSION * (d_ij_ideal - clash_dist)
        expected_score = gauss1_contrib + repulsion_contrib
        
        score = calculate_vinardo_score(self.protein_O_atom, self.ligand_C_atom, 0)
        self.assertAlmostEqual(score, expected_score, delta=1e-3,
                               msg="Score for repulsive VdW interaction is not as expected.")
        self.assertGreater(score, 0.0, "Repulsive score should be positive.")


    def test_hbond_score(self):
        """Test hydrogen bond contribution at ideal distance."""
        # Using self.protein_NH_mol (N_D) and self.ligand_CO_mol (O_A)
        # Ligand O is placed at IDEAL_HB_DIST (2.8A) from Protein N.
        # Expected H-bond contribution for this N-O pair: WEIGHT_HBOND * exp(0) = WEIGHT_HBOND
        
        # Steric contribution between N and O:
        # N_D atom type, O_A atom type from ligand
        # d_ij_ideal_steric = VDW_RADII_SUM_DEFAULTS.get("N_D-O_A", DEFAULT_VDW_SUM) # Should be ~3.3A
        # d = IDEAL_HB_DIST = 2.8A
        # d < d_ij_ideal_steric, so there's repulsion and modified attraction.
        
        d_NO = IDEAL_HB_DIST # 2.8A
        d_ij_NO_steric = VDW_RADII_SUM_DEFAULTS.get("N_D-O_A", DEFAULT_VDW_SUM) # ~3.3A
        
        gauss1_contrib_NO = WEIGHT_GAUSS1 * math.exp(-(((d_NO - d_ij_NO_steric) / C1_GAUSS)**2))
        repulsion_contrib_NO = 0.0
        if d_NO < d_ij_NO_steric:
            repulsion_contrib_NO = WEIGHT_REPLUSION * (d_ij_NO_steric - d_NO)
        expected_steric_NO = gauss1_contrib_NO + repulsion_contrib_NO
        
        # Interactions with Hydrogens on protein N and ligand C=O also exist.
        # This makes exact calculation complex. The goal is to see if WEIGHT_HBOND is applied.
        # The H-bond term is between the N and O atoms.
        
        score = calculate_vinardo_score(self.protein_NH_mol, self.ligand_CO_mol, 0)
        
        # We expect the score to be roughly: expected_steric_NO + WEIGHT_HBOND 
        # (ignoring other steric interactions with H atoms for this rough check)
        # This is not a perfect isolation of the H-bond term.
        # A more focused test would require a way to get component scores.
        # For now, check that it's significantly more negative than steric alone.
        self.assertLess(score, expected_steric_NO + (WEIGHT_HBOND / 2.0), 
                        "Score with H-bond does not show expected strong favorable contribution.")
        self.assertLess(score, 0.0, "H-bond score should be negative overall.")


    def test_torsional_penalty(self):
        """Test torsional penalty contribution."""
        butane_mol = Chem.MolFromSmiles("CCCC")
        butane_mol = Chem.AddHs(butane_mol)
        assign_atom_types(butane_mol)
        
        conf_butane = Chem.Conformer(butane_mol.GetNumAtoms())
        for i in range(butane_mol.GetNumAtoms()): conf_butane.SetAtomPosition(i, np.array([float(i)*10.0, 0.0, 0.0])) # Spread out
        butane_mol.AddConformer(conf_butane, assignId=True)

        num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(butane_mol)
        self.assertEqual(num_rot_bonds, 1, "Number of rotatable bonds for butane should be 1.")

        # Use a very distant single-atom protein to minimize other interactions
        protein_far_atom = Chem.MolFromSmiles("[He]") # Inert atom, far away
        protein_far_atom.GetAtomWithIdx(0).SetProp("atom_type", "UNK")
        conf_p_far = Chem.Conformer(protein_far_atom.GetNumAtoms())
        conf_p_far.SetAtomPosition(0, np.array([1000.0, 0.0, 0.0])) 
        protein_far_atom.AddConformer(conf_p_far)
        
        score = calculate_vinardo_score(protein_far_atom, butane_mol, num_rot_bonds)
        
        # Expected score = WEIGHT_ROT * num_rot_bonds if other terms are effectively zero.
        # Steric terms for distant atoms should be very close to zero.
        # WEIGHT_GAUSS1 * exp(-large_number) approx 0. Repulsion is 0.
        expected_torsion_penalty = WEIGHT_ROT * num_rot_bonds
        self.assertAlmostEqual(score, expected_torsion_penalty, delta=1e-2, # Small delta for near-zero other terms
                               msg="Score does not primarily reflect the new torsional penalty.")

    def test_missing_conformers(self):
        """Test behavior when molecules lack conformers."""
        protein_no_conf = Chem.MolFromSmiles("[O]") # Use single atom to avoid AddHs adding conformer
        protein_no_conf.GetAtomWithIdx(0).SetProp("atom_type", "O_A")
        
        ligand_no_conf = Chem.MolFromSmiles("[C]")
        ligand_no_conf.GetAtomWithIdx(0).SetProp("atom_type", "C_H")

        # Case 1: Protein has conformer, ligand does not
        score1 = calculate_vinardo_score(self.protein_O_atom, ligand_no_conf, 0)
        self.assertEqual(score1, float('inf'), "Score should be float('inf') if ligand has no conformer.")

        # Case 2: Ligand has conformer, protein does not
        score2 = calculate_vinardo_score(protein_no_conf, self.ligand_C_atom, 0)
        self.assertEqual(score2, float('inf'), "Score should be float('inf') if protein has no conformer.")

        # Case 3: Neither has conformer
        score3 = calculate_vinardo_score(protein_no_conf, ligand_no_conf, 0)
        self.assertEqual(score3, float('inf'), "Score should be float('inf') if neither has conformer.")

    def test_input_molecule_none(self):
        """Test behavior with None inputs."""
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided"):
            calculate_vinardo_score(None, self.ligand_C_atom, 0)
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided"):
            calculate_vinardo_score(self.protein_O_atom, None, 0)
        with self.assertRaisesRegex(ValueError, "Protein and ligand molecules must be provided"):
            calculate_vinardo_score(None, None, 0)


if __name__ == '__main__':
    unittest.main()

```
