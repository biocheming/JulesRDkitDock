import math
import numpy as np # Not strictly used in calculate_vinardo_score but good for general math context
from rdkit import Chem
# from rdkit.Chem import rdMolDescriptors # Not used in this file directly

# --- Vinardo Weights and Parameters ---
# Steric Term
WEIGHT_GAUSS1 = -0.045      # Attractive Gaussian component
WEIGHT_REPLUSION = 0.8       # Repulsive linear component for d < d_ij_ideal
C1_GAUSS = 0.5               # Width for the attractive Gaussian

# Hydrogen Bond Term
WEIGHT_HBOND = -0.600
IDEAL_HB_DIST = 2.8          # Optimal H-bond distance (Angstroms)
C_HB_DIST = 0.4              # Width for H-bond distance decay

# Hydrophobic Term
WEIGHT_HYDROPHOBIC = -0.035
IDEAL_HP_DIST = 4.0          # Optimal hydrophobic contact distance (Angstroms)
C_HP_DIST = 1.5              # Width for hydrophobic distance decay (was 1.0, updated to 1.5)

# Ligand Torsional Penalty
WEIGHT_ROT = 0.05846         # Per rotatable bond

# VdW radii sums (placeholder, ideally more comprehensive)
# (protein_atom_type - ligand_atom_type: sum_of_vdw_radii)
# These are used for d_ij_ideal in steric calculations.
VDW_RADII_SUM_DEFAULTS = {
    'C_H-C_H': 3.5, 'C_H-N_D': 3.6, 'C_H-O_A': 3.4, 'C_H-S_P': 3.9, 'C_H-C_P': 3.5,
    'C_P-C_P': 3.5, 'C_P-N_D': 3.6, 'C_P-O_A': 3.4, 'C_P-S_P': 3.9,
    'N_D-N_D': 3.4, 'N_D-O_A': 3.3, 'N_D-S_P': 3.8, 
    'N_A-N_A': 3.4, 'N_A-O_A': 3.3, 'N_A-S_P': 3.8, 
    'O_A-O_A': 3.2, 'O_A-S_P': 3.7,                 
    'O_D-O_D': 3.2, 'O_D-S_P': 3.7,
    # Common pairs with H (assuming H is a type that would be on protein/ligand)
    'H-C_H': 2.8, 'H-O_A': 2.6, 'H-N_D': 2.7,
    # Add more pairs as needed, or use a more systematic way based on individual radii.
    # Default types from atom_typer:
    # H, C_H, C_P, N_D, N_A, N_P, O_A, O_D, S_P, P_P, F_H, Cl_H, Br_H, I_H, Metal, UNK
}
DEFAULT_VDW_SUM = 3.8  # Default VdW sum if a specific pair is not found


# --- Helper Function for Distance Calculation ---
def get_distance(atom1, atom2, protein_conf, ligand_conf):
    """
    Calculates the Euclidean distance between two atoms.
    Assumes atom1 is from protein and atom2 is from ligand.
    """
    pos1 = protein_conf.GetAtomPosition(atom1.GetIdx())
    pos2 = ligand_conf.GetAtomPosition(atom2.GetIdx())
    return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)

# --- Main Scoring Function ---
def calculate_vinardo_score(protein_mol, ligand_mol, num_rotatable_bonds):
    """
    Calculates a Vinardo-like docking score with updated terms and weights.

    Args:
        protein_mol (rdkit.Chem.rdchem.Mol): RDKit molecule for the protein.
            Assumed to have 'atom_type' property set for each atom.
        ligand_mol (rdkit.Chem.rdchem.Mol): RDKit molecule for the ligand.
            Assumed to have 'atom_type' property set for each atom.
        num_rotatable_bonds (int): Number of rotatable bonds in the ligand.

    Returns:
        float: The calculated Vinardo-like score, or float('inf') if critical errors occur.
    """
    if not protein_mol or not ligand_mol:
        raise ValueError("Protein and ligand molecules must be provided and be valid RDKit molecules.")

    protein_conf = protein_mol.GetConformer()
    ligand_conf = ligand_mol.GetConformer()

    if protein_conf is None or ligand_conf is None:
        return float('inf') 

    # --- Component Scores Initialization ---
    score_steric = 0.0
    score_hbond = 0.0
    score_hydrophobic = 0.0
    
    # --- 1. Steric Term (Updated) ---
    for p_atom in protein_mol.GetAtoms():
        for l_atom in ligand_mol.GetAtoms():
            try:
                p_atom_type = p_atom.GetProp("atom_type")
                l_atom_type = l_atom.GetProp("atom_type")
            except KeyError:
                continue 

            d = get_distance(p_atom, l_atom, protein_conf, ligand_conf)
            
            type_pair_key1 = f"{p_atom_type}-{l_atom_type}"
            type_pair_key2 = f"{l_atom_type}-{p_atom_type}"
            d_ij_ideal = VDW_RADII_SUM_DEFAULTS.get(type_pair_key1, 
                                               VDW_RADII_SUM_DEFAULTS.get(type_pair_key2, DEFAULT_VDW_SUM))

            # Attractive Gaussian term
            gauss1_contrib = WEIGHT_GAUSS1 * math.exp(-(((d - d_ij_ideal) / C1_GAUSS)**2))
            
            # Repulsion term (linear penalty for d < d_ij_ideal)
            repulsion_contrib = 0.0
            if d < d_ij_ideal:
                repulsion_contrib = WEIGHT_REPLUSION * (d_ij_ideal - d)
            
            score_steric += gauss1_contrib + repulsion_contrib
                
    # --- 2. Hydrogen Bond Term (Updated Weight) ---
    donor_types = {'N_D', 'O_D'}      
    acceptor_types = {'N_A', 'O_A'}  
    
    # Ligand Donors to Protein Acceptors
    for l_atom in ligand_mol.GetAtoms():
        try:
            l_atom_type = l_atom.GetProp("atom_type")
        except KeyError: continue
            
        if l_atom_type in donor_types:
            for p_atom in protein_mol.GetAtoms():
                try:
                    p_atom_type = p_atom.GetProp("atom_type")
                except KeyError: continue

                if p_atom_type in acceptor_types:
                    d_DA = get_distance(p_atom, l_atom, protein_conf, ligand_conf)
                    # Distance-dependent term (Gaussian form)
                    # Note: Angular component is a future improvement.
                    energy_dist = math.exp(-(((d_DA - IDEAL_HB_DIST) / C_HB_DIST)**2))
                    score_hbond += WEIGHT_HBOND * energy_dist

    # Ligand Acceptors to Protein Donors
    for l_atom in ligand_mol.GetAtoms():
        try:
            l_atom_type = l_atom.GetProp("atom_type")
        except KeyError: continue

        if l_atom_type in acceptor_types:
            for p_atom in protein_mol.GetAtoms():
                try:
                    p_atom_type = p_atom.GetProp("atom_type")
                except KeyError: continue
                
                if p_atom_type in donor_types:
                    d_DA = get_distance(p_atom, l_atom, protein_conf, ligand_conf)
                    energy_dist = math.exp(-(((d_DA - IDEAL_HB_DIST) / C_HB_DIST)**2))
                    # Note: Angular component is a future improvement.
                    score_hbond += WEIGHT_HBOND * energy_dist
                    
    # --- 3. Hydrophobic Term (Updated Weight & Width) ---
    hydrophobic_types = {'C_H', 'F_H', 'Cl_H', 'Br_H', 'I_H'} 
    
    for l_atom in ligand_mol.GetAtoms():
        try:
            l_atom_type = l_atom.GetProp("atom_type")
        except KeyError: continue

        if l_atom_type in hydrophobic_types:
            for p_atom in protein_mol.GetAtoms():
                try:
                    p_atom_type = p_atom.GetProp("atom_type")
                except KeyError: continue

                if p_atom_type in hydrophobic_types:
                    d_hp = get_distance(p_atom, l_atom, protein_conf, ligand_conf)
                    # Distance-dependent term (Gaussian form)
                    energy_hp = math.exp(-(((d_hp - IDEAL_HP_DIST) / C_HP_DIST)**2))
                    score_hydrophobic += WEIGHT_HYDROPHOBIC * energy_hp

    # --- 4. Ligand Torsional Penalty (Updated Weight) ---
    score_torsion = WEIGHT_ROT * num_rotatable_bonds

    # --- Total Score (Direct Summation) ---
    # Individual term weights are now applied within their calculation.
    total_score = score_steric + score_hbond + score_hydrophobic + score_torsion

    return total_score
```
