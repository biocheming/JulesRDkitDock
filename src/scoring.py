import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors # For num_rotatable_bonds if not passed directly

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
    Calculates a simplified Vinardo-like docking score.

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
        # This check is more of a safeguard; calling code should ensure valid molecules.
        # print("Error: Protein and/or ligand molecule is None.")
        raise ValueError("Protein and ligand molecules must be provided and be valid RDKit molecules.")

    protein_conf = protein_mol.GetConformer()
    ligand_conf = ligand_mol.GetConformer()

    if protein_conf is None or ligand_conf is None:
        # Molecules for scoring must have 3D coordinates.
        # print("Warning: Protein or ligand molecule is missing a conformer. Cannot calculate score.")
        # Depending on strictness, could raise error or return a score indicating failure.
        return float('inf') 

    # --- Component Scores Initialization ---
    score_steric = 0.0
    score_hbond = 0.0
    score_hydrophobic = 0.0
    
    # --- 1. Steric Term (Gaussian-based) ---
    # Placeholder VdW radii sums (in Angstroms).
    # These would typically be derived from individual atom VdW radii based on their types.
    vdw_sum_radii = {
        'C_H-C_H': 3.5, 'C_H-N_D': 3.6, 'C_H-O_A': 3.4, 'C_H-S_P': 3.9, 'C_H-C_P': 3.5,
        'C_P-C_P': 3.5, 'C_P-N_D': 3.6, 'C_P-O_A': 3.4, 'C_P-S_P': 3.9,
        'N_D-N_D': 3.4, 'N_D-O_A': 3.3, 'N_D-S_P': 3.8, # Example, adjust as needed
        'N_A-N_A': 3.4, 'N_A-O_A': 3.3, 'N_A-S_P': 3.8, # Example
        'O_A-O_A': 3.2, 'O_A-S_P': 3.7,                 # Example
        'O_D-O_D': 3.2, 'O_D-S_P': 3.7,                 # Example
        # Symmetric pairs are not strictly needed due to lookup logic but can be explicit.
    }
    default_vdw_sum = 3.8  # Default VdW sum for type pairs not in the dictionary

    for p_atom in protein_mol.GetAtoms():
        for l_atom in ligand_mol.GetAtoms():
            try:
                p_atom_type = p_atom.GetProp("atom_type")
                l_atom_type = l_atom.GetProp("atom_type")
            except KeyError:
                # This indicates atom typing was not performed or was incomplete.
                # print(f"Warning: 'atom_type' not found for P_idx {p_atom.GetIdx()} ({p_atom.GetSymbol()}) or L_idx {l_atom.GetIdx()} ({l_atom.GetSymbol()}). Skipping pair for steric.")
                continue 

            d = get_distance(p_atom, l_atom, protein_conf, ligand_conf)
            
            # Lookup d_ij: check type1-type2, then type2-type1, then default
            type_pair_key1 = f"{p_atom_type}-{l_atom_type}"
            type_pair_key2 = f"{l_atom_type}-{p_atom_type}"
            d_ij = vdw_sum_radii.get(type_pair_key1, vdw_sum_radii.get(type_pair_key2, default_vdw_sum))

            gauss1 = math.exp(-(((d - d_ij) / 0.5)**2))  # Attractive part
            gauss2 = math.exp(-(((d - d_ij) / 2.0)**2))  # Repulsive part (broader, for d < d_ij)

            if d < d_ij:
                # Repulsive term dominates when closer than optimal distance
                score_steric += -0.5 * gauss1 + 1.0 * (1.0 - gauss2)
            else:
                # Only attractive term when at or further than optimal distance
                score_steric += -0.5 * gauss1
                
    # --- 2. Hydrogen Bond Term ---
    # Atom types involved in H-bonds
    donor_types = {'N_D', 'O_D'}      # Add other donor types if any
    acceptor_types = {'N_A', 'O_A'}  # Add other acceptor types if any
    
    d_opt_hbond = 2.8  # Optimal H-bond distance in Angstroms
    hbond_strength = -2.0 # Strength of H-bond (kcal/mol), placeholder

    # Ligand Donors to Protein Acceptors
    for l_atom in ligand_mol.GetAtoms():
        try:
            l_atom_type = l_atom.GetProp("atom_type")
        except KeyError: continue # Skip if no type
            
        if l_atom_type in donor_types:
            for p_atom in protein_mol.GetAtoms():
                try:
                    p_atom_type = p_atom.GetProp("atom_type")
                except KeyError: continue # Skip if no type

                if p_atom_type in acceptor_types:
                    d_DA = get_distance(p_atom, l_atom, protein_conf, ligand_conf)
                    # Distance-dependent term (Gaussian form)
                    energy_dist = math.exp(-(((d_DA - d_opt_hbond) / 0.4)**2))
                    # Angular term is highly simplified here (effectively ignored or assumed optimal)
                    score_hbond += hbond_strength * energy_dist

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
                    energy_dist = math.exp(-(((d_DA - d_opt_hbond) / 0.4)**2))
                    # Angular term simplified
                    score_hbond += hbond_strength * energy_dist
                    
    # --- 3. Hydrophobic Term ---
    # Atom types considered hydrophobic as per prompt
    hydrophobic_types = {'C_H', 'F_H', 'Cl_H', 'Br_H', 'I_H'}
    
    d_hp_opt = 4.0  # Optimal hydrophobic contact distance in Angstroms
    hydrophobic_strength = -0.3 # Strength of hydrophobic interaction (kcal/mol), placeholder

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
                    energy_hp = math.exp(-(((d_hp - d_hp_opt) / 1.0)**2))
                    score_hydrophobic += hydrophobic_strength * energy_hp

    # --- 4. Ligand Torsional Penalty ---
    # This is a simplified penalty based only on the number of rotatable bonds.
    # More sophisticated models use specific torsional parameters based on atom types around the bond.
    torsion_penalty_per_bond = 0.35 # Penalty per rotatable bond (kcal/mol), placeholder
    score_torsion = torsion_penalty_per_bond * num_rotatable_bonds

    # --- Total Score ---
    # Weights for each component term. All set to 1.0 as per prompt for now.
    w_steric = 1.0
    w_hbond = 1.0
    w_hydrophobic = 1.0
    w_torsion = 1.0 # This weight applies to the already calculated score_torsion

    total_score = (w_steric * score_steric +
                   w_hbond * score_hbond +
                   w_hydrophobic * score_hydrophobic +
                   w_torsion * score_torsion)

    return total_score
