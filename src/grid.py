import numpy as np
import math
from rdkit import Chem

# --- Vinardo Steric Interaction Parameters (mirrored from scoring.py for grid calculation) ---
WEIGHT_GAUSS1 = -0.045      # Attractive Gaussian component
WEIGHT_REPLUSION = 0.8       # Repulsive linear component for d < d_ij_ideal
C1_GAUSS = 0.5               # Width for the attractive Gaussian

# --- Global VDW Radii Sum Defaults ---
# (protein_atom_type - probe_atom_type: sum_of_vdw_radii)
# Based on types from atom_typer.py:
# H, C_H, C_P, N_D, N_A, N_P, O_A, O_D, S_P, P_P, F_H, Cl_H, Br_H, I_H, Metal, UNK
VDW_RADII_SUM_DEFAULTS = {
    # Probe C_H
    'C_H-C_H': 3.5, 'C_P-C_H': 3.5, 'N_D-C_H': 3.6, 'N_A-C_H': 3.6, 'N_P-C_H': 3.6,
    'O_D-C_H': 3.4, 'O_A-C_H': 3.4, 'S_P-C_H': 3.9, 'P_P-C_H': 4.0, 'H-C_H': 2.8,
    'F_H-C_H': 3.3, 'Cl_H-C_H': 3.7, 'Br_H-C_H': 3.8, 'I_H-C_H': 4.0, 'Metal-C_H': 3.5, 'UNK-C_H': 3.8,

    # Probe N_D (Donor N)
    'C_H-N_D': 3.6, 'C_P-N_D': 3.6, 'N_D-N_D': 3.4, 'N_A-N_D': 3.4, 'N_P-N_D': 3.4,
    'O_D-N_D': 3.3, 'O_A-N_D': 3.3, 'S_P-N_D': 3.8, 'P_P-N_D': 3.9, 'H-N_D': 2.7,
    'F_H-N_D': 3.2, 'Cl_H-N_D': 3.6, 'Br_H-N_D': 3.7, 'I_H-N_D': 3.9, 'Metal-N_D': 3.4, 'UNK-N_D': 3.7,

    # Probe O_A (Acceptor O)
    'C_H-O_A': 3.4, 'C_P-O_A': 3.4, 'N_D-O_A': 3.3, 'N_A-O_A': 3.3, 'N_P-O_A': 3.3,
    'O_D-O_A': 3.2, 'O_A-O_A': 3.2, 'S_P-O_A': 3.7, 'P_P-O_A': 3.8, 'H-O_A': 2.6,
    'F_H-O_A': 3.1, 'Cl_H-O_A': 3.5, 'Br_H-O_A': 3.6, 'I_H-O_A': 3.8, 'Metal-O_A': 3.3, 'UNK-O_A': 3.6,
    
    # Probe UNK (for a generic probe)
    'C_H-UNK': 3.8, 'C_P-UNK': 3.8, 'N_D-UNK': 3.7, 'N_A-UNK': 3.7, 'N_P-UNK': 3.7,
    'O_D-UNK': 3.6, 'O_A-UNK': 3.6, 'S_P-UNK': 4.0, 'P_P-UNK': 4.1, 'H-UNK': 3.0,
    'F_H-UNK': 3.6, 'Cl_H-UNK': 3.9, 'Br_H-UNK': 4.0, 'I_H-UNK': 4.2, 'Metal-UNK': 3.8, 'UNK-UNK': 3.8,
}
DEFAULT_VDW_SUM = 3.8  # Default VdW sum if a specific pair is not found

def define_grid_around_binding_site(protein_mol, ligand_mol, padding=5.0, spacing=0.5):
    """
    Defines a grid around the ligand binding site in a protein.
    The grid is centered on the ligand. Its size is determined by the ligand's
    dimensions plus padding.

    Args:
        protein_mol (rdkit.Chem.rdchem.Mol): Protein molecule (used to ensure it exists).
        ligand_mol (rdkit.Chem.rdchem.Mol): Ligand molecule (defines grid center and influences size).
        padding (float): Padding in Angstroms around the ligand to define grid extent.
        spacing (float): Grid spacing in Angstroms.

    Returns:
        dict: Grid definition with 'center', 'dimensions' (nx,ny,nz points), 
              'spacing', and 'origin' (min corner coordinates).
    Raises:
        ValueError: If molecules or conformers are missing, or ligand has no atoms.
    """
    if not protein_mol or not ligand_mol:
        raise ValueError("Protein and ligand molecules must be provided.")
    
    protein_conf = protein_mol.GetConformer() # Check protein has a conformer
    ligand_conf = ligand_mol.GetConformer()

    if protein_conf is None or ligand_conf is None:
        raise ValueError("Protein and ligand molecules must have conformers.")

    ligand_coords = ligand_conf.GetPositions()
    if ligand_coords.shape[0] == 0:
        raise ValueError("Ligand has no atoms or coordinates.")
    
    grid_center = ligand_coords.mean(axis=0)

    # Determine grid size based on ligand's extent plus padding
    lig_min = ligand_coords.min(axis=0)
    lig_max = ligand_coords.max(axis=0)
    
    # Half-sizes from the center to the edge of the padded box around the ligand
    # If ligand is a single point, lig_max - lig_min is 0, so half_sizes is just padding.
    if np.array_equal(lig_min, lig_max): # Handle single-atom ligand case
        half_sizes = np.array([padding, padding, padding])
    else:
        half_sizes = (lig_max - lig_min) / 2.0 + padding
    
    # Define min/max coordinates for the grid box based on grid_center and half_sizes
    min_box_coords = grid_center - half_sizes
    max_box_coords = grid_center + half_sizes
    
    span = max_box_coords - min_box_coords
    
    # Calculate number of points, ensuring it's at least 1 in each dimension
    num_points = np.maximum(np.ceil(span / spacing).astype(int) + 1, 1)

    # Adjust span to be an exact multiple of spacing based on num_points
    adjusted_span = (num_points - 1) * spacing
    
    # Origin (min corner of the grid box), ensuring grid_center is maintained
    origin = grid_center - (adjusted_span / 2.0)
    
    dimensions = tuple(num_points) # nx, ny, nz

    return {
        "center": tuple(grid_center.tolist()),
        "dimensions": dimensions,
        "spacing": spacing,
        "origin": tuple(origin.tolist())
    }


def populate_energy_grid(protein_mol, grid_definition, atom_types_to_probe):
    """
    Populates a grid with interaction energies between protein and probe atom types.

    Args:
        protein_mol (rdkit.Chem.rdchem.Mol): Protein molecule with 'atom_type' properties.
        grid_definition (dict): Output from define_grid_around_binding_site.
        atom_types_to_probe (list): List of strings, e.g., ['C_H', 'N_D', 'O_A'].

    Returns:
        tuple: (energy_grid (np.ndarray), atom_types_to_probe (list))
               energy_grid shape: (nx, ny, nz, num_probe_atom_types)
    Raises:
        ValueError: If protein molecule or its conformer is missing.
    """
    if not protein_mol:
        raise ValueError("Protein molecule must be provided.")
    protein_conf = protein_mol.GetConformer()
    if protein_conf is None:
        raise ValueError("Protein molecule must have a conformer.")

    nx, ny, nz = grid_definition["dimensions"]
    origin_x, origin_y, origin_z = grid_definition["origin"]
    spacing = grid_definition["spacing"]
    
    num_probe_types = len(atom_types_to_probe)
    # probe_type_to_index = {ptype: i for i, ptype in enumerate(atom_types_to_probe)} # Not strictly needed if using enumerate

    energy_grid = np.zeros((nx, ny, nz, num_probe_types), dtype=np.float32)

    protein_atoms = list(protein_mol.GetAtoms()) 
    protein_atom_coords = protein_conf.GetPositions()
    protein_atom_types = []
    for p_atom in protein_atoms:
        try:
            protein_atom_types.append(p_atom.GetProp("atom_type"))
        except KeyError:
            protein_atom_types.append("UNK")


    for ix in range(nx):
        gx = origin_x + ix * spacing
        for iy in range(ny):
            gy = origin_y + iy * spacing
            for iz in range(nz):
                gz = origin_z + iz * spacing
                grid_point_coord = np.array([gx, gy, gz])

                for probe_idx, probe_atom_type in enumerate(atom_types_to_probe):
                    total_interaction_energy_for_probe = 0.0
                    
                    for i, p_atom in enumerate(protein_atoms):
                        p_atom_type = protein_atom_types[i]
                        p_atom_coord = protein_atom_coords[p_atom.GetIdx()] # Use GetIdx() to be safe
                        
                        distance = np.linalg.norm(grid_point_coord - p_atom_coord)

                        pair_key1 = f"{p_atom_type}-{probe_atom_type}"
                        pair_key2 = f"{probe_atom_type}-{p_atom_type}" 
                        
                        d_ij_ideal = VDW_RADII_SUM_DEFAULTS.get(
                            pair_key1, 
                            VDW_RADII_SUM_DEFAULTS.get(pair_key2, DEFAULT_VDW_SUM)
                        )
                        
                        # New Vinardo steric interaction calculation
                        if distance < 1e-6: # Avoid issues with distance being exactly zero if atoms overlap perfectly
                            # Apply a very high repulsion or use the formula which will also be very high
                            # Using the formula directly: d_ij_ideal - 0.0 (max possible repulsion for this pair)
                            interaction_energy = WEIGHT_REPLUSION * d_ij_ideal 
                                                 # Potentially add WEIGHT_GAUSS1 if d=0 makes exp(huge_negative) -> 0
                                                 # Or simply a large penalty:
                            # interaction_energy = 1000.0 # Large penalty for direct overlap
                        else:
                            gauss1_attractive_exp_term = math.exp(-(((distance - d_ij_ideal) / C1_GAUSS)**2))
                            gauss1_contrib = WEIGHT_GAUSS1 * gauss1_attractive_exp_term
                            
                            repulsion_contrib = 0.0
                            if distance < d_ij_ideal:
                                repulsion_contrib = WEIGHT_REPLUSION * (d_ij_ideal - distance)
                            
                            interaction_energy = gauss1_contrib + repulsion_contrib
                        
                        total_interaction_energy_for_probe += interaction_energy
                    
                    energy_grid[ix, iy, iz, probe_idx] = total_interaction_energy_for_probe
    
    return energy_grid, atom_types_to_probe
