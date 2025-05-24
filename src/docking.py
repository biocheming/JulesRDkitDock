import math
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from rdkit.Numerics import rdGeometry # Used for Point3D

# Import scoring module. Handles potential ImportError for different execution contexts.
try:
    from . import scoring 
except ImportError:
    import scoring 


# --- Helper: Get Grid Score ---
def get_grid_score(ligand_mol_topology, ligand_conformer, grid_definition, energy_grid, atom_types_in_grid):
    """
    Calculates the interaction score of a ligand conformer with the energy grid.
    Lower scores are better.
    """
    total_score = 0.0
    penalty_out_of_bounds = 1000.0 
    penalty_unknown_type = 100.0   

    origin = np.array(grid_definition['origin'])
    spacing = grid_definition['spacing']
    dims = grid_definition['dimensions'] 

    probe_type_to_idx_map = {name: i for i, name in enumerate(atom_types_in_grid)}
    unk_idx = probe_type_to_idx_map.get('UNK', -1) 

    for atom_idx in range(ligand_conformer.GetNumAtoms()):
        atom = ligand_mol_topology.GetAtomWithIdx(atom_idx)
        try:
            atom_type = atom.GetProp("atom_type")
        except KeyError:
            atom_type = "UNK" 

        pos = ligand_conformer.GetAtomPosition(atom_idx)
        world_coords = np.array([pos.x, pos.y, pos.z])
        
        grid_indices_float = (world_coords - origin) / spacing
        ix = int(round(grid_indices_float[0]))
        iy = int(round(grid_indices_float[1]))
        iz = int(round(grid_indices_float[2]))

        if not (0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]):
            total_score += penalty_out_of_bounds
            continue 

        probe_type_index = probe_type_to_idx_map.get(atom_type)
        
        if probe_type_index is None: 
            if unk_idx != -1: 
                probe_type_index = unk_idx
            else: 
                total_score += penalty_unknown_type
                continue
        
        total_score += energy_grid[ix, iy, iz, probe_type_index]
        
    return total_score

# --- Helper: Create Ligand Molecule from Conformer ---
def _create_ligand_with_conformer(template_ligand_mol, conformer_to_assign):
    new_mol = Chem.Mol(template_ligand_mol, quickCopy=True) 
    new_mol.RemoveAllConformers()
    if isinstance(conformer_to_assign, Chem.Conformer):
        new_mol.AddConformer(Chem.Conformer(conformer_to_assign), assignId=True)
    else:
        raise TypeError("conformer_to_assign must be an RDKit Conformer object.")
    
    for i, atom in enumerate(template_ligand_mol.GetAtoms()):
        new_atom = new_mol.GetAtomWithIdx(i)
        for prop_name in atom.GetPropNames():
            new_atom.SetProp(prop_name, atom.GetProp(prop_name))
    return new_mol

# --- Helper: Random Transformation ---
def _get_random_rotation_matrix():
    axis = np.random.rand(3) - 0.5 
    axis /= np.linalg.norm(axis)    
    angle = random.uniform(0, 2 * np.pi)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def _transform_conformer_manual(conformer, rotation_matrix, translation_vector):
    for i in range(conformer.GetNumAtoms()):
        pos = conformer.GetAtomPosition(i)
        original_coords = np.array([pos.x, pos.y, pos.z])
        rotated_coords = np.dot(rotation_matrix, original_coords)
        new_coords = rotated_coords + translation_vector
        conformer.SetAtomPosition(i, rdGeometry.Point3D(new_coords[0], new_coords[1], new_coords[2]))

def _get_random_rotation_matrix_small_angle(max_angle_rad=0.087): # approx 5 degrees
    axis = np.random.rand(3) - 0.5 
    axis /= np.linalg.norm(axis)    
    angle = random.uniform(-max_angle_rad, max_angle_rad)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

# --- Main Docking Function ---
def perform_docking(protein_mol, ligand_mol_with_conformers, 
                    energy_grid, grid_definition, atom_types_in_grid,
                    n_orientations=10, n_local_search_steps=50, top_n_poses=10,
                    refine_top_poses_with_mmff94s=True, mmff94s_max_iterations=200): # Added refinement params
    """
    Performs a simplified Monte Carlo docking procedure.
    Optionally refines the top poses using MMFF94s.
    """
    if not protein_mol or not ligand_mol_with_conformers:
        raise ValueError("Protein and ligand molecules must be provided.")
    if ligand_mol_with_conformers.GetNumConformers() == 0:
        return [] 

    protein_conf = protein_mol.GetConformer() 
    if not protein_conf:
        raise ValueError("Protein molecule must have a conformer.")

    all_scored_poses = [] 

    try:
        num_rot_bonds = Lipinski.NumRotatableBonds(ligand_mol_with_conformers)
    except Exception: 
        num_rot_bonds = 0 

    grid_center = np.array(grid_definition['center'])
    ligand_topology = ligand_mol_with_conformers 

    for lig_conf_template in ligand_mol_with_conformers.GetConformers():
        for _ in range(n_orientations):
            current_search_conf = Chem.Conformer(lig_conf_template) 

            lig_centroid = np.array([current_search_conf.GetAtomPosition(i) for i in range(current_search_conf.GetNumAtoms())]).mean(axis=0)
            translation_to_grid_center = grid_center - lig_centroid
            _transform_conformer_manual(current_search_conf, np.eye(3), translation_to_grid_center)
            
            random_rotation = _get_random_rotation_matrix()
            _transform_conformer_manual(current_search_conf, random_rotation, np.zeros(3))

            small_random_translation = np.random.uniform(-1.0, 1.0, 3) 
            _transform_conformer_manual(current_search_conf, np.eye(3), small_random_translation)

            best_local_conformer = Chem.Conformer(current_search_conf)
            best_local_grid_score = get_grid_score(ligand_topology, best_local_conformer, 
                                                   grid_definition, energy_grid, atom_types_in_grid)

            for _ in range(n_local_search_steps):
                trial_conf = Chem.Conformer(best_local_conformer) 
                
                rand_trans_step = np.random.uniform(-0.2, 0.2, 3)
                small_rot_matrix = _get_random_rotation_matrix_small_angle(max_angle_rad=0.087)

                _transform_conformer_manual(trial_conf, small_rot_matrix, rand_trans_step)
                
                trial_grid_score = get_grid_score(ligand_topology, trial_conf,
                                                  grid_definition, energy_grid, atom_types_in_grid)

                if trial_grid_score < best_local_grid_score:
                    best_local_grid_score = trial_grid_score
                    best_local_conformer = Chem.Conformer(trial_conf) 

            temp_ligand_for_scoring = _create_ligand_with_conformer(ligand_topology, best_local_conformer)
            
            final_vinardo_score = scoring.calculate_vinardo_score(
                protein_mol, temp_ligand_for_scoring, num_rot_bonds
            )
            
            all_scored_poses.append({'score': final_vinardo_score, 'conformer': best_local_conformer})

    all_scored_poses.sort(key=lambda x: x['score'])
    
    top_results_mols = []
    for i in range(min(top_n_poses, len(all_scored_poses))):
        pose_data = all_scored_poses[i]
        mol_for_pose = _create_ligand_with_conformer(ligand_topology, pose_data['conformer'])
        # Store the Vinardo score that was used for initial ranking
        mol_for_pose.SetProp("VinardoScore", f"{pose_data['score']:.4f}")
        mol_for_pose.SetProp("PoseRank", str(i+1)) # Initial rank before refinement
        top_results_mols.append(mol_for_pose)
        
    if refine_top_poses_with_mmff94s and top_results_mols:
        # print(f"Refining top {len(top_results_mols)} poses with MMFF94s...") # For debugging
        refined_poses = refine_poses_mmff94s(
            protein_mol=protein_mol,
            list_of_ligand_poses=top_results_mols, # Pass the list of Mol objects
            max_iterations=mmff94s_max_iterations,
            rescore_with_vinardo=True 
        )
        return refined_poses
    else:
        return top_results_mols

# --- MMFF94s Refinement Function ---
def refine_poses_mmff94s(protein_mol, list_of_ligand_poses, 
                         num_threads=0, max_iterations=200, 
                         rescore_with_vinardo=True):
    """
    Refines a list of docked ligand poses using MMFF94s force field, treating the protein as rigid.
    num_threads is part of signature but not directly used in RDKit MMFF Python API for this.
    """
    if not protein_mol:
        raise ValueError("Protein molecule must be provided for refinement.")
    if not list_of_ligand_poses:
        return []

    refined_poses_data = [] # Store dicts: {'mol': refined_mol, 'score': new_score}
    
    num_protein_atoms = protein_mol.GetNumAtoms()

    for i, ligand_pose_mol_template in enumerate(list_of_ligand_poses):
        if not isinstance(ligand_pose_mol_template, Chem.Mol) or ligand_pose_mol_template.GetNumConformers() == 0:
            # print(f"Warning: Ligand pose {i} is invalid or has no conformer, skipping refinement.")
            continue

        # 1. Create complex. Ensure protein_mol also has a conformer.
        if protein_mol.GetNumConformers() == 0:
            raise ValueError("Protein molecule for refinement must have a conformer.")
        
        # CombineMols needs both molecules to have conformers.
        # The ligand_pose_mol_template is guaranteed to have one.
        complex_mol = Chem.CombineMols(protein_mol, ligand_pose_mol_template)
        
        # 2. Add Hydrogens to the complex
        try:
            complex_mol_hs = Chem.AddHs(complex_mol, addCoords=True)
            if complex_mol_hs.GetNumConformers() == 0 : # AddHs might fail to generate coords for complex
                # print(f"Warning: AddHs failed to generate conformer for complex of pose {i}. Skipping.")
                continue
        except Exception as e:
            # print(f"Warning: Failed to add hydrogens to complex for pose {i}: {e}. Skipping.")
            continue

        # 3. Set up MMFF94s parameters
        try:
            mp = AllChem.MMFFGetMoleculeProperties(complex_mol_hs, mmffVariant="MMFF94s")
            if mp is None:
                # print(f"Warning: Could not get MMFF properties for complex of pose {i}. Skipping.")
                continue
            ff = AllChem.MMFFGetMoleculeForceField(complex_mol_hs, mp, confId=0) 
            if ff is None:
                # print(f"Warning: Could not get MMFF force field for complex of pose {i}. Skipping.")
                continue
        except Exception as e:
            # print(f"Warning: Error setting up MMFF for complex of pose {i}: {e}. Skipping.")
            continue

        # 4. Constrain protein atoms
        for atom_idx in range(num_protein_atoms):
            ff.AddFixedPoint(atom_idx)

        # 5. Perform minimization
        try:
            ff.Minimize(maxIts=max_iterations)
        except Exception as e:
            # print(f"Warning: MMFF minimization failed for complex of pose {i}: {e}. Skipping.")
            continue # If minimization fails, do not proceed with this pose

        # 6. Extract optimized ligand coordinates
        complex_conformer = complex_mol_hs.GetConformer(0)
        
        # Create a new molecule for the refined ligand using the original template for topology & props
        refined_ligand_mol = Chem.Mol(ligand_pose_mol_template) 
        refined_ligand_mol.RemoveAllConformers()
        
        new_lig_conformer = Chem.Conformer(ligand_pose_mol_template.GetNumAtoms())
        
        for lig_atom_idx in range(ligand_pose_mol_template.GetNumAtoms()):
            complex_atom_idx = num_protein_atoms + lig_atom_idx
            new_pos = complex_conformer.GetAtomPosition(complex_atom_idx)
            new_lig_conformer.SetAtomPosition(lig_atom_idx, new_pos)
        
        refined_ligand_mol.AddConformer(new_lig_conformer, assignId=True)

        # 7. Optionally re-score
        current_score_val = float('inf')
        score_prop_key = "VinardoScore_MMFF94s_Refined" # New property name for clarity

        if rescore_with_vinardo:
            try:
                num_rot_bonds_refined_lig = Lipinski.NumRotatableBonds(refined_ligand_mol)
                new_vinardo_score = scoring.calculate_vinardo_score(protein_mol, refined_ligand_mol, num_rot_bonds_refined_lig)
                refined_ligand_mol.SetProp(score_prop_key, f"{new_vinardo_score:.4f}")
                current_score_val = new_vinardo_score
            except Exception as e:
                # print(f"Warning: Failed to re-score refined pose {i} with Vinardo: {e}.")
                # If re-scoring fails, try to use original score, or mark as high penalty
                if ligand_pose_mol_template.HasProp("VinardoScore"):
                    refined_ligand_mol.SetProp("VinardoScore", ligand_pose_mol_template.GetProp("VinardoScore"))
                    try:
                        current_score_val = float(ligand_pose_mol_template.GetProp("VinardoScore"))
                    except ValueError:
                        pass # Keep as inf
                refined_ligand_mol.SetProp(score_prop_key, "Error")


        # Copy original VinardoScore and PoseRank for reference
        if ligand_pose_mol_template.HasProp("VinardoScore"):
            refined_ligand_mol.SetProp("VinardoScore_Initial", ligand_pose_mol_template.GetProp("VinardoScore"))
        if ligand_pose_mol_template.HasProp("PoseRank"):
            refined_ligand_mol.SetProp("PoseRank_Initial", ligand_pose_mol_template.GetProp("PoseRank"))
        
        refined_poses_data.append({'mol': refined_ligand_mol, 'score': current_score_val})

    # 8. Sort refined poses by score (lower is better)
    refined_poses_data.sort(key=lambda x: x['score'])

    # Prepare final list of molecules with updated PoseRank
    final_refined_mols_list = []
    for rank, data in enumerate(refined_poses_data):
        mol = data['mol']
        mol.SetProp("PoseRank_MMFF94s_Refined", str(rank + 1))
        # Ensure the score used for sorting is present and clearly named
        if not mol.HasProp(score_prop_key) and data['score'] != float('inf') and data['score'] != float('-inf'):
             mol.SetProp(score_prop_key, f"{data['score']:.4f}")
        final_refined_mols_list.append(mol)
        
    return final_refined_mols_list
