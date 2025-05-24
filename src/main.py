import argparse
import os
import sys
import time # For basic profiling or logging if needed

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    sys.exit("RDKit is not installed. Please install RDKit to use this script.")

# Custom module imports
# This structure assumes main.py is in the src/ directory and can import siblings.
# If running from outside src/, PYTHONPATH might need to be adjusted.
try:
    import atom_typer
    import conformer_generation
    import grid
    import docking
    import scoring # Though not directly called by main, docking uses it.
except ImportError as e:
    sys.exit(f"Error importing custom modules: {e}. Ensure main.py is run correctly relative to other src files or PYTHONPATH is set.")

# Define the atom types that the grid will be sensitive to.
# These should generally match the types defined in atom_typer.ATOM_TYPE_DEFINITIONS
ATOM_TYPES_FOR_GRID_PROBE = [
    'C_H', 'C_P', 'N_D', 'N_A', 'N_P', 'O_A', 'O_D', 
    'S_P', 'P_P', 'F_H', 'Cl_H', 'Br_H', 'I_H', 'Metal', 'UNK'
]

def load_molecule(filepath, add_hydrogens=True, is_protein=False):
    """
    Loads a molecule from a file, optionally adds hydrogens.
    Supports PDB, MOL2 for proteins; SDF, MOL2, PDB for ligands.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None
    
    mol = None
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext == '.pdb':
        mol = Chem.MolFromPDBFile(filepath, removeHs=False) # Keep existing Hs if any initially
    elif file_ext == '.mol2':
        mol = Chem.MolFromMol2File(filepath, removeHs=False)
    elif file_ext == '.sdf' and not is_protein: # Proteins usually not in SDF for this context
        suppl = Chem.SDMolSupplier(filepath, removeHs=False)
        if suppl and len(suppl) > 0:
            mol = suppl[0] # Take the first molecule from SDF
    else:
        print(f"Error: Unsupported file type {file_ext} for {'protein' if is_protein else 'ligand'}: {filepath}")
        return None

    if mol is None:
        print(f"Error: Failed to load molecule from {filepath}. RDKit returned None.")
        return None

    if add_hydrogens:
        # AddHs can fail for very large molecules if sanitize=True (default) and issues exist.
        # For proteins, especially from PDB, it's common to need cleanup or specific handling.
        # addCoords=True is important if the molecule will be used for conformer generation or needs 3D Hs.
        try:
            mol = Chem.AddHs(mol, addCoords=True, explicitOnly=False) # explicitOnly=False for proteins from PDB often
        except Exception as e:
            print(f"Warning: Failed to add hydrogens to {filepath}: {e}. Proceeding without added hydrogens.")
            # If AddHs fails, mol might be in an intermediate state or unchanged.
            # It's safer to re-evaluate if it's usable. For now, we proceed with original mol.
    
    # Basic check for 3D coordinates if not a protein that will be typed/gridded later
    if not is_protein and mol.GetNumConformers() == 0:
        # Try to compute 2D coords then embed if no 3D coords (e.g. from SMILES or 0D SDF)
        # This is more for ligand if it comes from a source without 3D.
        # print(f"Warning: Ligand {filepath} has no 3D conformer. Attempting to generate one.")
        # AllChem.Compute2DCoords(mol)
        # if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1: # Failure to embed
        #     print(f"Error: Failed to generate 3D coordinates for ligand {filepath}.")
        #     return None
        # AllChem.UFFOptimizeMolecule(mol) # Basic optimization
        pass # Conformer generation step will handle this for the ligand.

    return mol

def main():
    parser = argparse.ArgumentParser(description="Simple molecular docking script using RDKit.")
    parser.add_argument("--protein_file", required=True, help="Path to protein structure file (PDB, MOL2).")
    parser.add_argument("--ligand_file", required=True, help="Path to ligand structure file (SDF, MOL2, PDB).")
    parser.add_argument("--output_file", required=True, help="Path to save docked ligand poses (SDF).")
    
    parser.add_argument("--num_conformers", type=int, default=50, help="Number of conformers to generate for the ligand (default: 50).")
    parser.add_argument("--grid_padding", type=float, default=5.0, help="Padding around the ligand for grid definition (default: 5.0 A).")
    parser.add_argument("--grid_spacing", type=float, default=0.5, help="Spacing for the energy grid (default: 0.5 A).")
    
    parser.add_argument("--docking_orientations", type=int, default=10, help="Number of random orientations per ligand conformer during docking (default: 10).")
    parser.add_argument("--docking_local_search_steps", type=int, default=50, help="Number of local search steps during docking (default: 50).")
    parser.add_argument("--top_n_poses", type=int, default=10, help="Number of top poses to save (default: 10).")
    
    parser.add_argument("--mmff94s_refinement", action=argparse.BooleanOptionalAction, default=True, help="Perform MMFF94s refinement of top poses (default: True).")
    parser.add_argument("--mmff94s_max_iterations", type=int, default=200, help="Max iterations for MMFF94s refinement (default: 200).")

    args = parser.parse_args()

    print("--- Starting Docking Workflow ---")
    start_time = time.time()

    # 1. Load Molecules
    print(f"Loading protein from: {args.protein_file}")
    protein_mol = load_molecule(args.protein_file, add_hydrogens=True, is_protein=True)
    if not protein_mol:
        sys.exit("Exiting due to protein loading failure.")

    print(f"Loading ligand from: {args.ligand_file}")
    # For ligand, AddHs is typically done before conformer generation if not already present.
    # The generate_conformers function also calls AddHs.
    ligand_mol = load_molecule(args.ligand_file, add_hydrogens=False, is_protein=False) # AddHs will be done by conformer generator
    if not ligand_mol:
        sys.exit("Exiting due to ligand loading failure.")

    # 2. Atom Typing
    print("Assigning atom types...")
    try:
        protein_mol = atom_typer.assign_atom_types(protein_mol)
        ligand_mol = atom_typer.assign_atom_types(ligand_mol) # Type before conformer gen to ensure types are on topology
        print("Atom typing completed.")
    except Exception as e:
        sys.exit(f"Error during atom typing: {e}")

    # 3. Conformer Generation for Ligand
    print(f"Generating {args.num_conformers} conformers for the ligand...")
    try:
        ligand_mol_with_conformers = conformer_generation.generate_conformers(
            ligand_mol, 
            num_conformers=args.num_conformers
        ) # Max attempts and random seed use defaults in generate_conformers
    except Exception as e:
        sys.exit(f"Error during conformer generation: {e}")

    if ligand_mol_with_conformers.GetNumConformers() == 0:
        sys.exit("Error: No conformers generated for the ligand. Cannot proceed with docking.")
    print(f"Generated {ligand_mol_with_conformers.GetNumConformers()} conformers.")

    # 4. Grid Generation
    print("Defining grid around binding site...")
    # Use the first generated conformer of the ligand to help define the grid center and extents.
    # Ensure protein also has a conformer (load_molecule should ensure this for proteins if AddHs worked)
    if protein_mol.GetNumConformers() == 0:
        sys.exit("Error: Protein has no conformer, cannot define grid.")
        
    first_lig_conf = ligand_mol_with_conformers.GetConformer(0)
    try:
        grid_definition = grid.define_grid_around_binding_site(
            protein_mol, 
            ligand_mol_with_conformers, # Pass full mol for ligand, function will use its conformer(s)
            padding=args.grid_padding, 
            spacing=args.grid_spacing
        )
        print(f"Grid defined: Center={grid_definition['center']}, Dims={grid_definition['dimensions']}, Spacing={grid_definition['spacing']}")
    except Exception as e:
        sys.exit(f"Error defining grid: {e}")

    print("Populating energy grid...")
    try:
        energy_grid, atom_types_in_grid = grid.populate_energy_grid(
            protein_mol, 
            grid_definition, 
            ATOM_TYPES_FOR_GRID_PROBE # Use predefined list
        )
        print(f"Energy grid populated. Shape: {energy_grid.shape}, Probed types: {atom_types_in_grid}")
    except Exception as e:
        sys.exit(f"Error populating energy grid: {e}")

    # 5. Docking
    print("Performing docking...")
    try:
        docked_poses_mols = docking.perform_docking(
            protein_mol, 
            ligand_mol_with_conformers, 
            energy_grid, 
            grid_definition, 
            atom_types_in_grid, # This is atom_types_to_probe used for grid creation
            n_orientations=args.docking_orientations,
            n_local_search_steps=args.docking_local_search_steps,
            top_n_poses=args.top_n_poses,
            refine_top_poses_with_mmff94s=args.mmff94s_refinement,
            mmff94s_max_iterations=args.mmff94s_max_iterations
        )
    except Exception as e:
        sys.exit(f"Error during docking: {e}")

    # 6. Saving Results
    if docked_poses_mols:
        print(f"Docking completed. Found {len(docked_poses_mols)} suitable poses.")
        print(f"Saving top poses to: {args.output_file}")
        try:
            writer = Chem.SDWriter(args.output_file)
            for pose_mol in docked_poses_mols:
                writer.write(pose_mol)
            writer.close()
            print(f"{len(docked_poses_mols)} poses saved successfully.")
        except Exception as e:
            print(f"Error saving docked poses: {e}")
    else:
        print("No suitable docked poses were found after the docking procedure.")

    end_time = time.time()
    print(f"--- Docking Workflow Finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
```
