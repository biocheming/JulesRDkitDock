# RDKit-based Molecular Docking Tool

This project is a simple molecular docking tool built using RDKit. It aims to perform basic docking of a flexible ligand to a rigid protein receptor using a pre-calculated energy grid and a Vinardo-like scoring function.

**Note:** This is a developmental tool. The scoring function is a simplified version, and advanced features like the direct use of a Torsion Library for conformer generation are currently deferred.

## Features
- Ligand conformer generation using RDKit.
- Atom typing for protein and ligand.
- Energy grid pre-calculation for the protein binding site.
- Monte Carlo-based docking search algorithm.
- Simplified Vinardo-like scoring function.
- Optional final pose refinement using MMFF94s force field.

## Dependencies
- Python 3.x
- RDKit: `conda install -c conda-forge rdkit` or `pip install rdkit-pypi`
- NumPy: `pip install numpy`

## Installation
1. Clone this repository.
2. Ensure Python and the dependencies listed above are installed.
3. The main script is `src/main.py`.

## Basic Usage

To run a docking calculation, use the `main.py` script from the `src` directory:

```bash
python src/main.py --protein_file data/example_protein.pdb --ligand_file data/example_ligand.sdf --output_file docked_poses.sdf
```

**Command-line Options:**

- `--protein_file PATH`: Path to the protein structure file (e.g., PDB). (Required)
- `--ligand_file PATH`: Path to the ligand structure file (e.g., SDF). (Required)
- `--output_file PATH`: Path to save the docked ligand poses (SDF format). (Required)
- `--num_conformers INT`: Number of conformers to generate for the ligand (default: 50).
- `--grid_padding FLOAT`: Padding around the ligand to define the grid (default: 5.0 Angstroms).
- `--grid_spacing FLOAT`: Spacing for the energy grid (default: 0.5 Angstroms).
- `--docking_orientations INT`: Number of random orientations for docking (default: 10).
- `--docking_local_search_steps INT`: Number of local search steps during docking (default: 50).
- `--top_n_poses INT`: Number of top poses to save (default: 10).
- `--mmff94s_refinement / --no-mmff94s_refinement`: Whether to perform MMFF94s refinement (default: True).
- `--mmff94s_max_iterations INT`: Max iterations for MMFF94s (default: 200).

Example with more options:
```bash
python src/main.py \
  --protein_file data/example_protein.pdb \
  --ligand_file data/example_ligand.sdf \
  --output_file output/my_docked_ligands.sdf \
  --num_conformers 100 \
  --top_n_poses 5 \
  --no-mmff94s_refinement
```

## Project Structure
- `src/`: Contains all source code (`main.py`, `atom_typer.py`, `scoring.py`, `grid.py`, `conformer_generation.py`, `docking.py`).
- `tests/`: Contains unit tests.
- `data/`: Contains example data files.

## To Do / Future Enhancements
- Implement more accurate Vinardo scoring terms and weights.
- Integrate Torsion Library for conformer generation.
- Improve local search algorithm during docking.
- More comprehensive unit and integration tests.
- Support for flexible receptor docking.
```
