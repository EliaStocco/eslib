from ase import Atoms


def read_pdb(file_path):
    positions = []
    symbols = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Only process lines that start with 'ATOM'
            if line.startswith('ATOM'):
                symbol = line[12:16].strip()  # Extract atom name (e.g., 'O', 'H')
                x = float(line[30:38].strip())  # X coordinate
                y = float(line[38:46].strip())  # Y coordinate
                z = float(line[46:54].strip())  # Z coordinate
                
                positions.append([x, y, z])  # Add position as [x, y, z]
                symbols.append(symbol)        # Add atom symbol
    
    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions)
    
    return atoms
