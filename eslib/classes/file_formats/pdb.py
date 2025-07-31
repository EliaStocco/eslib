from ase import Atoms

def read_pdb(file_path:str):
    positions = []
    symbols = []
    cell = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('CRYST1'):
                # Extract cell parameters: a, b, c, alpha, beta, gamma
                a = float(line[6:15])
                b = float(line[15:24])
                c = float(line[24:33])
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])
                
                # Convert to cell vectors using ASE function
                from ase.geometry import cellpar_to_cell
                cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])
            
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                symbol = line[76:78].strip() or line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                positions.append([x, y, z])
                symbols.append(symbol)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True if cell is not None else False)
    return atoms

def write_pdb(file_path:str, atoms:Atoms):
    from ase.geometry import cell_to_cellpar

    with open(file_path, 'w') as f:
        # Write CRYST1 line if cell is defined
        if atoms.cell is not None and atoms.cell.volume > 0:
            a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
            f.write(
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
                f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
            )

        for i, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions), start=1):
            f.write(
                f"ATOM  {i:5d} {symbol:<4s} MOL     1    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00          {symbol:>2s}\n"
            )
        f.write("END\n")
