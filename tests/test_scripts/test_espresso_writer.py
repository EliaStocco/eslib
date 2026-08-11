from ase import Atoms

from eslib.classes.atomic_structures import AtomicStructures


def test_write_espresso_in_uses_crystal_positions(tmp_path):
    atoms = Atoms(
        symbols=["Ba", "Ti", "O", "O", "O"],
        cell=[
            [3.977555723995, 0.0, 0.0],
            [0.0, 3.977555723995, 0.0],
            [0.0, 0.0, 3.977555723995],
        ],
        scaled_positions=[
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ],
        pbc=True,
    )
    output = tmp_path / "BaTiO3.in"

    AtomicStructures([atoms]).to_file(str(output), format="espresso-in")

    assert output.read_text() == (
        "CELL_PARAMETERS angstrom\n"
        "  3.977555723995  0.000000000000  0.000000000000\n"
        "  0.000000000000  3.977555723995  0.000000000000\n"
        "  0.000000000000  0.000000000000  3.977555723995\n"
        "\n"
        "ATOMIC_POSITIONS crystal\n"
        "Ba  0.500000000000  0.500000000000  0.500000000000\n"
        "Ti  0.000000000000  0.000000000000  0.000000000000\n"
        "O   0.000000000000  0.000000000000  0.500000000000\n"
        "O   0.000000000000  0.500000000000  0.000000000000\n"
        "O   0.500000000000  0.000000000000  0.000000000000\n"
    )
