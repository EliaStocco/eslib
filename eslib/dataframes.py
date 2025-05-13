import numpy as np
import pandas as pd
from typing import List
from eslib.formatting import float_format

import pandas as pd
import numpy as np
from typing import List

def df2txt(
    df: pd.DataFrame,
    filename: str,
    int_columns: List[str],
    float_format: str = float_format,
    float_width: int = 24,  # New parameter for integer column width
    int_width: int = 15  # New parameter for integer column width
) -> None:
    """
    Save a DataFrame to a txt file using np.savetxt with aligned header and formatting.

    Parameters:
    - df: pandas DataFrame to save
    - filename: output filename
    - int_columns: list of column names to treat as integers
    - float_format: format for float columns (default "%24.12e")
    - int_width: width for integer columns (default 10)
    """
    # Ensure int columns are actually int type
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Build header and format list with adjusted column widths
    header = ""
    fmt = []
    
    for col in df.columns:
        if col in int_columns:
            # Use the provided int_width for integer columns
            header += f"{col:>{int_width}}"
            fmt.append(f"%{int_width}d")  # Use the fixed int width
        else:
            # Use the default float format for non-integer columns
            header += f"{col:>{float_width}s}"
            fmt.append(float_format)
    
    # Save to file
    np.savetxt(
        filename,
        df.to_numpy(),
        fmt=fmt,
        header=header,
        comments=""
    )
