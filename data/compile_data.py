from pathlib import Path
from typing import cast

import pandas as pd
from pandas._typing import DtypeArg

DTYPE_MAP = {
    "[QUOTE_UNIXTIME]": "int64",
    "[QUOTE_TIME_HOURS]": "float64",
    "[UNDERLYING_LAST]": "float64",
    "[EXPIRE_UNIX]": "int64",
    "[DTE]": "float64",
    "[C_DELTA]": "float64",
    "[C_GAMMA]": "float64",
    "[C_VEGA]": "float64",
    "[C_THETA]": "float64",
    "[C_RHO]": "float64",
    "[C_IV]": "float64",
    "[C_VOLUME]": "float64",
    "[C_LAST]": "float64",
    "[C_SIZE]": "string",
    "[C_BID]": "float64",
    "[C_ASK]": "float64",
    "[STRIKE]": "float64",
    "[P_BID]": "float64",
    "[P_ASK]": "float64",
    "[P_SIZE]": "string",
    "[P_LAST]": "float64",
    "[P_DELTA]": "float64",
    "[P_GAMMA]": "float64",
    "[P_VEGA]": "float64",
    "[P_THETA]": "float64",
    "[P_RHO]": "float64",
    "[P_IV]": "float64",
    "[P_VOLUME]": "float64",
    "[STRIKE_DISTANCE]": "float64",
    "[STRIKE_DISTANCE_PCT]": "float64",
}


def process_data(data_path, out_path):
    print(f"Input:\t{data_path};\nOutput:\t{out_path}")
    data = pd.DataFrame()
    data_path = Path(data_path)

    for idx, file in enumerate(data_path.iterdir()):
        print(f"\rProcessing file {idx}: {file}", end="", flush=True)
        if file.is_file():
            curr = pd.read_csv(
                file,
                dtype=cast(DtypeArg, DTYPE_MAP),
                parse_dates=[
                    "[QUOTE_READTIME]",
                    "[QUOTE_DATE]",
                    "[EXPIRE_DATE]",
                ],
                na_values=["", " "],
                low_memory=False,
                skipinitialspace=True,
            )
            curr.columns = curr.columns.str.replace(r"[\[\]]", "", regex=True)
            if curr is not None:
                data = pd.concat([data, curr], ignore_index=True)

    print("\nDone processing.")
    print(f"Writing to {out_path}")
    data.to_parquet(out_path)


if __name__ == "__main__":
    data_path = "./raw/"
    out_path = "./full_data.parquet"
    process_data(data_path, out_path)
