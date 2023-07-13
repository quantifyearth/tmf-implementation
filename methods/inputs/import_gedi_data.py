import glob
import os
import sys
from multiprocessing import Pool

import pandas as pd
import sqlalchemy

from biomassrecovery import environment  # type: ignore
from biomassrecovery.data.gedi_database_loader import parse_file  # type: ignore

# this constant is taken from Amelia's code, as she didn't
# want to swamp the DB with connections
PROCESSES = 8

def import_file(filename: str) -> None:
	try:
		gedi_data = parse_file(filename)
	except OSError:
		print(f"Failed to open {filename}", file=sys.stderr)
		return
	if gedi_data.empty:
		return

	engine = sqlalchemy.create_engine(environment.DB_CONFIG, echo=False)
	with engine.begin() as con:
		existing = pd.read_sql_table(
			table_name="level_4a_granules", columns=["granule_name"], con=con
		)

		gedi_data = gedi_data.astype({"shot_number": "int64"})
		this_granule_name = gedi_data["granule_name"].head(1).item()

		if (existing.granule_name==this_granule_name).any():
			print(f"Skipping {filename} as granule {this_granule_name} already exists.")
			return

		granules_entry = pd.DataFrame(
			data={
				"granule_name": [this_granule_name],
				"created_date": [pd.Timestamp.utcnow()],
			}
		)
		granules_entry.to_sql(
			name="level_4a_granules",
			con=con,
			index=False,
			if_exists="append",
		)
		gedi_data.to_postgis(
			name="level_4a",
			con=con,
			index=False,
			if_exists="append",
		)
		del gedi_data

def import_gedi_data(
	gedi_data_folder: str,
) -> None:
	gedi_files = [os.path.join(gedi_data_folder, x) for x in glob.glob('*.h5', root_dir=gedi_data_folder)]
	with Pool(processes=PROCESSES) as pool:
		pool.map(import_file, gedi_files)

if __name__ == "__main__":
	try:
		gedi_data_folder = sys.argv[1]
	except IndexError:
		print(f"Usage: {sys.argv[0]} GEDI_DATA_FOLDER", file=sys.stderr)
		sys.exit(1)

	import_gedi_data(gedi_data_folder)
