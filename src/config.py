# Store useful variables and configuration
from pathlib import Path

# Get the root directory of the project
project_root = Path(__file__).parent.parent.absolute()

# ? Define paths relative to the project root
PATH_DATA_EXTERNAL = project_root / "data" / "external"
PATH_DATA_INTERIM = project_root / "data" / "interim"
PATH_DATA_PROCESSED = project_root / "data" / "processed"
PATH_DATA_RAW = project_root / "data" / "raw"

PATH_MODELING = project_root / "src" / "modeling"
PATH_SERVICES = project_root / "src" / "services"
PATH_FIGURES = project_root / "reports" / "figures"
