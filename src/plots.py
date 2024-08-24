# Code to create visualizations
import matplotlib.pyplot as plt


def make_map(data):
    """
    Creates a scatter plot visualization on a 2D map to display geographic data, highlighting population density 
    and median house values across different locations.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'longitude': Longitude values for the locations (x-axis).
        - 'latitude': Latitude values for the locations (y-axis).
        - 'population': Population size of each location, used to determine the size of the scatter points.
        - 'median_house_value': Median house value for each location, used to color the scatter points.

    Functionality:
    --------------
    - The scatter plot represents each location as a point, with the size of the point proportional to the population.
    - The color of each point reflects the median house value, using the 'jet' color map.
    - Points are semi-transparent to help visualize overlapping data.
    - A color bar is included to show the relationship between color intensity and median house values.
    - A legend is added to describe the population data represented by the point sizes.

    Returns:
    --------
    None
        This function directly displays the scatter plot using Matplotlib.
    """
    data.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=data["population"] / 100,
        label="population",
        figsize=(10, 7),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True,
    )
    plt.legend()
