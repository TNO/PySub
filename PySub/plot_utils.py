# -*- coding: utf-8 -*-
"""plot tools
"""
import os
import xml
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm

from matplotlib import cm
from warnings import warn
from cartopy import crs as ccrs
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Polygon, Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PySub import utils as _utils
from PySub import WMTS_utils as _WMTS_utils
from PySub import Points as _Points
from PySub.SubsidenceSuite import ModelSuite as _ModelSuite

from tkinter import Tk, Button, Entry, Label, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

WHITE_SHADOW = [pe.withStroke(linewidth=2, foreground="white")]


def get_unit_label(variable, unit):
    """Get the right unit for each variable.

    Parameters
    ----------
    variable : str
        The variable name of which the unit is requiested.
    unit : str
        The unit of length.

    Returns
    -------
    unit_label : str
        Complete unit label for the variable.

    """
    if variable.startswith("slope"):
        unit_label = f"{unit}/m"
    elif variable.startswith("concavity"):
        unit_label = f"{unit}/m²"
    elif variable.startswith("compaction"):
        unit_label = f"{unit}³"
    elif variable.endswith("rate"):
        unit_label = f"{unit}/year"
    else:
        unit_label = unit
    return unit_label


def raise_invalid_color_exception(number_of_groups):
    """Raises exception when the function get_colors_grouped determines the
    amount of colors and plottable objects are not equal in number or are not
    considered colors by matplotlib.
    """
    raise Exception(
        f"Invalid entry for group coloring. It should be a list of RGBA colors with the same length as the number of groups ({number_of_groups})."
    )


def get_colors_grouped(
    plot_objects, groups, cmap="gist_rainbow", group_colors=None
):
    """Distribute colors between the groups and have the plot objects (plot_objects)
    change in a shade of that color to better visualize connection between a group.

    Parameters
    ----------
    plot_objects : list, str
        A list of strings of which all the entries are unique. Each value
        must have a value assigned to it in the groups parameter, determined by
        the same position in the list.
    groups : list, str
        A list of strings with the groups that are going to be assigned the colors.
        The groups do not have to be unique. When group_colors id not None,
        group_colors hould have the same length as the number of unique groups
        labeled here.
    cmap : str, optional
        matplotlib.cm cmap name. The default is 'gist_rainbow'. If groups_colored
        is None, the colors for the groups will be sampled from this color map.
        If groups_colored is not None, this value will be ignored.
    group_colors : list, tuple, float, optional
        A list of tuples with RGB(A) values. The default is None. If None, the
        colors will be sampled from the given cmap. If not None, the colors for
        each group will be picked from this list.

    Returns
    -------
    colors : list, tuples, floats
        List of tuples with RGBA values.

    """
    unique_groups, index, counts = np.unique(
        groups, return_index=True, return_counts=True
    )
    sortation_index = np.argsort(index)
    unique_groups = np.array(groups)[np.sort(index)]
    number_of_groups = len(unique_groups)
    counts = counts[sortation_index]
    if group_colors is None:
        cmap = cm.get_cmap(cmap)
        color_range = cmap(np.linspace(0, 1, len(unique_groups)))
    else:
        if isinstance(group_colors, list):
            if len(group_colors) != number_of_groups:
                raise_invalid_color_exception(number_of_groups)
        else:
            raise_invalid_color_exception(number_of_groups)
        color_range = np.array(group_colors)

    _colors = []
    for g, c in zip(color_range, counts):
        faded_g = g[None, :] * np.ones(c)[:, None]

        # Exclude alpha from darkening
        faded_g[:, :-1] = g[None, :-1] * np.linspace(1, 0.5, c)[:, None]
        _colors.append(faded_g)
    counter = np.zeros(len(unique_groups)).astype(int)
    colors = []
    for occurence in plot_objects:
        is_field = [occurence.startswith(f) for f in unique_groups]
        occurence_number = counter[is_field][0]
        colors_field = _colors[np.where(is_field)[0][0]][occurence_number]
        colors.append(colors_field)
        counter[is_field] += 1
    colors = np.array(colors).tolist()
    return colors


def colors_from_cmap(cmap, number_of_colors):
    """Get a color uniformly sampled from a colormap.

    Parameters
    ----------
    cmap : str or matplotlib.colors.LinearSegmentedColormap
        The colormap from which will be sampled.
    number_of_colors : int
        The number of samples taken.

    Raises
    ------
    Exception
        When no cmap is found.

    Returns
    -------
    np.ndarray
        An array with the RGBA colors sampled from the cmap, shape (number_of_colors, 4).
        If number_of_colors > 256, can return repeating colors.

    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if isinstance(cmap, (ListedColormap, LinearSegmentedColormap)):
        return cmap(np.linspace(0, 1, number_of_colors))
    else:
        raise Exception(f"Invalid cmap: {cmap}.")


def _line_style(i):
    style = [
        (0, ()),  # solid
        (0, (1, 5)),  # dotted
        (0, (5, 5)),  # dashed
        (0, (3, 5, 1, 5)),  # dashdotted
        (0, (1, 10)),  # loosely dotted
        (0, (5, 10)),  # loosely dashed
        (0, (3, 10, 1, 10)),  # loosely dashdotted
        (0, (1, 1)),  # densely dotted
        (0, (5, 1)),  # densely dashed
        (0, (3, 1, 1, 1)),
    ]  # densely dashdotted
    return style[i]


def plot_probability_distribution(
    Model,
    values,
    probabilities=None,
    unit="cm",
    figsize=(8, 6),
    title=None,
    final=True,
    fname="",
    svg=False,
    **kwargs,
):
    """

    Parameters
    ----------
    p : list
        list of values.
    unit : str, optional
        SI unit of length. The default is 'cm'.
    figsize : tuple, float, optional
        The size of the figure in inches. The default is (8, 6).
    title : str, optional
        Title of the figure. The default is None.
    final : boolean, optional
        If True, returns None and plot becomes immuteable, If False, returns fig
        and ax object. The default is True.
    kwargs : keyword arguments, optional
        The keyword arguments for the matplotlib.pyplot.plot
        function. Which keyword arguments are available
        and which values fit with it, are noted here:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    Returns
    -------
    fig : a matpltlib.pyplot.Figure object
    ax : a matpltlib.pyplot.Ax or cartopy.mpl.geoaxes.GeoAxesSubplot object
    """
    if _utils.is_iterable(values):
        number_of_samples = len(values)
        fig, ax = plt.subplots(figsize=figsize)
        values = _utils.convert_SI(np.array(values), "m", unit)
        if probabilities is None:
            probabilities = [1 / number_of_samples] * number_of_samples
        p_df = pd.DataFrame()
        p_df["probabilities"] = np.array(probabilities) * 100
        p_df["values"] = values
        sorted_p_df = p_df.sort_values("values", ascending=False)
        sorted_p_df["cumsum"] = np.cumsum(sorted_p_df["probabilities"])
        ax.plot(
            sorted_p_df["values"].values,
            sorted_p_df["cumsum"].values,
            **kwargs,
        )

        ax.invert_xaxis()
        if title is None:
            title = "Probability distribution of maximum subsidence"

        add_title(ax, title)
        plt.xlabel(f"Maximum subsidence ({unit})")
        plt.ylabel("Probabilty (%)")
        add_horizontal_line(ax, 10, unit="m")
        add_horizontal_line(ax, 50, unit="m")
        add_horizontal_line(ax, 90, unit="m")
        plt.grid()
        if final:
            if fname is None or not fname:
                fname = Model.name + "_probability_distribution"
            savefig(Model, fname, svg=svg)
            plt.show()
        else:
            return fig, ax
    else:
        raise Exception(f"Invalid input type: {type(values)}")


def adjust_background_url(url):
    if url is not None:
        required = ["{x}", "{y}", "{z}"]
        if not all(k in url for k in required):
            raise Exception(
                "URL pointing to a tile source must contain {x}, {y}, and {z}."
            )
        _WMTS_utils.__dict__["URL"] = url


def adjust_background_keywords(keywords):
    if keywords is not None:
        if not isinstance(keywords, dict):
            raise Exception(
                "Argument keywords must be a dictionary. Current argument is of "
                f"type {type(keywords)}"
            )

        _WMTS_utils.__dict__["URL_KEYWORD_DICT"] = keywords


def set_background(
    url=None, keywords={}, arcgis_service=None, google_service=None
):
    """Set formatted text as the url at which the tiles will be fetched from.
    default can be set by:
        adjust_background(arcgis_service = 'World_Topo_Map')

    Parameters
    ----------
    url : string, optional
    Use only if not using standard arcgis or google services!
        URL pointing to a tile source and containing {x}, {y}, and {z}.
        Such as: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'
        Note that the string must not be formatted. Additional keywords can be
        added. Such as: 'https://server.arcgisonline.com/ArcGIS/rest/services/{map_service}/MapServer/tile/{z}/{y}/{x}.jpg'
        When other keywords are added than {x}, {y}, and {z}, add the keywords
        by setting the "keywords" argument as for instance:
            keywords = {'map_service': 'World_Shaded_Relief'}

        Examples:
            arcgis service url: ('https://server.arcgisonline.com/ArcGIS/rest/services/{map_service}/MapServer/tile/{z}/{y}/{x}.jpg')
            google service url: ('https://mts0.google.com/vt/lyrs={map_service}@177000000&hl=en&src=api&x={x}&y={y}&z={z}&s=G')
    keywords : dict, optional
        Use only if not using standard arcgis or google services!
        When other keywords are added to the url (other than than {x}, {y},
        and {z}), add the keywords by setting this argument as for instance:
            keywords = {'map_service': 'World_Shaded_Relief'}
    arcgis_service: string, optional
        See for available options:
            https://server.arcgisonline.com/ArcGIS/rest/services/
        For instance (the default is):
            adjust_background(arcgis_service = 'World_Topo_Map')
    google_service: string, optional
        Choose from: "street", "satellite", "terrain", "only_streets"
        For instance:
            adjust_background(arcgis_service = 'only_streets')

    """
    default_keywords = _WMTS_utils.__dict__["URL_KEYWORD_DICT"]
    keywords = set_defaults(keywords, defaults=default_keywords)
    if arcgis_service is not None:
        url = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "{map_service}/MapServer/tile/{z}/{y}/{x}.jpg"
        )
        keywords["map_service"] = arcgis_service
        adjust_background_keywords(keywords)
    elif google_service is not None:
        url = (
            "https://mts0.google.com/vt/lyrs={map_service}"
            "@177000000&hl=en&src=api&x={x}&y={y}&z={z}&s=G"
        )
        styles = ["street", "satellite", "terrain", "only_streets"]
        if google_service not in styles:
            raise ValueError(
                f"The {google_service} service does not exist. "
                f"Choose from {styles}"
            )
        style_dict = {
            "street": "m",
            "satellite": "s",
            "terrain": "t",
            "only_streets": "h",
        }
        keywords["map_service"] = style_dict[google_service]

    adjust_background_url(url)
    adjust_background_keywords(keywords)


def add_background(ax=None, zoom_level=10):
    """Add a WMTS map background to an existing figure.

    Parameters
    ----------
    ax : a matpltlib.pyplot.Ax or cartopy.mpl.geoaxes.GeoAxesSubplot object
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    """
    ax.add_image(_WMTS_utils.Tiles("RGBA"), zoom_level)


def get_crs(epsg):
    crs = ccrs.epsg(epsg)
    return crs


def get_background(
    extent,
    fig=None,
    ax=None,
    basemap=True,
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
):
    """Create a figure and axis with a certain extent with the possibility to
    add a WMTS map background.

    Parameters
    ----------
    extent : list, int/float
            List with 4 values representing the extend of the figure
            of the model grid:
            [0] lower x
            [1] upper x
            [2] lower y
            [3] upper y..
    fig : a matpltlib.pyplot.Figure object
    ax : a matpltlib.pyplot.Ax or cartopy.mpl.geoaxes.GeoAxesSubplot object
    basemap : bool, optional
        The default is True. When True, will plot a WMTS as background. When
        False, not.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    figsize : tuple, int/float, optional
        The size of the image in inches
    espg : int, optional
        Coordinate system. The default is 28992.

    Returns
    -------
    fig, ax
    """
    crs = get_crs(epsg)
    if ax is None:
        ax = plt.subplot(projection=crs)
        if crs is None:
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
        else:
            ax.set_extent(extent, crs=crs)
    else:
        ax = ax
    if fig is None:
        fig = plt.gcf()
    else:
        fig = fig
    fig.set_size_inches(figsize)
    x_ticks = np.array(ax.get_xticks())
    x_ticks = x_ticks[(x_ticks < extent[0]) & (x_ticks > extent[2])]
    y_ticks = np.array(ax.get_yticks())
    y_ticks = y_ticks[(y_ticks < extent[1]) & (y_ticks > extent[3])]

    ax.set_xticks(ax.get_xticks(), ccrs.epsg(epsg))
    ax.set_yticks(ax.get_yticks(), ccrs.epsg(epsg))
    if basemap:
        add_background(ax=ax, zoom_level=zoom_level)

    return fig, ax


def add_horizontal_line(ax, horizontal_line, unit="cm"):
    """Add a red horizontal line to a plot.

    Parameters
    ----------
    ax : matplotlib Axes object
        The figure you want the errorbars to be plotted in.
    value : float/dict
        When a float, it must be the value on the y-axis at which the horizontal
        line will be placed. When a dict, the key of the dictionary will be the label
        of the line and the entry will be the value on the y-axis the horizontal
        line will be plotted along.
    unit : str, optional
        The SI unit the value will need to be converted to. The default is 'cm'.

    Returns
    -------
    None.

    """
    if horizontal_line is not None:
        if _utils.is_number(horizontal_line):
            horizontal_line = _utils.convert_SI(horizontal_line, "m", unit)
            ax.axhline(y=horizontal_line, color="r", linestyle="-")
        elif isinstance(horizontal_line, dict):
            for label, value in horizontal_line.items():
                value = _utils.convert_SI(value, "m", unit)
                ax.axhline(y=value, color="r", linestyle="-", label=label)


def add_colorbar(
    ax, contours, fig, contour_lines=None, pad=0.01, colorbar_kwargs={}
):
    """Add colorbar to existing axis.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The next to which the colorbar will be plotted.
    contours : matplotlib.contour.QuadContourSet
        The matplotlib contour set on which the colorbar will base its
        color and values.
    fig : matplotlib.figure.Figure
        Figure into which the colorbar will be plotted.
    contour_lines : matplotlib.contour.QuadContourSet, optional
        The matplotlib contour set on which the colorbar will base the
        color and values of its lines dicretizing the contours. The default is
        None.
    pad : float
        Value for the distance from the axis the colorbar as a factor of the x-axis.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    """
    number_of_axes = len(fig.axes)
    cax = fig.add_axes([0, 0, 0.1, 0.1])
    if number_of_axes > 1:
        colorbar_width = 0.02
    else:
        colorbar_width = 0.04

    posn = ax.get_position()
    cax.set_position(
        [posn.x0 + posn.width + pad, posn.y0, colorbar_width, posn.height]
    )
    colorbar_kwargs = {"cax": cax, "ax": ax, **colorbar_kwargs}
    colorbar = plt.colorbar(contours, **colorbar_kwargs)
    if contour_lines is not None:
        colorbar.add_lines(contour_lines)
    return colorbar


def add_north_arrow():
    pass


def add_scalebar(ax, extent):
    """Add a scalebar based on the extent of the axis.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot onto which the title will be added.
    extent : list, int/float
            List with 4 values representing the extend of the figure
            of the model grid:
            [0] lower x
            [1] lower y
            [2] upper x
            [3] upper y.
    """
    x_size = np.diff(extent)[0]
    if x_size == 0:
        raise Exception(
            "The extent of the figure is 0. No figure can be displayed"
        )
    ten_percent = x_size / 10

    # oom = order of magnitude
    scale_bar_oom = _utils.get_order_of_magnitude(ten_percent)
    scale_bar_size = round(ten_percent, -(scale_bar_oom))

    if scale_bar_oom < 3:
        scale_bar_text = scale_bar_size
        unit = "m"
    elif scale_bar_oom >= 3:
        scale_bar_text = scale_bar_size / 1000
        unit = "km"
    elif scale_bar_oom < 0:
        scale_bar_text = scale_bar_size * 100
        unit = "cm"
    fontprops = fm.FontProperties()
    scalebar = AnchoredSizeBar(
        ax.transData,
        scale_bar_size,
        f"{scale_bar_text} {unit}",
        "lower right",
        pad=0.1,
        color="k",
        frameon=False,
        size_vertical=scale_bar_size // 20,
        fontproperties=fontprops,
    )

    ax.add_artist(scalebar)
    return scalebar


def add_title(ax, title):
    """Add title to a subplot.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot onto which the title will be added.
    title : str
        The title.
    """

    if title is not None:
        if isinstance(title, str):
            ax.set_title(title)
        else:
            raise Exception(
                f"Invalid object type for title: {type(title)}. Use a string."
            )


def add_custom_legend(
    ax,
    types=[],
    kwargs=[],
    labels=[],
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
):
    """Add a custom legend to a figure.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot onto which the title will be added.
    types : str or list of str
        An indication of how each legend entry should look like. If it is a list,
        the list should have the same number of entries as the kwargs and labels
        and contains strings only. If it is a string, the same type will be chosen
        for all entries to the legend. The default is []. The valid types are:
            'polygon', 'point' and 'line'.
    kwargs : list of dict, optional
        A list with the same length as types and labels. This list must contain
        the keyword arguments in a dictionary. The relevant keyword arguments are
        of the functions/classes:
            - matplotlib.patches.Polygon if type == polygon
            - matplotlib.pyplot.plot if type == line
            - matplotlib.pyplot.scatter if type == point
    labels : a list of strings, optional
        A list of string with the same length as the lists of types and kwargs.
        The default is []. Each entry will be the label to the legend entry.
    """
    if type(types) == str:
        types = [types] * len(labels)
    valid_types = ["polygon", "line", "point"]
    handles = []
    for t, k, l in zip(types, kwargs, labels):
        if not t in valid_types:
            raise Exception(f"{t} is not a valid type use: {valid_types}")
        if t.lower() == "polygon":
            handle = Polygon(np.zeros((2, 2)), **k)
        if t.lower() == "line":
            handle = plt.plot(0, 0, **k)
        if t.lower() == "point":
            handle = plt.scatter(0, 0, **k)
        handles.append(handle)
    ax.legend(
        handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox_to_anchor
    )


def _dummy_mappable(
    x, y, contour_levels, contour_steps, contourf_kwargs={}, contour_kwargs={}
):
    dummy_f, dummy_axis = plt.subplots()
    min_range, max_range = np.min(contour_levels), np.max(contour_levels)
    min_x, max_x = np.min(x), np.max(x)
    xx, yy = np.meshgrid(x, y)
    dy = (max_range + contour_steps) - (min_range - contour_steps)
    dx = max_x - min_x
    slope = dy / dx
    data = min_range + contour_steps + (xx - min_x) * slope
    dummy_mappable = dummy_axis.contourf(
        x, y, data, levels=contour_levels, **contourf_kwargs
    )
    dummy_mappable_lines = dummy_axis.contour(
        x, y, data, levels=contour_levels, **contour_kwargs
    )
    plt.close()
    return dummy_mappable, dummy_mappable_lines


def add_contours(
    ax,
    x,
    y,
    data,
    contour_levels,
    contour_steps,
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
):
    """Add contours to an existing matplotlib axis.


    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot onto which the contours will be added.
    x : list/np.array, float/int, optional
        1D itterable object with the x-coordinates of the data. The default is None.
        When None, the data is not plotted, but the map is.
    y : TYPE, optional
        1D itterable object with the y-coordinates of the data. The default is None.
        When None, the data is not plotted, but the map is.
    data : 2D numpy array or list
        Any variable as long as it has two dimension that equal the amount of
        entries for x and y. The default is None.
    contour_levels : list, float
        Draw contour lines at the specified levels. The values must be in increasing order.
        The default is None. When None, the contour lines will be chosen based
        on the subsidence data and the contour_steps parameter.
    contour_steps : float/int
        The difference in values between the contour levels. The default is 0.01.
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.

    Returns
    -------
    filled_contours : matplotlib.contour.QuadContourSet
        Object with the filled contours.
    contour_lines : matplotlib.contour.QuadContourSet
        Object with the contour lines.

    """
    contour_levels = set_contour_levels(
        data, contour_levels=contour_levels, contour_steps=contour_steps
    )
    shift_cmap(contour_levels, contourf_kwargs)
    shift_cmap(contour_levels, contour_kwargs)

    if contour_levels is None:
        warn("Warning: Not enough contour levels to plot contours.")
        return None, None
    elif len(contour_levels) > 0:
        if len(contour_levels) > 100:
            warn(
                f"Warning: {len(contour_levels)} contours found. This can take very long."
            )
        if np.min(data) < -contour_steps or np.max(data) > contour_steps:
            filled_contours = ax.contourf(
                x, y, data, levels=contour_levels, **contourf_kwargs
            )
            contour_lines = ax.contour(
                x, y, data, levels=contour_levels, **contour_kwargs
            )
            ax.clabel(contour_lines, contour_levels, **clabel_kwargs)

        filled_contours, contour_lines = _dummy_mappable(
            x,
            y,
            contour_levels,
            contour_steps,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
        )

        return filled_contours, contour_lines

    else:
        warn("Warning: Not enough contour levels to plot contours.")
        return None, None


def add_geometries(
    ax, geometries=[], scatter_kwargs={}, shape_kwargs={}, raster_kwargs={}
):
    """Add geometry related objects to a matplolib axis.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot onto which the contours will be added.
    geometries : list, Geometry object, optional
        PySub Geometry object. The default is [].
    scatter_kwargs : dict, optional
        Dictionary with keyword arguments for matplotlib.pyplot.scatter. The default is {}.
        Ignored if Geometry.type is not "point".
    shape_kwargs : dict, optional
        Dictionary with keyword arguments for matplotlib.matplotlib.patches.polygon.
        The default is {}.
        Ignored if Geometry.type is not "polygon".
    raster_kwargs : dict, optional
        Dictionary with keyword arguments for matplotlib.pyplot.imshow. The default is {}.
        Ignored if Geometry.type is not "raster".

    Raises
    ------
    Exception
        If geometry entry is of invalid type.

    """
    types = [
        geom.type if hasattr(geom, "type") else None for geom in geometries
    ]
    wrong = [t is None for t in types]
    if any(wrong):
        raise Exception(
            f"Invalid type of geometry: {type(np.array(geometries)[wrong][0])}."
        )
    unique_types, inverse, count = np.unique(
        types, return_counts=True, return_inverse=True
    )
    count_dict = {t: c for t, c in zip(unique_types, count)}
    index_dict = {
        t: np.where(inverse == i)[0] for i, t in enumerate(unique_types)
    }
    for t in unique_types:
        for i, geom_index in enumerate(index_dict[t]):
            geom = geometries[geom_index]
            if t == "raster":
                kwargs = raster_kwargs
            elif t == "point":
                kwargs = scatter_kwargs
            elif t == "polygon":
                kwargs = shape_kwargs
            else:
                raise Exception(f"Invalid type of geometry: {geom.type}.")
            individual_kwargs = _utils.pick_from_kwargs(
                kwargs, i, int(count_dict[geom.type])
            )
            geom.plot(ax, individual_kwargs)


def add_shapes(ax, geometries=[], shape_kwargs={}):
    """Add Polygon objects to the ax object.

    Parameters
    ----------
    ax : matplotlib Axes object
        The figure you want the shapes to be plotted in.
    geometries : shapely.Polygon object, optional
        The shapes that will be plotted in the ax. The default is [].
    shape_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted shapes.
        The default is {}. See SubsidenceModel attribute shape_defaults
        for additional information.
    """
    for geom in geometries:
        if np.array(geom).shape == (2,):
            plt.scatter(geom[0], geom[1], edgecolors="k")
        else:
            if geom[0] == geom[-1]:
                closed = True
            else:
                closed = False
            p = Polygon(geom, closed=closed, **shape_kwargs)
            ax.add_patch(p)


def add_raster(ax, raster, block_value=0, raster_kwargs={}):
    if not isinstance(raster, dict):
        raise Exception(
            'Add_raster takes an axis and dictionary as argument. The dictionary should have the keys "X", "Y", "mask".'
        )
    test = all([key in raster.keys() for key in ["X", "Y", "mask"]])
    if not test:
        raise Exception(
            'Add_raster takes an axis and dictionary as argument. The dictionary should have the keys "X", "Y", "mask".'
        )
    bounds = _utils.bounds_from_xy(raster["X"], raster["Y"])
    extent = np.array(bounds)[[0, 2, 1, 3]]
    if block_value is not None:
        masked_raster = np.ma.masked_where(
            raster["mask"] == block_value, raster["mask"]
        )

    im = ax.imshow(masked_raster, extent=extent, **raster_kwargs)
    return im


def add_rasters(ax, rasters=[], raster_kwargs={}):
    for raster in rasters:
        add_raster(ax, raster, raster_kwargs)


def add_annotations(ax, labels, points, annotation_kwargs={}):
    annotations = [
        ax.annotate(label, point, **annotation_kwargs)
        for label, point in zip(labels, points)
    ]
    adjust_text(
        annotations, arrowprops=dict(arrowstyle="-", color="k", lw=0.5), ax=ax
    )


def add_lines(ax=None, line=None, annotation_kwargs={}):
    """Add lines to existing figure subplot.

    Parameters
    ----------
    ax : matplotlib Axes object
        The figure you want the lines to be plotted in.
    line : list or dict, optional
        When a list, it must be a list of tuples containg x and y coordinates.
        When a dictionary, it must be dict with the shape {'label1': (x1, y1), 'label2': (x1,y1)}.
        The keys of the dictionary will be the labels which will annotated next to the line nodes.
        The default is None. When None, no lines will be plotted.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.

    Returns
    -------
    fig, ax
        Only returns a new fig and ax when ax in fuction parameters is None.
    """
    remember = ax
    if ax is None:
        fig, ax = plt.subplots()

    if line is not None:
        if _utils.is_iterable(line) and type(line) != dict:
            line = np.array(line)
            if line.shape[1] == 2:
                ax.plot(line.T[0], line.T[1], "ro-")
            else:
                raise (
                    Exception(
                        "Line object can be dict or list of sets with 2 nummerical entries."
                    )
                )
        elif type(line) == dict:
            labels = line.keys()
            points = list(line.values())
            xs, ys = np.array(list(line.values())).T
            ax.plot(xs, ys, "ro-")
            add_annotations(
                ax, labels, points, annotation_kwargs=annotation_kwargs
            )
        else:
            raise (
                Exception("Line object can be dict or list with 2x2 shape.")
            )
    if remember is None:
        return fig, ax


def add_points(
    ax=None, points=None, labels=None, scatter_kwargs={}, annotation_kwargs={}
):
    """Add points to existing matplotlib axes subplot

    Parameters
    ----------
    ax : matplotlib Axes object
        The figure you want the lines to be plotted in.
    points : list or dict, optional
        When a list, it must be a list of tuples containg x and y coordinates.
        When a dictionary, it must be dict with the shape {'label1': (x1, y1), 'label2': (x1,y1)}.
        The keys of the dictionary will be the labels which will annotated next to the points.
        The default is None. When None, no points will be plotted.
    scatter_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted points.
        The default is {}. See SubsidenceModel attribute point_defaults
        for additional information.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.

    Returns
    -------
    fig, ax
        Only returns a new fig and ax when ax in fuction parameters is None.
    """
    remember = ax
    if ax is None:
        fig, ax = plt.subplots()

    if points is not None:
        if _utils.is_iterable(points) and type(points) != dict:
            points = np.array(points)

            if points.shape[1] == 2:
                if "zorder" not in scatter_kwargs.keys():
                    zorder = max([_.zorder for _ in ax.get_children()]) + 1
                    scatter_kwargs_adjusted = scatter_kwargs
                else:
                    zorder = scatter_kwargs["zorder"]
                    scatter_kwargs_adjusted = {
                        k: v
                        for k, v in scatter_kwargs.items()
                        if k != "zorder"
                    }
                ax.scatter(
                    points.T[0],
                    points.T[1],
                    zorder=zorder,
                    **scatter_kwargs_adjusted,
                )
            else:
                raise (
                    Exception(
                        "point object can be dict or list of sets with 2 nummerical entries."
                    )
                )
            if labels is not None:
                add_annotations(
                    ax, labels, points, annotation_kwargs=annotation_kwargs
                )

        elif type(points) == dict:
            labels = points.keys()
            p = list(points.values())
            xs, ys = np.array(list(points.values())).T
            ax.scatter(xs, ys, **scatter_kwargs)
            if labels is not None:
                add_annotations(
                    ax, labels, p, annotation_kwargs=annotation_kwargs
                )

        else:
            raise (
                Exception("point object can be dict or list with 2x2 shape.")
            )
    if remember is None:
        return fig, ax


def add_errorbar(
    ax=None,
    x=None,
    y=None,
    upper_limits=None,
    lower_limits=None,
    c=None,
    label="_nolabel_",
    errorbar_marker="_",
    errorbar_kwargs={},
):
    """Add points with vertical errorbars to an existing subplot of a 1D series.

    Parameters
    ----------
    ax : matplotlib Axes object
        The figure you want the errorbars to be plotted in.
    x : 1D list or np.ndarray, optional
        The x-axis values of the plotted points. The default is None. When
        None, no errorbars will be plotted.
    y : 1D list or np.ndarray, optional
        The y-axis values of the plotted points. The default is None. When
        None, no errorbars will be plotted.
    upper_limits : The upper error of the point y-values. The default is None.
        When None, no errorbars will be plotted.
    lower_limits : 1D list or np.ndarray, optional
        The lower error of the point y-values. The default is None.
        When None, no errorbars will be plotted.
    c : 1D list or np.ndarray, optional
        SIngle entry of matplotlib color object. The default is None.
    label : str
        The label with which the points will be indicated in a legend.
        The default is '_nolabel_', which means it won;t show up in a legend.
    errorbar_marker : TYPE, optional
        DESCRIPTION. The default is '_'.
    errorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the errorbars annotations.
        The default is {}. See SubsidenceModel attribute errorbar_defaults
        for additional information.
    """
    if ax is None:
        fig, ax = plt.subplots()
    to_plot_test = [x, y, upper_limits, lower_limits]
    plot_test = []
    for i in to_plot_test:
        plot_test.append(i is None)

    if np.logical_not(np.array(plot_test)).all():
        l1, cap1, _ = ax.errorbar(
            x,
            y,
            yerr=upper_limits,
            uplims=True,
            lolims=False,
            c=c,
            label=label,
            **errorbar_kwargs,
        )
        for cap in cap1:
            cap.set_marker(errorbar_marker)
        l2, cap2, _ = ax.errorbar(
            x,
            y,
            yerr=lower_limits,
            uplims=False,
            lolims=True,
            c=c,
            label="_nolabel_",
            **errorbar_kwargs,
        )
        for cap in cap2:
            cap.set_marker(errorbar_marker)


def add_filled_area(ax, x1, y1, y2, label=None, kwargs={}):
    """

    Parameters
    ----------
    ax : matplotlib Axes object
        The figure you want the filled area to be plotted in.
    other keywords:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
    """
    if label is None:
        label = "_nolabel_"
    return ax.fill_between(x1, y1, y2, label=label, **kwargs)


def shiftedColorMap(
    cmap,
    start=0,
    midpoint=0.5,
    stop=1.0,
    levels=[0],
    alpha=1,
    midpoint_alpha=0,
):
    """https://stackoverflow.com/a/20528097
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and 'midpoint'.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, 'midpoint'
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          'midpoint' and 1.0.
      levels : list
          The contour levels to which the cmap is shifted
    """
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    if len(levels) == 1:
        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict["red"].append((si, r, r))
            cdict["green"].append((si, g, g))
            cdict["blue"].append((si, b, b))
            if midpoint + 0.025 > si > midpoint - 0.025:
                a = midpoint_alpha
            else:
                a = alpha
            cdict["alpha"].append((si, a, a))
    else:
        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict["red"].append((si, r, r))
            cdict["green"].append((si, g, g))
            cdict["blue"].append((si, b, b))

            if (
                midpoint + (1 / len(levels))
                > si
                > midpoint - (1 / len(levels))
            ):
                a = midpoint_alpha
            else:
                a = alpha
            cdict["alpha"].append((si, a, a))

    newcmap = LinearSegmentedColormap(f"shifted_{cmap.name}", cdict)
    return newcmap


def shift_cmap(levels, kwarg_dict, alpha=1):
    if levels is None:
        return
    if "cmap" in kwarg_dict.keys():
        if kwarg_dict["cmap"] is not None:
            if "alpha" in kwarg_dict.keys():
                alpha = kwarg_dict["alpha"]
                del kwarg_dict["alpha"]
                kwarg_dict["_alpha"] = alpha
            if "_alpha" in kwarg_dict.keys():
                alpha = kwarg_dict["_alpha"]
            midpoint = (0 - np.min(levels)) / (np.max(levels) - np.min(levels))
            if type(kwarg_dict["cmap"]) == str:
                cmap = cm.get_cmap(kwarg_dict["cmap"])
                kwarg_dict["ccmap"] = kwarg_dict["cmap"]
                shifted_cmap = shiftedColorMap(
                    cmap,
                    start=0,
                    midpoint=midpoint,
                    stop=1,
                    levels=levels,
                    alpha=alpha,
                )
                kwarg_dict["cmap"] = shifted_cmap
            elif type(kwarg_dict["cmap"]) == LinearSegmentedColormap:
                if kwarg_dict["cmap"].name.startswith("shifted_"):
                    shifted_cmap = shiftedColorMap(
                        cm.get_cmap(kwarg_dict["cmap"].name[8:]),
                        start=0,
                        midpoint=midpoint,
                        stop=1,
                        levels=levels,
                        alpha=alpha,
                    )
                    kwarg_dict["cmap"] = shifted_cmap
                else:
                    shifted_cmap = shiftedColorMap(
                        kwarg_dict["cmap"],
                        start=0,
                        midpoint=midpoint,
                        stop=1,
                        levels=levels,
                        alpha=alpha,
                    )
                    kwarg_dict["cmap"] = shifted_cmap
            else:
                raise Exception(
                    'Invalid colormap type: {type(type(kwarg_dict["cmap"]))}'
                )


def set_contour_levels(
    data=[0], contour_levels=None, contour_steps=0.01, drop_value=0
):
    """Get a set of contour levels based on the available data.

    Parameters
    ----------
    data : numpy ndarray
        The data for which the contour values need plotting.
    contour_levels : list, optional
        List of contour values. The default is None. When None, the levels are automatically determined.
    contour_steps : float, optional
        The steps taken in the data. The default is 0.01. This causes a linear interval.


    Returns
    -------
    contour_levels : list
        contour levels.

    """
    if contour_levels is None:
        contour_levels = np.array(
            _utils.stepped_space(np.min(data), np.max(data), contour_steps)
        )
    elif not _utils.is_iterable(contour_levels):
        raise Exception(
            f"Invalid contour level type: {type(contour_levels)}. Use list or other itterable to define contour levels explicitly, or None to define it on value range."
        )
    contour_levels = np.array(contour_levels)
    if len(contour_levels) == 0:
        contour_levels = None
    else:
        if min(contour_levels) > -contour_steps:
            clist = [-contour_steps]
            for l in contour_levels:
                clist.append(l)
            contour_levels = np.array(clist)
        if max(contour_levels) < contour_steps:
            clist = list(contour_levels)
            clist.append(contour_steps)
            contour_levels = np.array(clist)
        contour_levels = contour_levels[
            np.logical_not(np.isclose(contour_levels, drop_value, atol=1e-14))
        ]
    return contour_levels


def set_defaults(input_dict, defaults={}):
    """Sets the defaults if they are not specified in input_dict

    Parameters
    ----------
    input_dict : dict
    defaults : dict, optional
        default values for keys. The default is {}.

    Returns
    -------
    input_dict : dict
        The input_dict with added default values, if those values have not been specified in input_dict.

    """
    if type(input_dict) != dict or type(defaults) != dict:
        raise (
            TypeError(
                "The input values for set_defaults need to be dictionaries"
            )
        )
    output_dict = input_dict.copy()
    if "colors" in input_dict.keys():
        try:
            del defaults["c"]
        except:
            pass
        try:
            del defaults["cmap"]
        except:
            pass
    if "cmap" in input_dict.keys():
        try:
            del defaults["c"]
        except:
            pass
        try:
            del defaults["colors"]
        except:
            pass
    if "c" in input_dict.keys():
        try:
            del defaults["cmap"]
        except:
            pass
        try:
            del defaults["colors"]
        except:
            pass
    for key, value in defaults.items():
        if key not in output_dict.keys():
            output_dict[key] = value
    return output_dict


def get_cross_section(x, y, data, line, num):
    number_of_points = len(line)

    values = np.zeros(shape=(len(data), num * (number_of_points - 1)))
    distances = np.array([])
    start_x = 0
    inflection_distances = []
    for l in range(number_of_points - 1):
        A, B = list(line)[l], list(line)[l + 1]
        segment = (A, B)
        x_world, y_world = np.array(list(zip(*segment)))
        col = len(x) * (x_world - x.min()) / x.ptp()
        row = len(y) * (y_world - y.min()) / y.ptp()
        _A, _B = np.vstack((col, row)).T
        distance = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        _distance = np.linspace(start_x, start_x + distance, num)
        start_x += distance
        if l == 0:
            inflection_distances.append(_distance[0])

        distances = np.hstack((distances, _distance))
        inflection_distances.append(_distance[-1])

        for i in range(len(data)):
            segment_values = _utils.get_values_cross_section(
                _A, _B, data[i].T, num=num
            )
            values[i][l * num : (l + 1) * num] = segment_values

    return values, distances, np.array(inflection_distances)


def add_cross_section(
    ax,
    x,
    y,
    data,
    line,
    name="_nolegend_",
    inflection_point_names=None,
    c=None,
    num=1000,
    plot_kwargs={},
    annotation_kwargs={},
):
    """Add cross sections through the data to an existing axis.

    Data in grid format gets interpolated linearly between the points in line.

    Parameters
    ----------
    ax : matplotlib axes object
        The ax the cross section will be plotted in.
    x : list or np.ndarray, int/float
        The x-coordinates of the data.
    y : list or np.ndarray, int/float
        The y-coordinates of the data.
    data : list of 2D np.ndarray, int/float
        list of data which will be plotted.
    line : list/dict, int/float optional
        When a list, it must be a list of tuples containg x and y coordinates. Example of line made out of two points: ((120, 130), (560, 6009))
        When a dictionary, it must be dict with the shape {'label1': (x1, y1), 'label2': (x1,y1)}.
        The keys of the dictionary will be the labels which will annotated next to the line nodes.
        The default is None. When None, no lines will be plotted.
    name : list, str, optional
        The names each of the lines plotted will be given. Needs to have the same length as the data parameter. The default is '_nolegend_', which makes sure the plotted lines do not show up in a legend.
    inflection_point_names : list, str, optional
        The labels for the inflection points. The default is None. When None,
        the inflection points will be named according to alphabetical order.
    c : 2d list ot tuple, optional
        The colours assigned to the lines, must have the ssame length as the
        data parameter. The default is None.
    num : int, optional
        Amount of interpolation points between the points in line.
        The default is 1000.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines.
        The default is {}. See SubsidenceModel attribute plot_defaults
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.

    Returns
    -------
    None.

    """
    number_of_points = len(line)
    if _utils.is_iterable(line):
        line_dict = {}
        if inflection_point_names is None:
            for i, point in enumerate(line):
                line_dict[string.ascii_uppercase[i % 26]] = point
        elif _utils.is_iterable(inflection_point_names):
            if len(inflection_point_names) != number_of_points:
                warn(
                    "Warning: Invalid number of names for inflection point. The entry must have the same length as the amount of points in the line list/dict."
                )
                for i, point in enumerate(line):
                    line_dict[string.ascii_uppercase[i % 26]] = point
            else:
                for i, point in enumerate(line):
                    line_dict[inflection_point_names[i]] = point
        line = line_dict

    values, distances, inflection_distances = get_cross_section(
        x, y, data, line.values(), num
    )

    for l in range(number_of_points - 1):
        if l == 0:
            labels = [list(line.keys())[l]]
            points = [(inflection_distances[0], 0)]
            ax.axvline(
                x=inflection_distances[0],
                c="k",
                linestyle="dashed",
                alpha=0.25,
                label="_nolegend_",
            )
            add_annotations(
                ax, labels, points, annotation_kwargs=annotation_kwargs
            )
        labels = [list(line.keys())[l + 1]]
        points = [(inflection_distances[l + 1], 0)]
        ax.axvline(
            x=inflection_distances[l + 1],
            c="k",
            linestyle="dashed",
            alpha=0.25,
            label="_nolegend_",
        )
        add_annotations(
            ax, labels, points, annotation_kwargs=annotation_kwargs
        )

    for i in range(len(data)):
        ax.plot(distances, values[i], c=c[i], label=name[i], **plot_kwargs)


def set_ylim(
    ax, ylim=None, y_axis_exageration_factor=1, unit_in="m", unit_out="m"
):
    """Set the range of the y-axis of a plot based.

     Parameters
     ----------
    ax : matplotlib axes object
         The ax the from y-axis will be set.
     ylim : tuple, optional
         The extend of the y-axis. The default is None. If None, the default y-axis
         extend will be used, and the y_axis_exageration_factor to multiply its length.
     y_axis_exageration_factor : float, optional
         The y-axis' length can be extended by this factor. The default is 1.
     unit_in : str, optional
         SI-unit for length in which the data is in. The default is 'm'.
     unit_out : str, optional
         SI-unit for length in which the data will be plotted. The default is 'm'.

    """
    if ylim is None:
        ax.set_ylim(
            [
                np.min(ax.get_ylim()) * y_axis_exageration_factor,
                np.max(ax.get_ylim()),
            ]
        )
    else:
        ylim = _utils.convert_SI(np.array(ylim), unit_in, unit_out)
        ax.set_ylim(ylim)


def plot_2D_data(
    extent,
    x=None,
    y=None,
    data=None,
    figsize=(8, 8),
    contour_levels=None,
    contour_steps=0.01,
    title=None,
    geometries=[],
    basemap=True,
    zoom_level=10,
    epsg=28992,
    shape_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    scatter_kwargs={},
    raster_kwargs={},
):
    """Plot contours of 2D data.

    Parameters
    ----------
    extent : list, int/float
            List with 4 values representing the extend of the figure
            of the model grid:
            [0] lower x
            [1] upper x
            [2] lower y
            [3] upper y.
    x : list/np.array, float/int, optional
        1D itterable object with the x-coordinates of the data. The default is None.
        When None, the data is not plotted, but the map is.
    y : TYPE, optional
        1D itterable object with the y-coordinates of the data. The default is None.
        When None, the data is not plotted, but the map is.
    data : 2D numpy array or list, optional
        Any variable as long as it has two dimensions that equal the amount of
        entries for x and y. The default is None, when None, doesn't plot any data,
        just the background according to the given parameters. When the data is 3D
        use the plot_3D_data function to plot slices of 2D figures.
    figsize : tuple, float/int, optional
        The figure size in inches. The default is (8, 8).
    contour_levels : list, float, optional
        Draw contour lines at the specified levels. The values must be in increasing order.
        The default is None. When None, the contour lines will be chosen based
        on the subsidence data and the contour_steps parameter.
    contour_steps : float/int, optional
        The difference in values between the contour levels. The default is 0.01.
    title : str, optional
        Title to the figure. The default is None, when title is None there
        will be no title added to the figure.
    geometries : list, matplotlib.patches.Polygon, optional
        The shapes of the reservoirs to be plotted. The default is [].
    basemap : bool, optional
        The default is True. When True, will plot a WMTS as background. When
        False, not.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    espg : int, optional
        Coordinate system. The default is 28992.
    shape_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted shapes.
        The default is {}. See SubsidenceModel attribute shape_defaults
        for additional information.
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    scatter_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted points.
        The default is {}. See SubsidenceModel attribute point_defaults
        for additional information.

    Returns
    -------
    fig, ax
    """

    fig, ax = get_background(
        extent, basemap=basemap, zoom_level=zoom_level, epsg=epsg
    )
    fig.set_size_inches(figsize)

    add_geometries(
        ax,
        geometries=geometries,
        scatter_kwargs=scatter_kwargs,
        shape_kwargs=shape_kwargs,
        raster_kwargs=raster_kwargs,
    )
    to_plot_test = [x, y, extent, data]
    plot_test = []
    for i in to_plot_test:
        plot_test.append(i is None)

    if np.logical_not(np.array(plot_test)).all():
        contours, contour_lines = add_contours(
            ax,
            x,
            y,
            data,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
        )
        if contours is not None:
            add_colorbar(
                ax,
                contours,
                fig,
                contour_lines=contour_lines,
                colorbar_kwargs=colorbar_kwargs,
            )
    add_north_arrow()
    add_scalebar(ax, extent)
    add_title(ax, title)
    return fig, ax


def plot_3D_data(
    extent,
    x,
    y,
    data,
    figsize=(8, 8),
    contour_levels=None,
    contour_steps=0.01,
    title=None,
    geometries=[],
    basemap=True,
    zoom_level=10,
    epsg=28992,
    shape_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    raster_kwargs={},
    scatter_kwargs={},
):
    """Plot contours of 2D data.

    Parameters
    ----------
    extent : list, int/float
            List with 4 values representing the extend of the figure
            of the model grid:
            [0] lower x
            [1] lower y
            [2] upper x
            [3] upper y.
    x : list/np.array, float/int, optional
        1D itterable object with the x-coordinates of the data. The default is None.
        When None, the data is not plotted, but the map is.
    y : TYPE, optional
        1D itterable object with the y-coordinates of the data. The default is None.
        When None, the data is not plotted, but the map is.
    data : 3D numpy array or list, optional
        Any variable as long as it has three dimensions. The first dimension will
        determine the number of figures. The second and third dimensions represent
        the entries for x and y. The default is None, when None, doesn't plot any data,
        just the background according to the given parameters. When the data is 2D
        use the plot_2D_data function to plot slices of 2D figures.
    figsize : tuple, float/int, optional
        The figure size in inches. The default is (8, 8).
    contour_levels : list, float, optional
        Draw contour lines at the specified levels. The values must be in increasing order.
        The default is None. When None, the contour lines will be chosen based
        on the subsidence data and the contour_steps parameter.
    contour_steps : float/int, optional
        The difference in values between the contour levels. The default is 0.01.
    title : str, optional
        Title to the figure. The default is None, when title is None there
        will be no title added to the figure.
    geometries : list, matplotlib.patches.Polygon, optional
        The shapes of the reservoirs to be plotted. The default is [].
    basemap : bool, optional
        The default is True. When True, will plot a WMTS as background. When
        False, not.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    espg : int, optional
        Coordinate system. The default is 28992.
    shape_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted shapes.
        The default is {}. See SubsidenceModel attribute shape_defaults
        for additional information.
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    scatter_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted points.
        The default is {}. See SubsidenceModel attribute point_defaults
        for additional information.

    Returns
    -------
    fig, ax
    """
    fig = plt.figure(figsize=figsize)
    crs = get_crs(epsg)
    number_of_axes = len(data)
    if not _utils.is_iterable(title):
        title = [title] * number_of_axes
    parameter_name = np.array(["x", "y", "title", "geometries"])
    parameter_length = np.array([len(x), len(y), len(data), len(geometries)])
    check = parameter_length != number_of_axes

    number_of_rows = number_of_axes // 2 + 1
    number_of_columns = min(number_of_axes, 2)
    if sum(check) > 0:
        raise Exception(
            f"Parameters {parameter_name[check]} do not have the same number of first axis as the data. The first axis must have the same length as the number of plotted figures."
        )
    contours, contour_lines = [], []

    for i, d in enumerate(data):
        try:
            ax = fig.add_subplot(
                number_of_rows,
                number_of_columns,
                i + 1,
                projection=ccrs.epsg(epsg),
            )
        except xml.etree.ElementTree.ParseError:
            ax = fig.add_subplot(number_of_rows, number_of_columns, i + 1)

        if crs is None:
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
        else:
            ax.set_extent(extent, crs=crs)

        add_background(ax=ax, zoom_level=zoom_level)
        ax.axis("off")
        add_geometries(
            ax,
            geometries=geometries[i],
            scatter_kwargs=scatter_kwargs,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
        )
        _contours, _contour_lines = add_contours(
            ax,
            x[i],
            y[i],
            data[i],
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
        )
        contours.append(_contours)
        contour_lines.append(_contour_lines)

        add_north_arrow()
        add_scalebar(ax, extent)
        add_title(ax, title[i])
    fig.tight_layout(w_pad=6)

    for i, ax in enumerate(fig.axes):
        if contours[i] is not None:
            add_colorbar(
                ax,
                contours[i],
                fig,
                contour_lines=contour_lines[i],
                colorbar_kwargs=colorbar_kwargs,
            )

    fig.set_size_inches(figsize)
    return fig, fig.axes


def resolve_title(
    Model, title, variable, unit, step=-1, model=None, Suite=None
):
    if title is None:
        unit_label = get_unit_label(variable, unit)
        if Model.timesteps.dtype == np.int64:
            title = f"{variable.capitalize().replace('_', ' ')} ({unit_label}) - year {Model.timesteps[step]} - {Model.name}"
        elif np.issubdtype(Model.timesteps.dtype, np.datetime64):
            title = f"{variable.capitalize().replace('_', ' ')} ({unit_label}) - {np.datetime_as_string(Model.timesteps, unit = 'D')[step]} - {Model.name}"
    elif type(title) == str:
        title = title
    elif _utils.is_iterable(title):
        if len(title) == 1:
            title = title[0]
        elif Suite and model is not None:
            if len(title) == Suite.number_of_models:
                title = title[model]
        else:
            raise Exception("Invalid number of entries in title list.")
    else:
        raise Exception(
            f"Invalid input type ({type(step)}) for title. Assign a string or a list of strings."
        )
    return title


def extent_from_bound(bound):
    return (bound[0], bound[2], bound[1], bound[3])


def extent_from_model(Model, buffer):
    ext = (
        Model.bounds[0] - buffer,
        Model.bounds[2] + buffer,
        Model.bounds[1] - buffer,
        Model.bounds[3] + buffer,
    )
    return ext


def point_entry_to_index(Model, points):
    type_points = type(points)
    if points is None:
        point_index = list(range(Model.number_of_points))
    elif _utils.is_iterable(points):
        point_index = [Model.point_label_to_int(r) for r in points]
    elif type_points == int or type_points == str:
        point_index = [Model.point_label_to_int(points)]
    return point_index


def time_entry_to_index(Model, time, _2d=False):
    if time is None and _2d is False:
        steps = list(range(Model.number_of_steps))
    elif time is None and _2d:
        steps = [-1]
    elif _utils.is_iterable(time):
        if not _2d:
            steps = [Model.time_label_to_int(t) for t in time]
        else:
            steps = [Model.time_label_to_int(time[-1])]
    else:
        try:
            steps = [Model.time_label_to_int(time)]
        except:
            raise Exception(f"Invalid type for indexing time: {type(time)}")
    return steps


def reservoir_entry_to_index_model(Model, reservoir):
    if reservoir is None:
        reservoir = list(range(Model.number_of_reservoirs))
    elif _utils.is_iterable(reservoir):
        reservoir_index = [Model.reservoir_label_to_int(r) for r in reservoir]
        reservoir = [r for r in reservoir_index if r is not None]
    elif isinstance(reservoir, int):
        reservoir = [Model.reservoir_label_to_int(reservoir)]
    elif isinstance(reservoir, str):
        if reservoir in Model.reservoirs:
            reservoir = [Model.reservoir_label_to_int(reservoir)]
        else:
            reservoir = []
    else:
        raise Exception(f"Invalid input type for reservoir: {type(reservoir)}")
    return reservoir


def reservoir_entry_to_index_suite(Suite, reservoir):
    unique_reservoirs = Suite.unique_reservoirs()
    if reservoir is None:
        reservoir = list(range(len(unique_reservoirs)))
    elif _utils.is_iterable(reservoir):
        reservoir_index = [Suite.reservoir_label_to_int(r) for r in reservoir]
        reservoir = [r for r in reservoir_index if r is not None]
    elif isinstance(reservoir, int):
        reservoir = [Suite.reservoir_label_to_int(reservoir)]
    elif isinstance(reservoir, str):
        if reservoir in unique_reservoirs:
            reservoir = [Suite.reservoir_label_to_int(reservoir)]
        else:
            reservoir = []
    else:
        raise Exception(f"Invalid input type for reservoir: {type(reservoir)}")
    return reservoir


def reservoir_entry_to_index(Model, reservoir):
    if _utils.isSubsidenceModel(Model):
        return reservoir_entry_to_index_model(Model, reservoir)
    elif _utils.isSubsidenceSuite(Model):
        return reservoir_entry_to_index_suite(Model, reservoir)


def seperate_colors_from_dict(plot_kwargs, number_of_entries):
    if "c" in plot_kwargs.keys():
        c = plot_kwargs["c"]
        adjusted_kwargs = plot_kwargs.copy()
        del adjusted_kwargs["c"]
    elif "color" in plot_kwargs.keys():
        c = plot_kwargs["color"]
        adjusted_kwargs = plot_kwargs.copy()
        del adjusted_kwargs["color"]
    elif "facecolor" in plot_kwargs.keys():
        c = plot_kwargs["facecolor"]
        adjusted_kwargs = plot_kwargs.copy()
        del adjusted_kwargs["facecolor"]
    elif "fc" in plot_kwargs.keys():
        c = plot_kwargs["fc"]
        adjusted_kwargs = plot_kwargs.copy()
        del adjusted_kwargs["fc"]
    elif "cmap" in plot_kwargs.keys():
        cmap = plot_kwargs["cmap"]
        adjusted_kwargs = plot_kwargs.copy()
        del adjusted_kwargs["cmap"]
        c = colors_from_cmap(cmap, number_of_entries)
    else:
        c = colors_from_cmap(cm.brg, number_of_entries)
        adjusted_kwargs = plot_kwargs.copy()

    if _utils.is_iterable(c):
        if len(c) != number_of_entries:
            raise Exception(
                f"The number of arguments for color or c ({len(c)}) do not equal the amount of entries: {number_of_entries}."
            )
    else:
        c = [c for _ in range(number_of_entries)]

    return c, adjusted_kwargs


def get_2D_data_from_model(
    Model, reservoir=None, time=-1, variable="subsidence", unit="cm"
):
    array = Model[variable]

    data_coords = list(array.coords)
    if not ("x" in data_coords and "y" in data_coords):
        raise Exception(
            'Model variable needs to have the "x" and "y" dimensions.'
        )

    reservoir_index = reservoir_entry_to_index(Model, reservoir)

    selection_dict = {}
    for coord in [c for c in data_coords if c not in ["x", "y"]]:
        if coord == "reservoir":
            selection_dict[coord] = reservoir_index
        if coord == "time":
            step = time_entry_to_index(Model, time, _2d=True)

            selection_dict[coord] = step[0]

    data3D = array.isel(selection_dict, drop=True).transpose("y", "x", ...)

    if "reservoir" in data_coords:
        data = data3D.sum(dim="reservoir")
    else:
        data = data3D
    data = _utils.convert_SI(data, "m", unit)
    data = np.array(data)

    return data


def get_transient_data_from_model(
    Model, reservoir=None, time=-1, variable="subsidence", unit="cm"
):
    steps = time_entry_to_index(Model, time)
    reservoir_index = reservoir_entry_to_index(Model, reservoir)
    array = Model[variable]

    data_coords = list(array.coords)
    if not ("x" in data_coords and "y" in data_coords):
        raise Exception(
            'Model variable needs to have the "x" and "y" dimensions.'
        )

    selection_dict = {}
    for coord in [c for c in data_coords if c not in ["x", "y"]]:
        if coord == "reservoir":
            selection_dict[coord] = reservoir_index

    data = []
    for t in steps:
        if "time" in data_coords:
            selection_dict["time"] = t
        data3D = array.isel(selection_dict)
        if "reservoir" in data_coords:
            data3D = data3D.sum(dim="reservoir")
        data3D = _utils.convert_SI(data3D, "m", unit)
        data3D = np.reshape(np.array(data3D), (Model.ny, Model.nx))
        data.append(data3D)

    return data


def plot_subsidence_model(
    Model,
    reservoir=None,
    time=-1,
    buffer=0,
    variable="subsidence",
    unit="cm",
    plot_reservoir_shapes=True,
    additional_shapes=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    contour_levels=None,
    contour_steps=0.01,
    title=None,
    final=True,
    fname="subsidence",
    svg=False,
    shape_kwargs={},
    raster_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    scatter_kwargs={},
):
    shape_kwargs = set_defaults(shape_kwargs, defaults=Model.shape_defaults)
    contourf_kwargs = set_defaults(
        contourf_kwargs, defaults=Model.contourf_defaults
    )
    contour_kwargs = set_defaults(
        contour_kwargs, defaults=Model.contour_defaults
    )
    clabel_kwargs = set_defaults(clabel_kwargs, defaults=Model.clabel_defaults)
    colorbar_kwargs = set_defaults(
        colorbar_kwargs, defaults=Model.colorbar_defaults
    )
    raster_kwargs = set_defaults(raster_kwargs, defaults=Model.raster_defaults)
    scatter_kwargs = set_defaults(
        scatter_kwargs, defaults=Model.scatter_defaults
    )

    if contour_levels is None:
        contour_steps = _utils.convert_SI(np.array(contour_steps), "m", unit)
    else:
        contour_levels = _utils.convert_SI(np.array(contour_levels), "m", unit)
        contour_steps = _utils.convert_SI(np.array(contour_steps), "m", unit)

    if _utils.isSubsidenceModel(Model):
        if not Model.hasattr(variable):
            calculator = (
                f"calculate_{variable}"
                if not variable.endswith(("_x", "_y"))
                else f"calculate_{variable[:-2]}"
            )
            if not hasattr(Model, calculator):
                calculator = (
                    f"set_{variable}"
                    if not variable.endswith(("_x", "_y"))
                    else f"set_{variable[:-2]}"
                )
                if not hasattr(Model, calculator):
                    raise Exception(f"No {variable} in model.")
            raise Exception(
                f"No {variable} data available, run {calculator} before attempting to plot"
            )

        reservoir_index = reservoir_entry_to_index(Model, reservoir)
        step = time_entry_to_index(Model, time, _2d=True)
        data = get_2D_data_from_model(
            Model, reservoir=reservoir, time=time, variable=variable, unit=unit
        )

        Model.extent = extent_from_model(Model, buffer)

        title = resolve_title(Model, title, variable, unit, step=step[0])

        if plot_reservoir_shapes and Model.hasattr("shapes"):
            geometries = [
                Model.shapes[r] for r in reservoir_index
            ] + additional_shapes
        else:
            geometries = additional_shapes

        fig, ax = plot_2D_data(
            Model.extent,
            Model.x,
            Model.y,
            data,
            geometries=geometries,
            title=title,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
            colorbar_kwargs=colorbar_kwargs,
            scatter_kwargs=scatter_kwargs,
        )

        if final:
            savefig(Model, fname, svg=svg)
            plt.show()
            plt.close()
        else:
            return fig, ax
    else:
        raise Exception(
            "Invalid input type for model: {type(Model)}. SubsidenceModel object is required."
        )


def plot_subsidence_suite(
    Suite,
    reservoir=None,
    time=-1,
    buffer=0,
    model=None,
    variable="subsidence",
    unit="cm",
    plot_reservoir_shapes=True,
    additional_shapes=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    contour_levels=None,
    contour_steps=0.01,
    title=None,
    final=True,
    fname="subsidence",
    svg=False,
    shape_kwargs={},
    raster_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    scatter_kwargs={},
):
    if _utils.isSubsidenceSuite(Suite):
        shape_kwargs = set_defaults(
            shape_kwargs, defaults=Suite.shape_defaults
        )
        contourf_kwargs = set_defaults(
            contourf_kwargs, defaults=Suite.contourf_defaults
        )
        contour_kwargs = set_defaults(
            contour_kwargs, defaults=Suite.contour_defaults
        )
        clabel_kwargs = set_defaults(
            clabel_kwargs, defaults=Suite.clabel_defaults
        )
        colorbar_kwargs = set_defaults(
            colorbar_kwargs, defaults=Suite.colorbar_defaults
        )
        raster_kwargs = set_defaults(
            raster_kwargs, defaults=Suite.raster_defaults
        )
        scatter_kwargs = set_defaults(
            scatter_kwargs, defaults=Suite.scatter_defaults
        )

        extent = extent_from_model(Suite, buffer)
        model = Suite.model_label_to_index(model)
        data, x, y, geometries, titles = [], [], [], [], []
        for i, Model in enumerate(Suite._models):
            if i in model:
                if not Model.hasattr(variable):
                    raise Exception(
                        f"The variable {variable} has not been set/calculated for model {Model.name}."
                    )
                reservoir_index = reservoir_entry_to_index(Model, reservoir)
                step = Model.time_label_to_int(time)

                model_data = get_2D_data_from_model(
                    Model,
                    reservoir=reservoir_index,
                    time=step,
                    variable=variable,
                    unit=unit,
                )
                x.append(Model.x)
                y.append(Model.y)

                if plot_reservoir_shapes and Model.hasattr("shapes"):
                    model_geometries = [
                        Model.shapes[r] for r in reservoir_index
                    ] + additional_shapes
                else:
                    model_geometries = additional_shapes
                geometries.append(model_geometries)
                data.append(model_data)

                titles.append(
                    resolve_title(
                        Model,
                        title,
                        variable,
                        unit,
                        Suite=Suite,
                        model=i,
                        step=step,
                    )
                )
        if contour_levels is None:
            contour_steps = _utils.convert_SI(
                np.array(contour_steps), "m", unit
            )
        else:
            contour_levels = _utils.convert_SI(
                np.array(contour_levels), "m", unit
            )
            contour_steps = _utils.convert_SI(
                np.array(contour_steps), "m", unit
            )
        fig, ax = plot_3D_data(
            extent,
            x,
            y,
            data,
            geometries=geometries,
            title=titles,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
            colorbar_kwargs=colorbar_kwargs,
        )

        if final:
            savefig(Suite, fname, svg=svg)
            plt.show()
            plt.close()
        else:
            return fig, ax
    else:
        raise Exception(
            "Invalid input type for model: {type(Suite)}. ModelSuite object is required."
        )


def plot_subsidence(
    Model,
    reservoir=None,
    time=-1,
    buffer=0,
    model=None,
    variable="subsidence",
    unit="cm",
    plot_reservoir_shapes=True,
    additional_shapes=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    contour_levels=None,
    contour_steps=0.01,
    title=None,
    final=True,
    fname="subsidence",
    svg=False,
    shape_kwargs={},
    raster_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    scatter_kwargs={},
):
    """Plot contours of the calculated subsidence.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a
        list, an Exception will occur. The default is -1, the final
        timestep.
    buffer : float/int, optional.
        Additional space to be added to the edge of the plotted
        figure in m. The default is 0.
    plot_reservoir_shapes : bool, optional
        When True, the shapes of the reservoirs will be plotted behind the
        contours, when False, not. The default is True.
    additional_shapes : list of PySub.Gemetries objects
        A list if Geometries to plot inside the figures that are not the reservoirs.
        Use PySub.Geometries.fetch() to import plottable geometries as a list.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    figsize : tuple, float, optional
        The size of the figure in inches.
    epsg : int, optional
        The available epsg of the WMTS service
    contour_levels : list, float, optional
        Draw contour lines at the specified levels. The values must be in increasing order.
        The default is None. When None, the contour lines will be chosen based
        on the subsidence data and the contour_steps parameter.
    contour_steps : float/int, optional
        The difference in values between the contour levels. The default is 0.01.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    title : list, str, optional
        Title of the figure. The default is None.
    final : bool, optional
        If True, the function ends with a call to plt.show() and the figure
        is plotted. If False, the function returns a fig and ax object.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    shape_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted shapes.
        The default is {}. See SubsidenceModel attribute shape_defaults
        for additional information.
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.
    scatter_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted points.
        The default is {}. See SubsidenceModel attribute point_defaults
        for additional information.

    Returns
    -------
    None.
    OR
    fig, ax
        Matplotlib figure or ax objects to be expanded upon.

    """

    shape_kwargs = set_defaults(shape_kwargs, defaults=Model.shape_defaults)
    contourf_kwargs = set_defaults(
        contourf_kwargs, defaults=Model.contourf_defaults
    )
    contour_kwargs = set_defaults(
        contour_kwargs, defaults=Model.contour_defaults
    )
    clabel_kwargs = set_defaults(clabel_kwargs, defaults=Model.clabel_defaults)
    colorbar_kwargs = set_defaults(
        colorbar_kwargs, defaults=Model.colorbar_defaults
    )

    if contour_levels is None:
        Model.get_contour_levels(variable, contour_steps=contour_steps)

    if _utils.isSubsidenceModel(Model):
        return plot_subsidence_model(
            Model,
            reservoir=reservoir,
            time=time,
            buffer=buffer,
            variable=variable,
            unit=unit,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            title=title,
            final=final,
            fname=fname,
            svg=svg,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
            colorbar_kwargs=colorbar_kwargs,
            scatter_kwargs=scatter_kwargs,
        )

    if _utils.isSubsidenceSuite(Model):
        return plot_subsidence_suite(
            Model,
            reservoir=reservoir,
            model=model,
            time=time,
            buffer=buffer,
            variable=variable,
            unit=unit,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            title=title,
            final=final,
            fname=fname,
            svg=svg,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
            colorbar_kwargs=colorbar_kwargs,
            scatter_kwargs=scatter_kwargs,
        )


def time_to_legend(time, Model):
    steps = time_entry_to_index(Model, time)
    legend_labels = np.datetime_as_string(Model.timesteps[steps], unit="D")
    return legend_labels


def add_errorbars(ax=None, observations=None, unit="cm", errorbar_kwargs={}):
    remember = ax
    if ax is None:
        fig, ax = plt.subplots()

    if (
        not str(type(observations))
        == "<class 'PySub.Points.ObservationCollection'>"
    ):
        raise Exception(
            f"Invalid observation type: {type(observations)}. No Points.ObservationCollection to plot"
        )
        if remember is None:
            return fig, ax
        else:
            return
    c, adjusted_errorbar_kwargs = seperate_colors_from_dict(
        errorbar_kwargs, len(np.unique(observations.names))
    )
    labels = observations.names
    for pi in range(observations.number_of_observation_points):
        values = -observations.relative[pi]
        values = _utils.convert_SI(values, "m", unit)
        upper_limits = observations.upper_errors[pi]
        upper_limits = _utils.convert_SI(upper_limits, "m", unit)
        lower_limits = observations.lower_errors[pi]
        lower_limits = _utils.convert_SI(lower_limits, "m", unit)
        add_errorbar(
            ax=ax,
            x=observations.time[pi],
            y=values,
            upper_limits=upper_limits,
            lower_limits=lower_limits,
            c=c[pi],
            label=labels[pi],
            errorbar_kwargs=adjusted_errorbar_kwargs,
        )
    if remember is None:
        return fig, ax


class AskForLine:
    def __init__(
        self,
        Model,
        zoom_level=10,
        plot_reservoir_shapes=True,
        additional_shapes=[],
        figsize=(8, 8),
        epsg=28992,
    ):
        self.rows = 2
        self.point_names = []
        self.coordinates = []
        self.root = Tk()
        self.root.title("Click for line")

        # Figure
        self.fig, self.ax = plot_reservoirs(
            Model,
            annotate=False,
            final=False,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
        )

        # Bind the clicking action to the matplotlib figure canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(column=0, row=0, columnspan=3)
        self.canvas.mpl_connect("button_press_event", self.get_coordinates)

        # Close button
        done = Button(self.root, text="Done", command=self.close)
        done.grid(column=0, row=1, columnspan=3)
        done.state = "disabled"

        # Column labels
        l_point_name = Label(self.root, text="Point", width=10)
        x_point_name = Label(self.root, text="X", width=25)
        y_point_name = Label(self.root, text="Y", width=25)
        l_point_name.grid(column=0, row=2)
        x_point_name.grid(column=1, row=2)
        y_point_name.grid(column=2, row=2)

        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes, "-topmost", False)
        self.root.mainloop()

    def get_line(self):
        return {n: c for n, c in zip(self.point_names, self.coordinates)}

    def close(self):
        if len(self.point_names) < 2:
            messagebox.showinfo(
                "Line message",
                "Not enough points selected to continue. Select at least 2.",
            )
        else:
            self.root.destroy()
            plt.close(self.fig)

    def add_entry_line(self, x, y):
        self.rows += 1
        e_point_name = Entry(self.root, width=10)
        e_point_name.grid(column=0, row=self.rows)
        e_point_x = Entry(self.root, width=25)
        e_point_x.grid(column=1, row=self.rows)
        e_point_y = Entry(self.root, width=25)
        e_point_y.grid(column=2, row=self.rows)

        point_name = string.ascii_uppercase[len(self.point_names) % 26]
        self.point_names.append(point_name)
        e_point_name.insert(0, point_name)
        e_point_x.insert(0, x)
        e_point_y.insert(0, y)
        self.coordinates.append((x, y))

    def get_coordinates(self, event):
        x = event.xdata
        y = event.ydata
        self.add_entry_line(x, y)
        self.ax.scatter(x, y, marker="o", c="r")
        if not len(self.point_names) == 1:
            x0, y0 = self.coordinates[-2]
            segment_x = [x, x0]
            segment_y = [y, y0]
            self.ax.plot(segment_x, segment_y, "r-")

        self.canvas.draw()


def ask_for_line(
    Model,
    zoom_level=10,
    plot_reservoir_shapes=True,
    additional_shapes=[],
    figsize=(8, 8),
    epsg=28992,
):
    if _utils.is_iterable(Model):  # If the entry is a list of Models
        list_of_models = Model
        Model = _ModelSuite("", None)
        Model.set_models(list_of_models)
    window = AskForLine(
        Model,
        zoom_level=zoom_level,
        figsize=figsize,
        epsg=epsg,
        plot_reservoir_shapes=plot_reservoir_shapes,
        additional_shapes=additional_shapes,
    )
    Model.line = window.get_line()
    return Model.line


def plot_cross_section(
    Model,
    lines=None,
    variable="subsidence",
    reservoir=None,
    time=-1,
    model=None,
    unit="cm",
    num=1000,
    figsize=(8, 8),
    title=None,
    y_axis_exageration_factor=2,
    ylim=None,
    final=True,
    fname="",
    svg=False,
    legend=True,
    horizontal_line=None,
    plot_kwargs={},
    colorbar_kwargs={},
    annotation_kwargs={},
):

    """Plot a map of the cross section in a 2D representation, and
    plot a line or set of lines of the subsidence along that cross section.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    lines : list/dict, float/int
        This variable is a list or a dictionary.
        Valid formats are:
            A single line: [[0, 1], [0,2]]
            Multiple lines: [[[1, 1], [1,2]], [[1, 1], [1,2]]]
            A dictionary for a single line: {"Point 1": [[0, 1], [0,2]]}
            A list of dictionaries for multiple lines:
                {"Point 1": [[1, 1], [1,2]], "Point 2": [[1, 1], [1,2]]}
        These lines represent the line
        the cross section will be drawn along.
    variable : str: optional
        Any Model variable that is present in the model and can be represented as
        a grid. Default is 'subsidence'. Other values can be "slope", "compaction",
        "pressure", etc.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a
        list, an Exception will occur. The default is -1, the final
        timestep.
    model : int, str or list of str or int
        Label - or list of labels - of the models that you want to plot the cross
        section of. The default is None, then all model will be plotted.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    num : int, optional
        The amount of points sampled along the crossection.build_grid
        The default is 1000.
    title : str, optional
        The title of the figure displaying the cross section data.
        The default is None.
    y_axis_exageration_factor : int/float, optional
        The factor the length of the y_axis will be exagerated. If the lowest data
        point in the graph is -1, the y-axis will be from -y_axis_exageration_factor
        to the highest point. The default is 2.
    ylim : tuple, float, optional
        A tuple of values determinging the extend of the y-axis.The default is None.
        When None, the y-axis will be determined using the data and
        y_axis_exageration_factor.
    final : bool
        When True, the figure will be shown en return immutable. If False,
        the matplotlib figure and ax(es) objects wqill be returned.
    fname : str
        The location the figure will be saved it. The default is '' this, will
        indicate the figure will not be stored. When a path is given, this figure
        will be stored at that location, when just a name is given, the figure
        will be stored in the project folder.
    svg : bool
        Save this file also as an svg-file.
    legend : bool
        When True, shows a legend, when False, the figure will plot no legend.
        The default is True.
    horizontal_line : float/dict
        When a float, it must be the value on the y-axis at which the horizontal
        line will be placed. When a dict, the key of the dictionary will be the label
        of the line and the entry will be the value on the y-axis the horizontal
        line will be plotted along.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines.
        The default is {}. See SubsidenceModel attribute plot_defaults
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.

    Returns
    -------
    fig, ax:
        The matplotlib figure and axs objects with the cross sections in it.

    """

    if not Model.hasattr(variable):
        warn(
            f"Warning, no {variable} has been calculated. No cross section has been made."
        )
        return

    plot_kwargs = set_defaults(plot_kwargs, defaults=Model.plot_defaults)
    annotation_kwargs = set_defaults(
        annotation_kwargs, defaults=Model.annotation_defaults
    )
    colorbar_kwargs = set_defaults(
        colorbar_kwargs, defaults=Model.colorbar_defaults
    )

    if not _utils.is_iterable(lines):
        warn(
            f"Warning: Invalid line type used: {type(lines)}. Use an iterable containing sets of x- and y-coordinates. No cross section plotted."
        )
        return
    else:
        if not isinstance(lines, dict):
            test = 0
        else:
            test = list(lines.keys())[0]

        if not _utils.is_iterable(lines[test]):
            warn(
                f"Warning: Invalid line type used: {type(lines)}. Use an iterable containing sets of x- and y-coordinates. No cross section plotted."
            )
            return
        else:
            if len(lines[test]) != 2:
                warn(
                    f"Warning: Invalid line type used: {type(lines)}. Use an iterable containing sets of x- and y-coordinates. No cross section plotted."
                )
                return

    if not isinstance(lines, dict):
        line_dict = {
            string.ascii_uppercase[i % 26]: line
            for i, line in enumerate(lines)
        }

    else:
        line_dict = lines

    if _utils.isSubsidenceModel(Model):
        steps = time_entry_to_index(Model, time)
        # reservoir_index = reservoir_entry_to_index(Model, reservoir)
        data = get_transient_data_from_model(
            Model, reservoir=reservoir, time=time, variable=variable, unit=unit
        )

        legend_labels = time_to_legend(time, Model)

        fig, ax = plt.subplots()
        c, adjusted_plot_kwargs = seperate_colors_from_dict(
            plot_kwargs, len(steps)
        )
        add_cross_section(
            ax,
            Model.x,
            Model.y,
            data,
            line_dict.values(),
            name=legend_labels,
            inflection_point_names=None,
            c=c,
            num=num,
            plot_kwargs=adjusted_plot_kwargs,
            annotation_kwargs=annotation_kwargs,
        )

        fig.set_size_inches(figsize)
        unit_label = get_unit_label(variable, unit)
        if title is None:
            title = f"Cross section {variable} ({unit_label})"

        add_title(ax, title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(
            f"{variable.capitalize().replace('_', ' ')} ({unit_label})"
        )
        set_ylim(
            ax,
            ylim=ylim,
            y_axis_exageration_factor=y_axis_exageration_factor,
            unit_in="m",
            unit_out=unit,
        )

        add_horizontal_line(ax, horizontal_line, unit=unit)

        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax.grid()

        if final:
            if fname:
                fname_cross_section = f"{fname}_cross_section_{variable}"
                savefig(Model, fname_cross_section, svg=svg)
            plt.show()
            plt.close()
        else:
            return fig, ax
    if _utils.isSubsidenceSuite(Model):
        model = Model.model_label_to_index(model)
        fig, ax = plt.subplots()

        counter = 0
        map_title = []
        legend_labels = []
        for i, m in enumerate(Model._models):
            if i in model:

                steps = time_entry_to_index(m, time)
                data_coords = list(m.grid[variable].coords)
                if "time" not in data_coords:
                    steps = [steps[-1]]
                _legend_labels = time_to_legend(steps, m)
                _legend_labels = [
                    label + " " + m.name for label in _legend_labels
                ]
                legend_labels = legend_labels + _legend_labels
                map_title.append(f"Cross section - {_legend_labels[-1]}")
                # reservoir_index = reservoir_entry_to_index(m, reservoir)

                c, adjusted_plot_kwargs = seperate_colors_from_dict(
                    plot_kwargs, len(steps)
                )
                data = get_transient_data_from_model(
                    m,
                    reservoir=reservoir,
                    time=steps,
                    variable=variable,
                    unit=unit,
                )

                adjusted_plot_kwargs = set_defaults(
                    {"linestyle": _line_style(counter % 10)},
                    adjusted_plot_kwargs,
                )

                add_cross_section(
                    ax,
                    m.x,
                    m.y,
                    data,
                    line_dict.values(),
                    name=_legend_labels,
                    inflection_point_names=None,
                    c=c,
                    num=num,
                    plot_kwargs=adjusted_plot_kwargs,
                    annotation_kwargs=annotation_kwargs,
                )
                counter += 1

        fig.set_size_inches(figsize)

        unit_label = get_unit_label(variable, unit)
        if title is None:
            title = f"Cross section {variable} ({unit_label})"
        add_title(ax, title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(
            f"{variable.capitalize().replace('_', ' ')} ({unit_label})"
        )
        set_ylim(
            ax,
            ylim=ylim,
            y_axis_exageration_factor=y_axis_exageration_factor,
            unit_in="m",
            unit_out=unit,
        )
        add_horizontal_line(ax, horizontal_line, unit=unit)
        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid()

        if final:
            if fname:
                fname_cross_section = f"{fname}_cross_section_{variable}"
                savefig(Model, fname_cross_section, svg=svg)

            plt.show()
            plt.close()
        else:
            return fig, ax


def plot_reservoirs(
    Model,
    reservoir=None,
    model=None,
    buffer=0,
    annotate=True,
    plot_reservoir_shapes=True,
    additional_shapes=[],
    additional_labels=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    final=True,
    fname="reservoirs",
    svg=False,
    # kwargs
    shape_kwargs={},
    annotation_kwargs={},
    raster_kwargs={},
    scatter_kwargs={},
):
    """Plot the polygons in a SubsidenceModel object.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    model : int, str or list of int or str, optional
        The index or name of the model you want the data to plotted from. If it is a
        list, multiple models will be displayed. The default is None.
        When None, all models will be displayed.
    buffer : float/int, optional.
        Additional space to be added to the edge of the plotted
        figure in m. The default is 0.
    annotate : bool, optional
        When True, the names of the reservoirs will be added.
        The default is True. Determine the annotation presentation using
        the annotation_kwargs keyword argument.
    additional_shapes : list of PySub.Gemetries objects
        A list if Geometries to plot inside the figures that are not the reservoirs.
        Use PySub.Geometries.fetch() to import plottable geometries as a list.
    additional_labels : list of str
        A list of names of the additional shapes to be used as annotation in
        the figure (if annotate is True)
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    figsize : tuple, float, optional
        The size of the figure in inches.
    epsg : int, optional
        The available epsg of the WMTS service
    final : bool, optional
        If True, the function ends with a call to plt.show() and the figure
        is plotted. If False, the function returns a fig and ax object.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    shape_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted shapes.
        The default is {}. See SubsidenceModel attribute shape_defaults
        for additional information.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.

    Returns
    -------
    None.
    OR
    fig, ax
        Matplotlib figure or ax objects to be expanded upon.

    """

    shape_kwargs = set_defaults(shape_kwargs, defaults=Model.shape_defaults)
    annotation_kwargs = set_defaults(
        annotation_kwargs, defaults=Model.annotation_defaults
    )
    scatter_kwargs = set_defaults(
        scatter_kwargs, defaults=Model.scatter_defaults
    )
    raster_kwargs = set_defaults(raster_kwargs, defaults=Model.raster_defaults)

    if _utils.isSubsidenceModel(Model):
        return plot_reservoirs_model(
            Model,
            reservoir=reservoir,
            buffer=buffer,
            annotate=annotate,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            additional_labels=additional_labels,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            final=final,
            fname=fname,
            svg=svg,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
            scatter_kwargs=scatter_kwargs,
            annotation_kwargs=annotation_kwargs,
        )
    elif _utils.isSubsidenceSuite(Model):
        return plot_reservoirs_suite(
            Model,
            reservoir=reservoir,
            model=model,
            buffer=buffer,
            annotate=annotate,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            additional_labels=additional_labels,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            final=final,
            fname=fname,
            svg=svg,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
            scatter_kwargs=scatter_kwargs,
            annotation_kwargs=annotation_kwargs,
        )


def plot_reservoirs_suite(
    Suite,
    reservoir=None,
    model=None,
    buffer=0,
    annotate=True,
    plot_reservoir_shapes=True,
    additional_shapes=[],
    additional_labels=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    final=True,
    fname="reservoirs",
    svg=False,
    # kwargs
    shape_kwargs={},
    annotation_kwargs={},
    raster_kwargs={},
    scatter_kwargs={},
):
    extent = extent_from_model(Suite, buffer)
    fig, ax = get_background(
        extent, basemap=True, zoom_level=zoom_level, figsize=figsize, epsg=epsg
    )
    model_index = Suite.model_label_to_index(model)
    for i, m in enumerate(Suite._models):
        if i in model_index:
            if m.hasattr("shapes"):
                reservoir_index = reservoir_entry_to_index(m, reservoir)
                if plot_reservoir_shapes:
                    geometries = [m.shapes[r] for r in reservoir_index]
                else:
                    geometries = []
                geometries = geometries + additional_shapes
                add_geometries(
                    ax,
                    geometries=geometries,
                    scatter_kwargs=scatter_kwargs,
                    shape_kwargs=shape_kwargs,
                    raster_kwargs=raster_kwargs,
                )

    if annotate:
        for m in model_index:
            reservoir_index = reservoir_entry_to_index(
                Suite._models[m], reservoir
            )
            shapes = [
                Suite[m].shapes[i] for i in reservoir_index
            ] + additional_shapes
            _points = [s.midpoint for s in shapes]
            _labels = [Suite[m].reservoirs[i] for i in reservoir_index] + list(
                additional_labels
            )
            add_annotations(
                ax, _labels, _points, annotation_kwargs=annotation_kwargs
            )

    if final:
        savefig(Suite, fname, svg=svg)
        plt.show()
        plt.close()
    else:
        return fig, ax


def savefig(Model_or_Suite, fname, svg=False):
    if fname:
        if not hasattr(Model_or_Suite, "project_folder"):
            warn(
                f"Warning: no project folder has been defined for this Model/Suite {Model_or_Suite.name}. Figure {fname} has not been saved"
            )
            return
        fname, fext = os.path.splitext(fname)
        Model_or_Suite.project_folder.savefig(fname)
        if svg:
            Model_or_Suite.project_folder.savefig(fname + ".svg")


def plot_reservoirs_model(
    Model,
    reservoir=None,
    buffer=0,
    annotate=True,
    plot_reservoir_shapes=True,
    additional_shapes=[],
    additional_labels=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    final=True,
    fname="reservoirs",
    svg=False,
    shape_kwargs={},
    annotation_kwargs={},
    raster_kwargs={},
    scatter_kwargs={},
):
    shape_kwargs = set_defaults(shape_kwargs, defaults=Model.shape_defaults)
    annotation_kwargs = set_defaults(
        annotation_kwargs, defaults=Model.annotation_defaults
    )

    data = None
    Model.extent = extent_from_model(Model, buffer)

    if plot_reservoir_shapes:
        if Model.hasattr("buckets"):
            reservoir_index = reservoir_entry_to_index(Model, reservoir)
            geometries = [
                Model.buckets[r]["shapes"]["Values"]
                for i, r in enumerate(Model.reservoirs)
                if i in reservoir_index
            ]
            geometries = _utils.flatten_ragged_lists2D(geometries)
        elif Model.hasattr("shapes"):
            reservoir_index = reservoir_entry_to_index(Model, reservoir)
            geometries = [Model.shapes[r] for r in reservoir_index]

        else:
            reservoir, geometries = [], []
    else:
        reservoir_index, geometries = [], []
    geometries = geometries + additional_shapes

    fig, ax = plot_2D_data(
        Model.extent,
        Model.x,
        Model.y,
        data,
        geometries=geometries,
        zoom_level=zoom_level,
        figsize=figsize,
        epsg=epsg,
        shape_kwargs=shape_kwargs,
        raster_kwargs=raster_kwargs,
        scatter_kwargs=scatter_kwargs,
    )

    if annotate:

        labels = [
            r for i, r in enumerate(Model.reservoirs) if i in reservoir_index
        ] + list(additional_labels)
        points = [shape.midpoint for i, shape in enumerate(geometries)]
        add_annotations(
            ax, labels, points, annotation_kwargs=annotation_kwargs
        )

    if final:
        savefig(Model, fname, svg=svg)
        plt.show()
        plt.close()
    else:
        return fig, ax


def plot_points_on_map(
    Model,
    points=None,
    labels=None,
    reservoir=None,
    time=-1,
    buffer=0,
    unit="cm",
    show_data=True,
    plot_reservoir_shapes=True,
    additional_shapes=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    contour_levels=None,
    contour_steps=0.01,
    title=None,
    annotate_reservoirs=True,
    annotate_points=True,
    final=True,
    fname="point_locations",
    svg=True,
    shape_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    annotation_kwargs={},
    scatter_kwargs={},
):

    """Plot the selected points on map.

    Model : SubsidenceModel or ModelSuite objects
    points : Point, PointCollection, ObservationPoint, ObservationCollection or 2D numpy array (floats)
        This variable can be two generalized into two types of objects:
            - An object that has the 'x', 'y' and 'names' or 'name' attributes.
            - A 2D numpy array with the shape (m, 2), where m is the number of
              points, and each point must have an x- and y-coordinate.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a
        list, an Exception will occur. The default is -1, the final
        timestep.
    buffer : float/int, optional.
        Additional space to be added to the edge of the plotted
        figure in m. The default is 0.
    show_data : bool, optional.
        When False, it will not show any contour plots. If True and there is no
        data to plot, no data will be plotted. When True and there is data
        to plot, it will show the data as contour plots.
    plot_reservoir_shapes : bool, optional
        When True, the shapes of the reservoirs will be plotted behind the
        contours, when False, not. The default is True.
    additional_shapes : list of PySub.Gemetries objects
        A list if Geometries to plot inside the figures that are not the reservoirs.
        Use PySub.Geometries.fetch() to import plottable geometries as a list.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    contour_levels : list, float/int, optional
        The data vlaues to show the contours of. The default is None.
        When None, the contour levels will be based on the data and the
        contour_steps parameter.
    contour_steps : float/int, optional
        The step size of the data values to show the contours of.
        The default is 0.01. Only gets used when contour_levels is None.
        Else it will be ignored.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    title : str, optional
        Title to the figure. The default is 'Subsidence (m) in {timestep}'.
    annotate_reservoirs : bool
        If there is no subsidence data to plot with the points, or show_data = False,
        and plot_reservoir_shapes == True, the Reservoirs will be automatically
        annotated with the reservoir names. Set this variable to False to
        remove these notations.
    final : bool, optional
        If True, the function ends with a call to plt.show() and the figure
        is plotted. If False, the function returns a fig and ax object.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    svg : bool
        Save this file also as an svg-file.
    shape_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted shapes.
        The default is {}. See SubsidenceModel attribute shape_defaults
        for additional information.
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.
    scatter_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted points.
        The default is {}. See SubsidenceModel attribute point_defaults
        for additional information.

    Returns
    -------
    None.
    OR
    fig, ax
        Matplotlib figure or ax objects to be expanded upon.

    """

    contourf_kwargs = set_defaults(
        contourf_kwargs, defaults=Model.contourf_defaults
    )
    contour_kwargs = set_defaults(
        contour_kwargs, defaults=Model.contour_defaults
    )
    clabel_kwargs = set_defaults(clabel_kwargs, defaults=Model.clabel_defaults)
    colorbar_kwargs = set_defaults(
        colorbar_kwargs, defaults=Model.colorbar_defaults
    )
    annotation_kwargs = set_defaults(
        annotation_kwargs, defaults=Model.annotation_defaults
    )
    shape_kwargs = set_defaults(shape_kwargs, defaults=Model.shape_defaults)
    scatter_kwargs = set_defaults(
        scatter_kwargs, defaults=Model.scatter_defaults
    )

    # Any conversion to the desired unit will happen in the plot_subsidence function.

    if show_data and hasattr(Model.grid, "subsidence"):
        fig, ax = plot_subsidence(
            Model,
            reservoir=reservoir,
            time=time,
            buffer=buffer,
            unit=unit,
            title=title,
            additional_shapes=additional_shapes,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            plot_reservoir_shapes=plot_reservoir_shapes,
            shape_kwargs=shape_kwargs,
            final=False,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
            colorbar_kwargs=colorbar_kwargs,
        )

    else:
        fig, ax = plot_reservoirs(
            Model,
            reservoir=reservoir,
            plot_reservoir_shapes=plot_reservoir_shapes,
            buffer=buffer,
            annotate=annotate_reservoirs,
            additional_shapes=additional_shapes,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            final=False,
            # kwargs
            shape_kwargs=shape_kwargs,
            annotation_kwargs=annotation_kwargs,
        )

    if points is None:
        if Model.hasattr("points"):
            points = Model.points
        elif Model.hasattr("observation_points"):
            points = Model.observation_points
        else:
            points = None

    if points is not None:
        if hasattr(points, "x"):
            x = points.x
            y = points.y
            if labels is not None:
                names = labels
            elif hasattr(points, "names"):
                names = points.names
            elif hasattr(points, "name"):
                names = [points.name]
            else:
                if _utils.is_iterable(x) and _utils.is_iterable(y):
                    names = np.array(
                        [(ix, iy) for (ix, iy) in zip(x, y)]
                    ).astype(str)
                else:
                    names = [str(x), str(y)]
        else:
            if not _utils.is_iterable(points):
                raise Exception(
                    f"Invalid format for points: {points}. Enter a Point(Collection), Observation(Collection or Point), or numpy array with 2 dimensions, (m, 2) where m is the number of points and the 2 are the x- and y-coordinates."
                )
            points = np.array(points)
            if len(points.shape) == 1:
                points = points[None, :]

            if len(points.shape) != 2 or points.shape[1] != 2:
                raise Exception(
                    f"Invalid format for points: {points}. \n Enter a Point(Collection), Observation(Collection or Point), or numpy array with 2 dimensions, (m, 2) where m is the number of points and the 2 are the x- and y-coordinates."
                )
            x = points[:, 0]
            y = points[:, 1]
            if labels is not None:
                if _utils.is_iterable(labels):
                    if not len(labels) == len(x):
                        raise Exception(
                            "Point labels have length {len(labels)}, which does not equal the amount of points: {len(x)}"
                        )
                    else:
                        names = labels
                else:
                    raise Exception(
                        "Invalid type of labels, enter an iterable."
                    )
            else:
                names = [f"{ix}, {iy}" for ix, iy in zip(x, y)]

    if annotate_points:
        annotations = names
    else:
        annotations = None
    point = list(zip(x, y))

    add_points(
        ax=ax,
        points=point,
        labels=annotations,
        scatter_kwargs=scatter_kwargs,
        annotation_kwargs=annotation_kwargs,
    )

    if final:
        savefig(Model, fname, svg=svg)
        plt.show()
    else:
        return fig, ax


def add_subsidence_points(
    ax, Model, points=None, reservoir=None, unit="cm", plot_kwargs={}
):

    plot_kwargs = set_defaults(plot_kwargs, defaults=Model.plot_defaults)

    reservoir = reservoir_entry_to_index(Model, reservoir)
    point_index = point_entry_to_index(Model, points)

    if "label" in plot_kwargs.keys():
        label = plot_kwargs["label"]
        if _utils.is_iterable(label):
            if len(label) != Model.number_of_points:
                raise Exception(
                    "Invalid number of entries for labels. A list of labels needs to be equal to the amount of points in a model."
                )
        elif type(label) == str:
            label = [label for _ in range(Model.number_of_points)]
        elif type(label) == int:
            label = [str(label) for _ in range(Model.number_of_points)]
        else:
            raise Exception("Invalid labels")
        del plot_kwargs["label"]
    else:
        label = np.array(Model.points.names)[point_index]
    label = [str(l) + " " + Model.name for l in label]

    c, adjusted_plot_kwargs = seperate_colors_from_dict(
        plot_kwargs, Model.number_of_points
    )
    if len(c) != Model.number_of_points:
        c = [None for _ in Model.number_of_points]
        warn(
            "Warning: invalid color keyword argument added. Use a list with the length of the points in each model."
        )

    for pi in point_index:
        values = (
            Model.point_subsidence.isel(reservoir=reservoir)
            .sum(dim="reservoir")
            .isel(points=pi)
            .values
        )
        values = _utils.convert_SI(values, "m", unit)
        plt.plot(
            Model.timesteps,
            values,
            c=c[pi],
            label=label[pi],
            **adjusted_plot_kwargs,
        )


def plot_subsidence_points(
    Model,
    points=None,
    reservoir=None,
    model=None,
    unit="cm",
    title=None,
    y_axis_exageration_factor=2,
    ylim=None,
    figsize=(8, 8),
    final=True,
    fname="subsidence_at_points",
    svg=False,
    legend=True,
    horizontal_line=None,
    plot_kwargs={},
):
    """Plot the subsidence at the location of points stored in the
    SubsidenceModel object.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    points : int, str or list of int or str, optional
        The index or name of the points you want to plot. If it is a list,
        all the points in that list will be displayed.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, all the reservoirs in that list will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    model : int, str or list of str or int
        Label - or list of labels - of the models that you want to plot the cross
        section of. The default is None, then all model will be plotted.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    title : str, optional
        Title to the figure. The default is None, when title is None there
        will be no title added to the figure.
    y_axis_exageration_factor : int/float, optional
        The factor the length of the y_axis will be exagerated. If the lowest data
        point in the graph is -1, the y-axis will be from -y_axis_exageration_factor
    ylim : tuple, float, optional
        A tuple of values determinging the extend of the y-axis.The default is None.
        When None, the y-axis will be determined using the data and
        y_axis_exageration_factor.
    figsize : tuple, int/float, optional
        The size of the figure in inches. The default is (8, 8).
    final : bool, optional
        If True, the function ends with a call to plt.show() and the figure
        is plotted. If False, the function returns a fig and ax object.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    svg : bool
        Save this file also as an svg-file.
    legend : bool
        When True, shows a legend, when False, the figure will plot no legend.
        The default is True.
    horizontal_line : float/dict
        When a float, it must be the value on the y-axis at which the horizontal
        line will be placed. When a dict, the key of the dictionary will be the label
        of the line and the entry will be the value on the y-axis the horizontal
        line will be plotted along.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines.
        The default is {}. See SubsidenceModel attribute plot_defaults
        for additional information.

    Returns
    -------
    None.
    OR
    fig, ax
        Matplotlib figure or ax objects to be expanded upon.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    if _utils.isSubsidenceModel(Model):
        if not (
            hasattr(Model.grid, "point_subsidence")
            and Model.grid.point_subsidence.shape[0] != 0
        ):
            warn(
                "Warning: No subsidence has been calculated for points in this model."
            )
            if final:
                return
            else:
                return fig, ax
        add_subsidence_points(
            ax,
            Model,
            points=points,
            reservoir=reservoir,
            unit=unit,
            plot_kwargs=plot_kwargs,
        )
    elif _utils.isSubsidenceSuite(Model):
        model = Model.model_label_to_index(model)
        counter = 0
        for i, m in enumerate(Model._models):
            if i in model:
                if not (
                    hasattr(m.grid, "point_subsidence")
                    and m.grid.point_subsidence.shape[0] != 0
                ):
                    warn(
                        f"Warning: No subsidence has been calculated for points in model {m.name}."
                    )
                adjusted_kwargs = set_defaults(
                    {"linestyle": _line_style(counter % 10)}, plot_kwargs
                )
                add_subsidence_points(
                    ax,
                    m,
                    points=points,
                    reservoir=reservoir,
                    unit=unit,
                    plot_kwargs=adjusted_kwargs,
                )
                counter += 1

    else:
        raise Exception(
            f"Invalid type: {type(Model)}. Use SubsidenceModel or ModelSuite object."
        )

    add_horizontal_line(ax, horizontal_line, unit=unit)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    add_title(ax, title)
    ax.set_xlabel("Year")
    unit_label = get_unit_label("subsidence", unit)
    ax.set_ylabel(f"Subsidence ({unit_label})")
    set_ylim(
        ax,
        ylim=ylim,
        y_axis_exageration_factor=y_axis_exageration_factor,
        unit_in="m",
        unit_out=unit,
    )
    ax.grid()

    if final:
        savefig(Model, fname, svg=svg)
        plt.show()
    else:
        return fig, ax


def add_subsidence_at_observation_points(
    ax, Model, observations=None, reservoir=None, unit="cm", plot_kwargs={}
):

    if _utils.isSubsidenceModel(Model):
        if (
            not Model.hasattr("observation_points")
            or Model.number_of_observation_points == 0
        ):
            warn("Warning: No observation points to plot.")
            return plt.subplot()

        if (
            Model.hasattr("observation_subsidence")
            and Model.number_of_observation_points != 0
        ):
            reservoir = reservoir_entry_to_index(Model, reservoir)

            type_observations = type(observations)
            if observations is None:
                observations_index = list(
                    range(Model.number_of_observation_points)
                )
            elif _utils.is_iterable(observations):
                observations_index = [
                    Model.observation_label_to_int(r) for r in observations
                ]
            elif type_observations == int or type_observations == str:
                observations_index = [
                    Model.observation_label_to_int(observations)
                ]

            c, adjusted_plot_kwargs = seperate_colors_from_dict(
                plot_kwargs, len(observations_index)
            )
            if "label" in plot_kwargs.keys():
                label = plot_kwargs["label"]
                if _utils.is_iterable(label):
                    pass
                elif type(label) == str:
                    label = [
                        label
                        for _ in range(Model.number_of_observation_points)
                    ]
                elif type(label) == int:
                    label = [
                        str(label)
                        for _ in range(Model.number_of_observation_points)
                    ]
                else:
                    raise Exception("Invalid labels")
                del plot_kwargs["label"]
            else:
                label = np.array(Model.observation_points.names)[
                    observations_index
                ]
            label = [str(l) + " " + Model.name for l in label]
            for pi, p in enumerate(observations_index):
                values = (
                    Model.observation_subsidence.isel(reservoir=reservoir)
                    .sum(dim="reservoir")
                    .isel(observations=p)
                    .values
                )
                values = _utils.convert_SI(values, "m", unit)
                ax.plot(
                    Model.timesteps,
                    values,
                    c=c[pi],
                    label=label[pi],
                    **adjusted_plot_kwargs,
                )
    else:
        raise Exception(f"{type(Model)} is invalid entry.")


def plot_subsidence_observations(
    Model,
    observations=None,
    reservoir=None,
    model=None,
    unit="cm",
    title=None,
    y_axis_exageration_factor=2,
    ylim=None,
    figsize=(8, 8),
    final=True,
    fname="subsidence_at_observations",
    svg=False,
    legend=True,
    horizontal_line=None,
    plot_kwargs={},
    errorbar_kwargs={},
):
    """Plot the subsidence at the location of observation points stored in the
    SubsidenceModel object.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    observations : int, str or list of int or str, optional
        The index or name of the observations you want to plot. If it is a list,
        all the observations in that list will be displayed.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, all the reservoirs in that list will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    model : int, str or list of int or str, optional
        The index or name of the models you want the data to plotted from. If it is a
        list, all the models in that suite will be displayed. The default is None.
        When None, all models will be displayed.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    title : str, optional
        Title to the figure. The default is None, when title is None there
        will be no title added to the figure.
    y_axis_exageration_factor : int/float, optional
        The factor the length of the y_axis will be exagerated. If the lowest data
        point in the graph is -1, the y-axis will be from -y_axis_exageration_factor
        to the highest point. The default is 2.
    ylim : tuple, float, optional
        A tuple of values determinging the extend of the y-axis.The default is None.
        When None, the y-axis will be determined using the data and
        y_axis_exageration_factor.
    figsize : tuple, int/float, optional
        The size of the figure in inches. The default is (8, 8).
    final : bool, optional
        If True, the function ends with a call to plt.show() and the figure
        is plotted. If False, the function returns a fig and ax object.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    svg : bool
        Save this file also as an svg-file.
    legend : bool
        When True, shows a legend, when False, the figure will plot no legend.
        The default is True.
    horizontal_line : float/dict
        When a float, it must be the value on the y-axis at which the horizontal
        line will be placed. When a dict, the key of the dictionary will be the label
        of the line and the entry will be the value on the y-axis the horizontal
        line will be plotted along.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines.
        The default is {}. See SubsidenceModel attribute plot_defaults
        for additional information.

    Returns
    -------
    None.
    OR
    fig, ax
        Matplotlib figure or ax objects to be expanded upon.
    """
    plot_kwargs = set_defaults(plot_kwargs, defaults=Model.plot_defaults)
    errorbar_kwargs = set_defaults(
        errorbar_kwargs, defaults=Model.errorbar_defaults
    )

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    if _utils.isSubsidenceModel(Model):
        if observations is None:
            plot_observations = Model.observation_points
        else:
            if _utils.is_iterable(observations):
                plot_observations = _Points.ObservationCollection(
                    Model.observation_points[observations]
                )
            else:
                plot_observations = _Points.ObservationCollection(
                    [Model.observation_points[observations]]
                )
        add_subsidence_at_observation_points(
            ax,
            Model,
            observations=observations,
            reservoir=reservoir,
            unit=unit,
            plot_kwargs=plot_kwargs,
        )

        add_errorbars(
            ax=ax,
            observations=plot_observations,
            unit=unit,
            errorbar_kwargs=errorbar_kwargs,
        )

    elif _utils.isSubsidenceSuite(Model):
        model = Model.model_label_to_index(model)

        counter = 0
        c, pre_adjusted_kwargs = seperate_colors_from_dict(
            plot_kwargs,
            len(
                np.unique(
                    Model.unique_observations(observations=observations).names
                )
            ),
        )
        for i, m in enumerate(Model._models):
            if i in model:
                adjusted_plot_kwargs = set_defaults(
                    {"linestyle": _line_style(counter % 10)},
                    pre_adjusted_kwargs,
                )
                adjusted_plot_kwargs["c"] = c[
                    [
                        i
                        for i, e in enumerate(
                            np.unique(Model.unique_observations().names)
                        )
                        if e in m.observation_points.names
                    ]
                ]
                add_subsidence_at_observation_points(
                    ax,
                    m,
                    observations=observations,
                    reservoir=reservoir,
                    unit=unit,
                    plot_kwargs=adjusted_plot_kwargs,
                )
                counter += 1

        add_errorbars(
            ax=ax,
            observations=Model.unique_observations(observations=observations),
            unit=unit,
            errorbar_kwargs=errorbar_kwargs,
        )

    add_horizontal_line(ax, horizontal_line, unit=unit)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if title is None:
        title = "Subsidence at observations"

    add_title(ax, title)
    ax.set_xlabel("Year")
    unit_label = get_unit_label("subsidence", unit)
    ax.set_ylabel(f"Subsidence ({unit_label})")
    set_ylim(
        ax,
        ylim=ylim,
        y_axis_exageration_factor=y_axis_exageration_factor,
        unit_in="m",
        unit_out=unit,
    )
    ax.grid()

    if final:
        savefig(Model, fname, svg=svg)
        plt.show()
    else:
        return fig, ax


def plot_timeseries(
    Model,
    points=None,
    variable="subsidence",
    reservoir=None,
    model=None,
    unit="cm",
    title=None,
    y_axis_exageration_factor=2,
    ylim=None,
    figsize=(8, 8),
    final=True,
    fname="timeserie",
    svg=False,
    legend=True,
    horizontal_line=None,
    mode="coord",
    plot_kwargs={},
):
    plot_kwargs = set_defaults(plot_kwargs, defaults=Model.plot_defaults)
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)

    """Plot subsidence timeseries from SubsidenceModel objects.
    

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects.
    points : list, tuple np.ndarray
        An m x 2 numpy array where m is the amount of points and 2 is the
        x- and y-coordinate of each point. These points represent locations 
        at which subsidence will be/is determined.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a 
        list, all the reservoirs in that list will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    model : int, str or list of int or str, optional
        The index or name of the models you want the data to plotted from. If it is a 
        list, all the models in that suite will be displayed. The default is None.
        When None, all models will be displayed.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    title : str, optional
        The title you want above the figure. The default is None.
    y_axis_exageration_factor : int/float, optional
        The factor the length of the y_axis will be exagerated. If the lowest data 
        point in the graph is -1, the y-axis will be from -y_axis_exageration_factor
        to the highest point. The default is 2. 
    ylim : tuple, float, optional
        A tuple of values determinging the extend of the y-axis.The default is None.
        When None, the y-axis will be determined using the data and 
        y_axis_exageration_factor.figsize : tuple, int/float, optional
        The size of the figure in inches. The default is (8, 8).
    final : boolean, optional
        When True, the figure will plot, if false, a fig and ax(s) object will be 
        returned. The default is True.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    svg : bool
        Save this file also as an svg-file.
    legend : bool
        When True, shows a legend, when False, the figure will plot no legend.
        The default is True.
    horizontal_line : float/dict
        When a float, it must be the value on the y-axis at which the horizontal 
        line will be placed. When a dict, the key of the dictionary will be the label 
        of the line and the entry will be the value on the y-axis the horizontal 
        line will be plotted along.
    mode : str, optional
        When 'max', the points will be ignored and instead the subsidence at the 
        location with the most subsidence will be plotted. X and y will be ignored. 
        The default is 'coord', which means points will be used.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines. 
        The default is {}. See SubsidenceModel attribute plot_defaults
        for additional information.

    Returns
    -------
    None.
    OR
    fig, ax
        Matplotlib figure or ax objects to be expanded upon.

    """

    if mode.lower() != "max":
        if points is None:
            raise Exception(
                'The point values must be entered when not using mode = "max".'
            )

        points, x, y = _utils.point_or_points(points)
        number_of_points = len(points)

        if _utils.isSubsidenceModel(Model):
            if not Model.hasattr(variable):
                raise Exception(
                    f"No variable {variable} to plot timeseries from."
                )
            c, adjusted_plot_kwargs = seperate_colors_from_dict(
                plot_kwargs, number_of_points
            )
            reservoir_index = reservoir_entry_to_index(Model, reservoir)
            for j, p in enumerate(points):
                timeserie = np.array(
                    Model.get_timeseries(
                        variable=variable,
                        x=float(x[j]),
                        y=float(y[j]),
                        reservoir=reservoir_index,
                    )
                )
                timeserie = _utils.convert_SI(timeserie, "m", unit)
                timesteps = Model.timesteps
                label = str(x[j]) + ", " + str(y[j])
                ax.plot(
                    timesteps,
                    timeserie,
                    c=c[j],
                    label=label,
                    **adjusted_plot_kwargs,
                )
        if _utils.isSubsidenceSuite(Model):
            model = Model.model_label_to_index(model)
            c, adjusted_plot_kwargs = seperate_colors_from_dict(
                plot_kwargs, number_of_points
            )
            for j, p in enumerate(points):
                counter = 0
                for i, m in enumerate(Model._models):
                    if i in model:
                        if not m.hasattr(variable):
                            raise Exception(
                                f"No variable {variable} to plot timeseries from."
                            )
                        adjusted_kwargs = set_defaults(
                            {"linestyle": _line_style(counter % 10)},
                            adjusted_plot_kwargs,
                        )
                        reservoir_index = reservoir_entry_to_index(
                            m, reservoir
                        )
                        timeserie = np.array(
                            m.get_timeseries(
                                variable=variable,
                                reservoir=reservoir_index,
                                x=x[j],
                                y=y[j],
                            )
                        )
                        timeserie = _utils.convert_SI(timeserie, "m", unit)
                        timesteps = m.timesteps
                        label = str(x[j]) + ", " + str(y[j]) + " - " + m.name
                        ax.plot(
                            timesteps,
                            timeserie,
                            c=c[j],
                            label=label,
                            **adjusted_kwargs,
                        )
                        counter += 1
    else:
        if _utils.isSubsidenceModel(Model):
            if not Model.hasattr(variable):
                raise Exception(
                    f"No variable {variable} to plot timeseries from."
                )
            c, adjusted_plot_kwargs = seperate_colors_from_dict(plot_kwargs, 1)
            reservoir_index = reservoir_entry_to_index(Model, reservoir)
            _, (x, y) = Model.get_max_subsidence()
            data = Model.get_timeseries(
                variable=variable, reservoir=reservoir_index, x=x, y=y
            )

            timeserie = np.array(data)
            timeserie = _utils.convert_SI(timeserie, "m", unit)
            timesteps = Model.timesteps
            label = str(x) + ", " + str(y)
            ax.plot(
                timesteps,
                timeserie,
                c=c[0],
                label=label,
                **adjusted_plot_kwargs,
            )
        if _utils.isSubsidenceSuite(Model):

            model = Model.model_label_to_index(model)
            c, adjusted_plot_kwargs = seperate_colors_from_dict(
                plot_kwargs, Model.number_of_models
            )
            counter = 0
            _, _, (x, y) = Model.get_max_subsidence()
            for i, m in enumerate(Model._models):
                if i in model:
                    if not m.hasattr(variable):
                        raise Exception(
                            f"No variable {variable} to plot timeseries from."
                        )
                    adjusted_kwargs = set_defaults(
                        {"linestyle": _line_style(counter % 10)},
                        adjusted_plot_kwargs,
                    )
                    reservoir_index = reservoir_entry_to_index(m, reservoir)

                    data = m.get_timeseries(
                        variable=variable, reservoir=reservoir_index, x=x, y=y
                    )

                    timeserie = np.array(data)
                    timeserie = _utils.convert_SI(timeserie, "m", unit)
                    timesteps = m.timesteps
                    label = str(x) + ", " + str(y) + " - " + m.name
                    ax.plot(
                        timesteps,
                        timeserie,
                        c=c[i],
                        label=label,
                        **adjusted_kwargs,
                    )
                    counter += 1

    if title is None:
        if mode.lower() == "max":
            title = f'{variable.capitalize().replace("_", " ")} over time at locations with most subsidence (see legend).'
        else:
            title = f'{variable.capitalize().replace("_", " ")} over time at specified locations (see legend).'

    add_horizontal_line(ax, horizontal_line, unit=unit)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    add_title(ax, title)
    ax.set_xlabel("Year")
    unit_label = get_unit_label(variable, unit)
    ax.set_ylabel(f'{variable.capitalize().replace("_", " ")} ({unit_label})')
    set_ylim(
        ax,
        ylim=ylim,
        y_axis_exageration_factor=y_axis_exageration_factor,
        unit_in="m",
        unit_out=unit,
    )
    ax.grid()

    if final:
        savefig(Model, fname, svg=svg)
        plt.show()
    else:
        return fig, ax


def plot_min_mean_max(
    Suite,
    point=None,
    reservoir=None,
    model=None,
    unit="cm",
    title=None,
    y_axis_exageration_factor=2,
    ylim=None,
    figsize=(8, 8),
    final=True,
    fname="min_mean_max",
    svg=False,
    mode="coord",
    legend=True,
    horizontal_line=None,
    plot_kwargs={},
    fill_between_kwargs={},
):
    """Plot the minimum, mean and maximum values for the subsidence in a ModelSuite
    as a timeseries.

    Parameters
    ----------
    Suite : ModelSuite object.
    point : list, tuple np.ndarray
        An m x 2 numpy array where m is the amount of points and 2 is the
        x- and y-coordinate of each point. These points represent locations
        at which subsidence will be/is determined. If mode == 'max' this parameter will
        be ignored and the location with the most subsidence will be targeted.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, all the reservoirs in that list will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    model : int, str or list of int or str, optional
        The index or name of the models you want the data to plotted from. If it is a
        list, all the models in that suite will be displayed. The default is None.
        When None, all models will be displayed.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    title : str, optional
        The title you want above the figure. The default is None.
    y_axis_exageration_factor : int/float, optional
        The factor the length of the y_axis will be exagerated. If the lowest data
        point in the graph is -1, the y-axis will be from -y_axis_exageration_factor
        to the highest point. The default is 2.
    ylim : tuple, float, optional
        A tuple of values determinging the extend of the y-axis.The default is None.
        When None, the y-axis will be determined using the data and
        y_axis_exageration_factor.figsize : tuple, int/float, optional
        The size of the figure in inches. The default is (8, 8).
    final : boolean, optional
        When True, the figure will plot, if false, a fig and ax(s) object will be
        returned. The default is True.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    svg : bool
        Save this file also as an svg-file.
    mode : str, optional
        If 'max', the location with the most subsidence will be targeted. Else
        the value at point.
    legend : bool
        When True, shows a legend, when False, the figure will plot no legend.
        The default is True.
    horizontal_line : float/dict
        When a float, it must be the value on the y-axis at which the horizontal
        line will be placed. When a dict, the key of the dictionary will be the label
        of the line and the entry will be the value on the y-axis the horizontal
        line will be plotted along.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines.
        The default is {}. See SubsidenceModel attribute plot_defaults
        for additional information.
    fill_between_kwargs : dict, optional
        Dictionary with the keyword arguments for the filled area between lines.
        The default is {}. See SubsidenceModel attribute fill_area_defaults
        for additional information.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    OR
    None

    """
    plot_kwargs = set_defaults(plot_kwargs, defaults=Suite.plot_defaults)
    fill_between_kwargs = set_defaults(
        fill_between_kwargs, defaults=Suite.fill_between_defaults
    )

    if not _utils.isSubsidenceSuite(Suite):
        raise Exception(
            f"Invalid input type {type(Suite)} for first input argument. Must be ModelSuite."
        )
    if not Suite.hasattr("subsidence"):
        raise Exception("No subsidence to plot timeseries from.")
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    if mode.lower() != "max":
        if point is None:
            raise Exception(
                'The point values must be entered when not using mode = "max".'
            )
        elif not _utils.is_iterable(point):
            raise Exception(
                "The point values are required to be a tuple of an x- and y-coordinate."
            )
        elif len(point) != 2:
            raise Exception(
                "The point values are required to be a tuple of an x- and y-coordinate."
            )
        else:
            x, y = point
    else:
        _, _, xy = Suite.get_max_subsidence(reservoir=reservoir)
        x, y = xy

    c, adjusted_plot_kwargs = seperate_colors_from_dict(plot_kwargs, 1)

    _min, _mean, _max = Suite.get_subsidence_spread(
        x, y, reservoir=reservoir, model=model
    )
    _min = _utils.convert_SI(_min, "m", unit)
    _mean = _utils.convert_SI(_mean, "m", unit)
    _max = _utils.convert_SI(_max, "m", unit)
    add_filled_area(
        ax,
        _min.time.values,
        _min.values,
        _max.values,
        label="Range",
        kwargs=fill_between_kwargs,
    )
    ax.plot(
        _mean.time.values,
        _mean.values,
        c=c[0],
        label="Mean",
        **adjusted_plot_kwargs,
    )
    add_horizontal_line(ax, horizontal_line, unit=unit)
    if legend:
        ax.legend()

    ax.grid()
    add_title(ax, f"Range and mean of the model suite at: {x}, {y}")
    unit_label = get_unit_label("subsidence", unit)
    ax.set_ylabel(f"Subsidence {unit_label}")

    if final:
        savefig(Suite, fname, svg=svg)
        plt.show()
    else:
        return fig, ax


def plot_overlap(
    Model,
    time=-1,
    cutoff=0.01,
    buffer=0,
    reservoir=None,
    plot_reservoir_shapes=True,
    plot_subsidence_contours=False,
    additional_shapes=[],
    zoom_level=10,
    figsize=(8, 8),
    epsg=28992,
    contour_levels=None,
    contour_steps=0.01,
    unit="cm",
    title=None,
    legend=True,
    variable="subsidence",
    final=True,
    fname="",
    svg=False,
    shape_kwargs={},
    raster_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    overlap_kwargs={},
    annotation_kwargs={},
    scatter_kwargs={},
):
    """Plot contours of the overlap in calculated subsidence.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a
        list, an Exception will occur. The default is -1, the final
        timestep.
    cutoff : float, optional
        The value of matching overlap you are interested in. With a value of 1,
        All the values where 1 m of subsidence occurs and are plotted. Overlap
        between different reservoirs will be highlighted.
    buffer : float/int, optional.
        Additional space to be added to the edge of the plotted
        figure in m. The default is 0.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    plot_reservoir_shapes : bool, optional
        When True, the shapes of the reservoirs will be plotted behind the
        contours, when False, not. The default is True.
    plot_subsidence_contours : boolean, optional
        When True, the subsidence contours will also be plotted (as per plot_subsidence)
        behind the overlap raster. If False, it will not be plotted.
    additional_shapes : list of PySub.Gemetries objects
        A list if Geometries to plot inside the figures that are not the reservoirs.
        Use PySub.Geometries.fetch() to import plottable geometries as a list.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    figsize : tuple, float, optional
        The size of the figure in inches.
    epsg : int, optional
        The available epsg of the WMTS service

    contour_levels : list, float, optional
        Draw contour lines at the specified levels. The values must be in increasing order.
        The default is None. When None, the contour lines will be chosen based
        on the subsidence data and the contour_steps parameter.
    contour_steps : float/int, optional
        The difference in values between the contour levels. The default is 0.01.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    title : list, str, optional
        Title of the figure. The default is None.
    legend : boolean, optional
        If True, a legend wil appear, if False, not.
    final : bool, optional
        If True, the function ends with a call to plt.show() and the figure
        is plotted. If False, the function returns a fig and ax object.
    fname : str, optional
        When entered, the plotted figure will be saved under this name.
    shape_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted shapes.
        The default is {}. See SubsidenceModel attribute shape_defaults
        for additional information.
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    overlap_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted overlap raster.
        The default is {}. See SubsidenceModel attribute overlap_defaults
        for additional information.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.
    scatter_kwargs : dict, optional
        Dictionary with the keyword arguments for shapes that are point objects.
        The default is {}. See SubsidenceModel attribute scatter_defaults
        for additional information.

    Returns
    -------
    None.
    OR
    fig, ax
        Matplotlib figure or ax objects to be expanded upon.

    """
    shape_kwargs = set_defaults(shape_kwargs, defaults=Model.shape_defaults)
    contourf_kwargs = set_defaults(
        contourf_kwargs, defaults=Model.contourf_defaults
    )
    contour_kwargs = set_defaults(
        contour_kwargs, defaults=Model.contour_defaults
    )
    clabel_kwargs = set_defaults(clabel_kwargs, defaults=Model.clabel_defaults)
    colorbar_kwargs = set_defaults(
        colorbar_kwargs, defaults=Model.colorbar_defaults
    )
    overlap_kwargs = set_defaults(
        overlap_kwargs, defaults=Model.raster_defaults
    )
    annotation_kwargs = set_defaults(
        annotation_kwargs, defaults=Model.annotation_defaults
    )
    scatter_kwargs = set_defaults(
        scatter_kwargs, defaults=Model.scatter_defaults
    )

    if variable != "subsidence":
        warn(
            f"Warning: Variable {variable} has no meaningful measure for overlap."
        )
    time_index = time_entry_to_index(Model, time)
    if len(time_index) != 1:
        warn(
            "Warning: More than 1 timestep is requested, final timestep in sseries is displayed."
        )
    cutoff_units = _utils.convert_SI(cutoff, "m", unit)
    if title is None:
        title = f"Overlap of {cutoff_units} {unit} {variable} areas between reservoirs"
    reservoir_index = reservoir_entry_to_index(Model, reservoir)
    data = (
        (Model[variable].isel(reservoir=reservoir_index) < -cutoff)
        .sum(dim="reservoir")
        .isel(time=time_index[-1])
    )
    raster = {
        "X": data.x.values,
        "Y": data.y.values,
        "mask": np.flip(data.values, axis=0),
    }

    if not plot_subsidence_contours:
        fig, ax = plot_reservoirs(
            Model,
            reservoir=reservoir,
            model=None,
            buffer=0,
            annotate=False,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            final=False,
            shape_kwargs=shape_kwargs,
            annotation_kwargs=annotation_kwargs,
            raster_kwargs=raster_kwargs,
            scatter_kwargs=scatter_kwargs,
        )
        bbox_to_anchor = (1.0, 0.5)
    else:
        fig, ax = plot_subsidence_model(
            Model,
            reservoir=None,
            time=time_index[-1],
            buffer=buffer,
            variable=variable,
            unit=unit,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            title=title,
            final=False,
            fname="",
            svg=False,
            shape_kwargs={},
            raster_kwargs={},
            contourf_kwargs={},
            contour_kwargs={},
            clabel_kwargs={},
            colorbar_kwargs={},
        )
        bbox_to_anchor = (1.1, 0.5)
    adjusted_overlap_kwargs = set_defaults(
        {"zorder": 10000}, defaults=overlap_kwargs
    )
    im = add_raster(ax, raster, raster_kwargs=adjusted_overlap_kwargs)
    if legend:
        values = np.unique(data)
        values = values[values != 0]
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = []
        for i, value in enumerate(values):
            label = (
                f"Overlap between {values[i]} reservoirs"
                if i != 0
                else "No overlap"
            )
            patches.append(Patch(color=colors[i], label=label))
        # patches = [ Patch(color=colors[i], label=f"Overlap between {values[i] - 1} reservoirs") for i in range(len(values)) ]
        ax.legend(
            handles=patches, loc="center left", bbox_to_anchor=bbox_to_anchor
        )

    add_title(ax, title)

    if final:
        savefig(Model, fname, svg=svg)
        plt.show()
        plt.close()
    else:
        return fig, ax


def _individual_m(
    ax,
    Model,
    variable,
    unit,
    reservoir_index,
    line_dict,
    steps,
    num,
    c,
    plot_kwargs={},
    annotation_kwargs={},
):
    c_index = 0
    for i, r in enumerate(Model.reservoirs):
        if i in reservoir_index:
            data = get_2D_data_from_model(
                Model, reservoir=i, time=steps, variable=variable, unit=unit
            )

            add_cross_section(
                ax,
                Model.x,
                Model.y,
                [data],
                line_dict.values(),
                name=[r],
                inflection_point_names=None,
                c=[c[c_index]],
                num=num,
                plot_kwargs=plot_kwargs,
                annotation_kwargs=annotation_kwargs,
            )

            c_index += 1

    total_data = get_2D_data_from_model(
        Model,
        reservoir=reservoir_index,
        time=steps,
        variable=variable,
        unit=unit,
    )
    add_cross_section(
        ax,
        Model.x,
        Model.y,
        [total_data],
        line_dict.values(),
        name=["Total"],
        inflection_point_names=None,
        c="k",
        num=num,
        plot_kwargs=plot_kwargs,
        annotation_kwargs=annotation_kwargs,
    )


def _cumulative_m(
    ax,
    Model,
    variable,
    unit,
    reservoir_index,
    line_dict,
    steps,
    num,
    c,
    fill_between_kwargs={},
):
    _, adjusted_fill_between_kwargs = seperate_colors_from_dict(
        fill_between_kwargs, len(reservoir_index)
    )
    c_index = 0
    areas = []
    for i, r in enumerate(reservoir_index):
        adjusted_fill_between_kwargs["facecolor"] = c[c_index]
        adjusted_fill_between_kwargs[
            "zorder"
        ] = -c_index  # make sure the shape is below the previous
        data = get_2D_data_from_model(
            Model,
            reservoir=reservoir_index[: i + 1],
            time=steps,
            variable=variable,
            unit=unit,
        )
        (values, distances, inflection_distances) = get_cross_section(
            Model.x, Model.y, [data], line_dict.values(), num
        )

        area = add_filled_area(
            ax,
            distances,
            np.zeros_like(distances),
            values[0],
            label=Model.reservoirs[r],
            kwargs=adjusted_fill_between_kwargs,
        )

        for l in range(len(line_dict) - 1):
            if l == 0:

                ax.axvline(
                    x=inflection_distances[0],
                    c="k",
                    linestyle="dashed",
                    alpha=0.25,
                    label="_nolegend_",
                )
                ax.annotate(
                    list(line_dict.keys())[l],
                    (inflection_distances[0], 0),
                    path_effects=WHITE_SHADOW,
                )

            ax.axvline(
                x=inflection_distances[l + 1],
                c="k",
                linestyle="dashed",
                alpha=0.25,
                label="_nolegend_",
            )
            ax.annotate(
                list(line_dict.keys())[l + 1],
                (inflection_distances[l + 1], 0),
                path_effects=WHITE_SHADOW,
            )

        c_index += 1

    for area, color in zip(areas, c):
        area.set_facecolor(color)


def plot_overlap_cross_section(
    Model,
    lines=None,
    mode="cumulative",
    variable="subsidence",
    reservoir=None,
    time=-1,
    model=None,
    unit="cm",
    num=1000,
    zoom_level=10,
    figsize=(12, 8),
    epsg=28992,
    title=None,
    contour_levels=None,
    contour_steps=0.01,
    y_axis_exageration_factor=2,
    ylim=None,
    final=True,
    fname="overlap",
    svg=False,
    legend=True,
    horizontal_line=None,
    plot_kwargs={},
    annotation_kwargs={},
    fill_between_kwargs={},
):

    """Plot a map of the cross section in a 2D representation, and
    plot a line or set of lines of the subsidence along that cross section.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    lines : List, float/int
        List of tuples with points representing the x- and y-coordinates
        in the dataset coordinate system. These points represent the line
        the cross section will be drawn along.
    mode : string, optional
        Labels for selecting the type of plotting used. You can only use 'cumulative'
        or 'individual'. If not one of these values, a warning will occur and
        fall back to the default. The default value is 'cumulative'. When cumulative,
        the subsidence will be summed after each reservoir that is added to the
        plot and the area underneath will be filled with the color of each reservoir.
        The type 'individual' will produce plots of the subsidence caused by each
        reservoir and the total of the reservoir.
    variable : str: optional
        Any Model variable that is present in the model and can be represented as
        a grid. Default is 'subsidence'. Other values can be "slope", "compaction",
        "pressure", etc.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a
        list, an Exception will occur. The default is -1, the final
        timestep.
    model : int, str or list of str or int
        Label - or list of labels - of the models that you want to plot the cross
        section of. The default is None, then all model will be plotted.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    num : int, optional
        The amount of points sampled along the crossection.
        The default is 1000.
    figsize : tuple, float, optional
        The size of the figure with the cross section plots in inches.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    epsg : int, optional
        The available epsg of the WMTS service
    title : str, optional
        The title of the figure displaying the cross section data.
        The default is None.
    y_axis_exageration_factor : int/float, optional
        The factor the length of the y_axis will be exagerated. If the lowest data
        point in the graph is -1, the y-axis will be from -y_axis_exageration_factor
        to the highest point. The default is 2.
    ylim : tuple, float, optional
        A tuple of values determinging the extend of the y-axis.The default is None.
        When None, the y-axis will be determined using the data and
        y_axis_exageration_factor.
    final : bool
        When True, the figure will be shown en return immutable. If False,
        the matplotlib figure and ax(es) objects wqill be returned.
    fname : str
        The location the figure will be saved it. The default is '' this, will
        indicate the figure will not be stored. When a path is given, this figure
        will be stored at that location, when just a name is given, the figure
        will be stored in the project folder.
    svg : bool
        Save this file also as an svg-file.
    legend : bool
        When True, shows a legend, when False, the figure will plot no legend.
        The default is True.
    horizontal_line : float/dict
        When a float, it must be the value on the y-axis at which the horizontal
        line will be placed. When a dict, the key of the dictionary will be the label
        of the line and the entry will be the value on the y-axis the horizontal
        line will be plotted along.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines.
        The default is {}. See SubsidenceModel attribute plot_defaults
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.
    fill_between_kwargs : dict, optional
        Dictionary with the keyword arguments for the filled area between lines.
        The default is {}. See SubsidenceModel attribute fill_area_defaults
        for additional information.

    Returns
    -------
    None
    OR
    cross section fig, ax and map fig and ax

    """

    cumulative = mode == "cumulative"
    not_cumulative = mode == "individual"
    if not cumulative and not not_cumulative:
        warn(
            f'Warning: mode: {mode} is not recognised. The available options are "cumulative" and "individual". Set to "cumulative".'
        )
        cumulative = True

    if lines is None:
        line_dict = ask_for_line(
            Model, zoom_level=zoom_level, figsize=figsize, epsg=epsg
        )
    elif not isinstance(lines, dict):
        line_dict = {}
        for i, line in enumerate(lines):
            line_dict[string.ascii_uppercase[i % 26]] = line
    else:
        line_dict = lines

    unit_label = get_unit_label(variable, unit)

    if _utils.isSubsidenceModel(Model):

        steps = time_entry_to_index(Model, time)

        reservoir_index = reservoir_entry_to_index(Model, reservoir)

        # plot cross sections
        l_fig, l_ax = plt.subplots(figsize=figsize)

        if not_cumulative:
            c, adjusted_plot_kwargs = seperate_colors_from_dict(
                plot_kwargs, len(reservoir_index)
            )
            _individual_m(
                l_ax,
                Model,
                variable,
                unit,
                reservoir_index,
                line_dict,
                steps,
                num,
                c,
                plot_kwargs=adjusted_plot_kwargs,
                annotation_kwargs=annotation_kwargs,
            )
        if cumulative:
            c, adjusted_fill_between_kwargs = seperate_colors_from_dict(
                fill_between_kwargs, len(reservoir_index)
            )
            _cumulative_m(
                l_ax,
                Model,
                variable,
                unit,
                reservoir_index,
                line_dict,
                steps,
                num,
                c,
                fill_between_kwargs=adjusted_fill_between_kwargs,
            )

        l_ax.grid()
        if title is None:
            title = f"{variable.capitalize()} {unit_label} per reservoir and total {variable}."
        add_title(l_ax, title)
        if horizontal_line is not None:
            add_horizontal_line(
                l_ax, horizontal_line, unit=unit, label="__nolegend__"
            )
        if legend:
            l_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        l_ax.set_ylabel(f"{variable.capitalize()} ({unit_label})")
        l_ax.set_xlabel("Distance (m)")
        set_ylim(
            l_ax,
            ylim=ylim,
            y_axis_exageration_factor=y_axis_exageration_factor,
            unit_in="m",
            unit_out=unit,
        )

        if final:
            if fname:
                savefig(Model, fname, svg=svg)
            plt.show()
        else:
            return l_fig, l_ax

    elif _utils.isSubsidenceSuite(Model):
        model = Model.model_label_to_index(model)

        # Cross section
        reservoir_dict = Model.reservoir_dict()
        reservoir_index = reservoir_entry_to_index(Model, reservoir)

        for r, i in reservoir_dict.items():
            if i not in reservoir_index:
                reservoir_dict[r].drop()
        c, adjusted_plot_kwargs = seperate_colors_from_dict(
            plot_kwargs, len(reservoir_dict)
        )
        color_dict = {ur: cr for ur, cr in zip(reservoir_dict.keys(), c)}

        fig = plt.figure()
        number_of_axes = len(model)

        number_of_rows = number_of_axes // 2 + 1
        number_of_columns = min(number_of_axes, 2)

        counter = 0
        map_title = []
        for i, m in enumerate(Model._models):
            if i in model:
                steps = time_entry_to_index(m, time)
                data_coords = list(m.grid[variable].coords)
                if "time" not in data_coords:
                    steps = [steps[-1]]
                _legend_labels = time_to_legend(steps, m)
                _legend_labels = [
                    label + " " + m.name for label in _legend_labels
                ]

                map_title.append(
                    f"Cross section - {m.name} - {_legend_labels[-1]}"
                )

                ax = fig.add_subplot(
                    number_of_rows, number_of_columns, counter + 1
                )
                if not cumulative:
                    _individual_m(
                        ax,
                        m,
                        variable,
                        unit,
                        reservoir_index,
                        line_dict,
                        steps,
                        num,
                        c,
                        plot_kwargs=adjusted_plot_kwargs,
                        annotation_kwargs=annotation_kwargs,
                    )
                elif cumulative:
                    _cumulative_m(
                        ax,
                        m,
                        variable,
                        unit,
                        reservoir_index,
                        line_dict,
                        steps,
                        num,
                        c,
                        fill_between_kwargs=fill_between_kwargs,
                    )

                unit_label = get_unit_label(variable, unit)

                if title is None:
                    title = f"Cross section {variable} ({unit_label})"

                add_title(ax, title)
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel(
                    f"{variable.capitalize().replace('_', ' ')} ({unit_label})"
                )
                set_ylim(
                    ax,
                    ylim=ylim,
                    y_axis_exageration_factor=y_axis_exageration_factor,
                    unit_in="m",
                    unit_out=unit,
                )
                add_horizontal_line(ax, horizontal_line, unit=unit)

                ax.grid()

                counter += 1

        legend_handles = [
            Line2D([0], [0], color=color_dict[r]) for r in color_dict.keys()
        ]
        legend_handles += [Line2D([0], [0], color="k")]
        legend_labels = list(color_dict.keys()) + ["Total"]
        if legend:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

        fig.set_size_inches(figsize)
        fig.tight_layout()

        if fname:
            fname_cross_section = f"{fname}_cross_section_{variable}"
            savefig(Model, fname_cross_section, svg=svg)

        if final:
            plt.show()
            plt.close()
        else:
            return fig, ax


def plot_map_with_line(
    Model,
    lines=None,
    variable="subsidence",
    reservoir=None,
    time=-1,
    model=None,
    plot_reservoir_shapes=True,
    additional_shapes=[],
    unit="cm",
    buffer=0,
    num=1000,
    figsize=(8, 8),
    zoom_level=10,
    epsg=28992,
    title=None,
    contour_levels=None,
    contour_steps=0.01,
    y_axis_exageration_factor=2,
    ylim=None,
    final=True,
    fname="overlap",
    svg=False,
    legend=True,
    horizontal_line=None,
    plot_kwargs={},
    raster_kwargs={},
    shape_kwargs={},
    contourf_kwargs={},
    contour_kwargs={},
    clabel_kwargs={},
    colorbar_kwargs={},
    annotation_kwargs={},
    fill_between_kwargs={},
):

    """Plot a map of the cross section in a 2D representation, and
    plot a line or set of lines of the subsidence along that cross section.

    Parameters
    ----------
    Model : SubsidenceModel or ModelSuite objects
    lines : list/dict, float/int
        This variable is a list or a dictionary.
        Valid formats are:
            A single line: [[0, 1], [0,2]]
            Multiple lines: [[[1, 1], [1,2]], [[1, 1], [1,2]]]
            A dictionary for a single line: {"Point 1": [[0, 1], [0,2]]}
            A list of dictionaries for multiple lines:
                {"Point 1": [[1, 1], [1,2]], "Point 2": [[1, 1], [1,2]]}
        These lines represent the line
        the cross section will be drawn along.
    variable : str: optional
        Any Model variable that is present in the model and can be represented as
        a grid. Default is 'subsidence'. Other values can be "slope", "compaction",
        "pressure", etc.
    reservoir : int, str or list of int or str, optional
        The index or name of the reservoirs you want to plot. If it is a
        list, multiple reservoirs will be displayed. The default is None.
        When None, all reservoirs will be displayed.
    time : int, str, optional
        The index or name of the timestep you want to plot. If it is a
        list, an Exception will occur. The default is -1, the final
        timestep.
    model : int, str or list of str or int
        Label - or list of labels - of the models that you want to plot the cross
        section of. The default is None, then all model will be plotted.
    plot_reservoir_shapes : bool, optional
        The cross section draws a line on a map, to show where the cross section
        is taken over. If this value is True, this map will show the reservoirs
        as in the model. If False, the reservoirs are not shown.
    unit : str, optional
        The SI-unit the data will be plotted in. The default is 'cm'.
        Also available are 'mm', 'm' and 'km'.
    buffer : float/int, optional.
        Additional space to be added to the edge of the plotted
        figure in m. The default is 0.
    num : int, optional
        The amount of points sampled along the crossection.
        The default is 1000.
    plot_figsize : tuple, float, optional
        The size of the figure with the cross section plots in inches.
    zoom_level : int, optional
        An integer indicating the zoom level for the maps. Low numbers
        show maps on a large scale, higher numbers show maps on a smaller
        scale.
    map_figsize : tuple, float, optional
        The size of the figure with maps in inches.
    epsg : int, optional
        The available epsg of the WMTS service
    title: str, optional
        The title of the figure displaying the map and cross section line.
        The default is None.
    contour_levels : list, float, optional
        Draw contour lines at the specified levels. The values must be in increasing order.
        The default is None. When None, the contour lines will be chosen based
        on the subsidence data and the contour_steps parameter.
    contour_steps : float/int, optional
        The difference in values between the contour levels. The default is 0.01.
    y_axis_exageration_factor : int/float, optional
        The factor the length of the y_axis will be exagerated. If the lowest data
        point in the graph is -1, the y-axis will be from -y_axis_exageration_factor
        to the highest point. The default is 2.
    ylim : tuple, float, optional
        A tuple of values determinging the extend of the y-axis.The default is None.
        When None, the y-axis will be determined using the data and
        y_axis_exageration_factor.
    final : bool
        When True, the figure will be shown en return immutable. If False,
        the matplotlib figure and ax(es) objects wqill be returned.
    fname : str
        The location the figure will be saved it. The default is '' this, will
        indicate the figure will not be stored. When a path is given, this figure
        will be stored at that location, when just a name is given, the figure
        will be stored in the project folder.
    svg : bool
        Save this file also as an svg-file.
    legend : bool
        When True, shows a legend, when False, the figure will plot no legend.
        The default is True.
    horizontal_line : float/dict
        When a float, it must be the value on the y-axis at which the horizontal
        line will be placed. When a dict, the key of the dictionary will be the label
        of the line and the entry will be the value on the y-axis the horizontal
        line will be plotted along.
    plot_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted lines.
        The default is {}. See SubsidenceModel attribute plot_defaults
    contourf_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted filled contours.
        The default is {}. See SubsidenceModel attribute contourf_defaults
        for additional information.
    contour_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour lines.
        The default is {}. See SubsidenceModel attribute contour_defaults
        for additional information.
    clabel_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted contour line labels.
        The default is {}. See SubsidenceModel attribute clabel_defaults
        for additional information.
    colorbar_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted colorbar.
        The default is {}. See SubsidenceModel attribute colorbar_defaults
        for additional information.
    annotation_kwargs : dict, optional
        Dictionary with the keyword arguments for the plotted annotations.
        The default is {}. See SubsidenceModel attribute annotation_defaults
        for additional information.
    fill_between_kwargs : dict, optional
        Dictionary with the keyword arguments for the filled area between lines.
        The default is {}. See SubsidenceModel attribute fill_area_defaults
        for additional information.

    Returns
    -------
    None
    OR
    cross section fig, ax and map fig and ax

    """
    if lines is None:
        line_dict = ask_for_line(
            Model, zoom_level=zoom_level, figsize=figsize, epsg=epsg
        )
        lines = [line_dict]
    elif isinstance(lines, dict):
        lines = [lines]

    line_type = [type(l) == dict for l in lines]

    if not all(line_type):
        if _utils.isjagged(lines) or len(np.array(lines).shape) == 3:
            _lines = []
            str_count = _utils.letter_counter()
            for l in lines:

                if not type(l) == dict:
                    line_dict = {}
                    for _, line_segment in enumerate(l):
                        line_dict[str_count.get_letter()] = line_segment
                        str_count.count += 1
                    _lines.append(line_dict)
                else:
                    _lines.append(l)
            lines = _lines
        elif len(np.array(lines).shape) == 2:
            line_dict = {}
            str_count = _utils.letter_counter()
            for _, line_segment in enumerate(lines):
                line_dict[str_count.get_letter()] = line_segment
                str_count.count += 1
            lines = [line_dict]
        else:
            raise Exception("Wrong line format")

    unit_label = get_unit_label(variable, unit)

    if _utils.isSubsidenceModel(Model):
        steps = time_entry_to_index(Model, time)

        if title is None:
            title = f"Cross section {variable} ({unit_label}) - {time_to_legend(Model.timesteps[steps][-1], Model)[0]}"

        m_fig, m_ax = plot_subsidence(
            Model,
            reservoir=reservoir,
            time=time,
            buffer=buffer,
            variable=variable,
            unit=unit,
            additional_shapes=additional_shapes,
            title=title,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            plot_reservoir_shapes=plot_reservoir_shapes,
            final=False,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
            colorbar_kwargs=colorbar_kwargs,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
        )

        for line_dict in lines:
            add_lines(
                ax=m_ax, line=line_dict, annotation_kwargs=annotation_kwargs
            )

        if final:
            if fname:
                fname_map = f"{fname}_map_{variable}"
                savefig(Model, fname_map, svg=svg)
            plt.show()

        else:
            return m_fig, m_ax

    if _utils.isSubsidenceSuite(Model):

        m_fig, m_ax = plot_subsidence(
            Model,
            reservoir=reservoir,
            time=time,
            model=model,
            variable=variable,
            plot_reservoir_shapes=plot_reservoir_shapes,
            additional_shapes=additional_shapes,
            buffer=buffer,
            unit=unit,
            title=title,
            zoom_level=zoom_level,
            figsize=figsize,
            epsg=epsg,
            contour_levels=contour_levels,
            contour_steps=contour_steps,
            shape_kwargs=shape_kwargs,
            raster_kwargs=raster_kwargs,
            final=False,
            contourf_kwargs=contourf_kwargs,
            contour_kwargs=contour_kwargs,
            clabel_kwargs=clabel_kwargs,
            colorbar_kwargs=colorbar_kwargs,
        )

        for _ax in m_ax:
            for line_dict in lines:
                add_lines(
                    ax=_ax, line=line_dict, annotation_kwargs=annotation_kwargs
                )

        if final:
            plt.show()
            if fname:
                fname_map = f"{fname}_map_{variable}"
                savefig(Model, fname_map, svg=svg)
        else:
            return m_fig, m_ax
