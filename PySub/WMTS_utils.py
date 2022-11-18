"""https://github.com/SciTools/cartopy/issues/1049
"""
from owslib.wmts import WebMapTileService

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

PIXEL_SIZE = 0.00028 # in 1/m, works for Dutch maps...

CASH = {}

def other_WMTS(
        ax,
        url,
        layer, 
        wmts_kwargs = {}
        ):
    """General WMTS tool. Not all WMTS work with this. See WMTS class for more flexible options.
    """
    ax.add_wmts(url, layer_name=layer, wmts_kwargs = wmts_kwargs)


class WMTS(WebMapTileService):
    """Base WMTS client, used to inherit from. See OpenTopo for example"""
    def __init__(self, url, layer, ax = None, epsg = 28992):
        self.layer = layer
        self.ax = ax
        self.set_identifier = f'EPSG:{epsg}'
        self.tiles = []
        WebMapTileService.__init__(self, url)
        
    def clear_tiles(self):
        for tile in self.tiles:
            tile.remove()
        self.tiles = []
    
    def set_ax(self, ax):
        """Set the ax of the WMTS-object"""
        self.ax=ax

    def plot(self, ax=None,
             **imshow_kwargs):
        """Plot the backgrond map on a matplotlib ax.

        Parameters
        ----------
        ax: matplotlib Axes (optional)
            The axes in which the background map should be plotted
        **imshow_kwargs: dict
            a dictionary with keyword arguments to be passed into imshow
        
        Returns
        -------
        ims: list
            list of image handles, returned from imshow
    
        """
        if ax is not None:
            self.ax = ax
        elif self.ax is None:
            self.ax = plt.gca()
        
        if not self.set_identifier in self.tilematrixsets.keys():
            raise(Exception(f'Set id {self.set_identifier} Does not exist. Available sets: {list(self.tilematrixsets.keys())}'))

        resolution = self.get_resolution()
       
        canvas = self.ax.axis()
        zoom_level = self.get_zoom(resolution, canvas)
        
        # determine the needed matrix and tiles
        
        tm = self.tilematrixsets[self.set_identifier].tilematrix[zoom_level]
        rows, columns = self.get_locations_tiles(tm, canvas)
        
        self.clear_tiles()
        
        for i, op in enumerate(self.operations):
            if(not hasattr(op, 'name')):
                self.operations[i].name = ""
        for row in rows:
            for column in columns:
                left = tm.topleftcorner[0] + column * tm.tilewidth * tm.scaledenominator * PIXEL_SIZE
                right = tm.topleftcorner[0] + (column + 1) * tm.tilewidth * tm.scaledenominator * PIXEL_SIZE
                bottom = tm.topleftcorner[1] - (row + 1) * tm.tileheight * tm.scaledenominator * PIXEL_SIZE
                top = tm.topleftcorner[1] - row * tm.tileheight * tm.scaledenominator * PIXEL_SIZE
                
                im = self.download_tile(tm, row, column)
                
                extent = (left, right, bottom, top)
                im = self.ax.imshow(im, extent=extent, **imshow_kwargs)
                self.tiles.append(im)
        return self.tiles

    def get_locations_tiles(self, tilematrix, canvas):
        """Determine the row and the column of every tile that should be used in the plot

        Parameters
        ----------
        tilematrix : 2D tilematrix object
            The map, chosen by projection and zoom level.
        canvas : array-like
            The bounds of the canvas artist of the ax object, with lowest and highest x and y.

        Returns
        -------
        rows : list
            Rows of the timeatrix object where the tiles to be plotted are stored.
        columns : list
            Columns of the timeatrix object where the tiles to be plotted are stored.

        """
        xstep = tilematrix.tilewidth * tilematrix.scaledenominator * PIXEL_SIZE
        x_right_tile = xstep * tilematrix.matrixwidth
        x = np.linspace(tilematrix.topleftcorner[0],
                        tilematrix.topleftcorner[0] + x_right_tile,
                        tilematrix.matrixwidth + 1)
        columns = np.where(np.logical_and(x[1:] > canvas[0], x[:-1] < canvas[1]))[0]
        ystep = tilematrix.tilewidth * tilematrix.scaledenominator * PIXEL_SIZE
        y_top_tile = - ystep * tilematrix.matrixwidth
        y = np.linspace(tilematrix.topleftcorner[1],
                        tilematrix.topleftcorner[1] + y_top_tile,
                        tilematrix.matrixwidth + 1)
        rows = np.where(np.logical_and(y[1:] < canvas[3], y[:-1] > canvas[2]))[0]
        return rows, columns
    
    def get_zoom(self, resolution, canvas, scale = 1):
        """Determine the identifier of the zoom level that is best for plot visualization (based on scale)

        Parameters
        ----------
        resolution : dict
            The disired resolution .
        canvas : TYPE
            DESCRIPTION.
        scale : float, optional
            higher scale causes higher resolution, but smaller text in maps. The default is 1.

        Returns
        -------
        zoom_level : str
            identifier for zoom layer of wmts.

        """
        fig = self.ax.get_figure()
        width, height = fig.canvas.get_width_height()
        bbox = self.ax.get_position()
        resolution_x = (canvas[1] - canvas[0]) / (bbox.width * width)
        resolution_y = (canvas[3] - canvas[2]) / (bbox.height * height)
        ax_resolution = min(resolution_x, resolution_y) # in meter per pixel
        ax_resolution = ax_resolution / scale  
        zoom_level = min(resolution.items(), 
                         key = lambda map_resolution: abs(map_resolution[1] - ax_resolution))[0]
        return zoom_level

    def get_resolution(self):
        """Get the zoom layer identifier and resolution in meters of the selected tilematrixsets

        Returns
        -------
        resolution : dict
            keys: string with the zoom layer identifier.
            values: resolution in meters.

        """
        tilematrix_keys = self.tilematrixsets[self.set_identifier].tilematrix.keys()
        resolution = {}
        
        for key in tilematrix_keys:
            tilematrix = self.tilematrixsets[self.set_identifier].tilematrix[key]
            resolution[key] = tilematrix.scaledenominator * PIXEL_SIZE
        return resolution
    
    def _download_tile(self, tilematrix, row, column):
        """Download an image from the selected WMTS-server
        
        Parameters
        ----------
        tilematrix : 2D tilematrix object
            The map, chosen by projection and zoom level.
        row, column : integers
            location of the desired tile
            

        Returns
        -------
        rows : list
            Rows of the timeatrix object where the tiles to be plotted are stored.
        columns : list
            Columns of the timeatrix object where the tiles to be plotted are stored.
        """
        tile = self.gettile(layer = self.layer, 
                            tilematrix = tilematrix.identifier,
                            row = row, column = column)
        try:
            im = Image.open(tile).convert('RGB')
        except: 
            im = Image.open(tile)
     
        return im

    def download_tile(self, tilematrix, row, column):
        key = f"{self.set_identifier}_{tilematrix.identifier}_{row}_{column}"
        if key in CASH.keys():
            return CASH[key]
        else:
            tile = self._download_tile(tilematrix, row, column)
            CASH[key] = tile
            return tile

class OpenTopoAchtergrondKaart(WMTS):
    """An object to plot the OpenTopo-map (https://www.opentopo.nl/) on a matplotlib ax"""
    def __init__(self, ax=None, layer = 'opentopoachtergrondkaart'):
        url = 'https://geodata.nationaalgeoregister.nl/tiles/service/wmts?request=GetCapabilities&service=WMTS'
        WMTS.__init__(self, url, layer, ax)
        
class BrtAchtergrondKaart(WMTS):
    def __init__(self, ax=None, layer='standaard'):
        url = 'https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0?request=getcapabilities&service=wmts'
        if layer not in ['standaard', 'grijs', 'pastel', 'water']:
            exception = f'Unknown value for kind: {layer}. Specify None, "standaard, "grijs", "pastel" or "water"'
            raise(Exception(exception))
        WMTS.__init__(self, url, layer, ax)

class LuchtFoto(WMTS):
    """An object to plot the OpenTopo-map (https://www.opentopo.nl/) on a matplotlib ax"""
    def __init__(self, ax=None, layer = 'Actueel_ortho25'):
        url = 'https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=WMTS'
        WMTS.__init__(self, url, layer, ax)