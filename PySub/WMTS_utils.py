"""https://scitools.org.uk/cartopy/docs/latest/_modules/cartopy/io/img_tiles.html
"""
from PIL import Image
from string import Formatter
from cartopy.io.img_tiles import GoogleWTS

URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/{map_service}/MapServer/tile/{z}/{y}/{x}.jpg'
URL_KEYWORD_DICT = {'map_service': 'World_Topo_Map'}


def formatted_string_keywords(string):
    return [fname for _, fname, _, _ in Formatter().parse(string) if fname]


class Tiles(GoogleWTS):
    def __init__(self, desired_tile_form='RGB', 
                 cache=False):
        """
        Parameters
        ----------
        desired_tile_form: optional
            Defaults to 'RGB'.
        style: optional
            The style for the Google Maps tiles.  One of 'street',
            'satellite', 'terrain', and 'only_streets'.  Defaults to 'street'.
        url: optional
            URL pointing to a tile source and containing {x}, {y}, and {z}.
            Such as: ``'https://server.arcgisonline.com/ArcGIS/rest/services/\
                World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'``

        """
        self.url = URL
        keywords = formatted_string_keywords(self.url)
        
        keywords = [k for k in keywords if k not in ['z', 'y', 'x']]
        missing = [k for k in keywords if k not in URL_KEYWORD_DICT]
        if len(missing) > 0:
            raise Exception(f'The keywords {missing} are missing from the URL_KEYWORD_DICT')
        self.keyword_dict = {
            k: URL_KEYWORD_DICT[k]
            for k in keywords
            }

        # The 'satellite' and 'terrain' styles require pillow with a jpeg
        # decoder.
        if not hasattr(Image.core, "jpeg_decoder") or \
            not Image.core.jpeg_decoder:
            raise ValueError(
                "The service requires pillow with jpeg decoding "
                "support.")
        return super().__init__(desired_tile_form=desired_tile_form,
                                cache=cache)


    def _image_url(self, tile):
        url = self.url.format(
            x=tile[0], X=tile[0],
            y=tile[1], Y=tile[1],
            z=tile[2], Z=tile[2],
            **self.keyword_dict)
        return url
