try:
    import os
    import multiresolutionimageinterface as mir
except:
    import os
    import sys
    ASAP_bin_path = os.path.join(os.getcwd(), '/opt/ASAP/bin/')  #这里是ASAP的路径，linux/win10/win11下相对应修改一下成为自己的即可
    os.environ['PATH'] = ASAP_bin_path + ";" + os.environ['PATH']
    sys.path.append(ASAP_bin_path)
    import multiresolutionimageinterface as mir
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 700000000


__all__ = ['Reader', 'Writer']


class Reader(mir.MultiResolutionImage):
    def __init__(self, svs_file, level=2):
        """
        open svs file with open-slide
        :param svs_file: svs file, absolute path
        :return: slide
        """
        # super().__init__(svs_file)
        self._filepath = svs_file
        self._basename = os.path.basename(svs_file).split('.')[0]
        reader = mir.MultiResolutionImageReader()
        self.slide = reader.open(svs_file)
        self._level = level

    def getDimensions(self):
        """
        return (w, h)
        """
        return self.slide.getDimensions()

    def get_basename(self):
        """
        return svs file basename, not contain file suffix
        :return:
        """

        return self._basename

    def get_filepath(self):
        """
        get absolute svs file
        :return:
        """

        return self._filepath

    def get_level(self):
        """
        return level
        :return:
        """

        return self._level

    def get_level_count(self):
        """
        return number of levels
        :return:
        """

        return self.slide.getNumberOfLevels()

    def get_level_downsample(self, level=2):
        """
        get the expected level downsample, default level two
        :param level: level, default 2
        :return: the level downsample
        """

        return self.slide.getLevelDownsample(level)

    def get_level_dimension(self, level=2):
        """
        get the expected level dimension, default level two
        :param level: level, default 0
        :return:
        """

        return self.slide.getLevelDimensions(level)

    # def svs_to_png(self,save_dir):
    #     """
    #     convert svs to png
    #     :return:
    #     """
    #     self.get_thumb().save(save_dir)

    # def expand_img(self, im, size, value=(0, 0, 0)):
    #     """
    #     expand the image
    #     :param im: the image want to expand
    #     :param size: tuple, the size of expand
    #     :param value: tuple, the pixel value at the expand region
    #     :return: the expanded image
    #     """

    #     im_new = Image.new("RGB", size, value)
    #     im_new.paste(im, (0, 0))

    #     return im_new

    def get_mpp(self):
        """
        get the value of mpp
        :return: 0.00025/0.0005
        """
        mpp = self.slide.getProperty('openslide.mpp-x')
        if mpp == '':
            mpp = self.slide.getSpacing()[0]

        return np.float32(mpp) / 1000

    def get_thumb(self, level=2):
        """
        get thumb image
        :return:
        """
        tile_size = self.slide.getLevelDimensions(level)
        tile = self.slide.getUCharPatch(
            startX=0, startY=0, width=tile_size[0], height=tile_size[1], level=level
        )

        return tile

    def read_region(self, location: tuple, level: int, size: tuple):
        """
        get the tile base on the input coordinate tuple and level value and tile_size tuple
        tuple is requried as (x,y)
        :return: tle in np.array format
        """
        tile = self.slide.getUCharPatch(
            startX=location[0], startY=location[1], width=size[0], height=size[1], level=level
        )

        return tile

    def get_best_level_downsample(self, downsample: int):
        """
        get the best downsample level for input target downsample value
        :return:
        """
        return self.slide.getBestLevelForDownSample(downsample)
    # def __del__(self):
    #     self.slide.close()


class Writer(mir.MultiResolutionImage):
    def __init__(self, output_path: str, tile_size: int, dimensions: tuple, spacing: float, color_type: str, *args, **kwargs):
        self.output_path = output_path
        self.tile_size = tile_size
        self.W, self.H = dimensions
        # 要求横纵分辨率一致
        self.spacing = spacing
        self.color_type = color_type

        self._writer = mir.MultiResolutionImageWriter()
        self._writer.openFile(self.output_path)
        self._writer.setTileSize(self.tile_size)

        if color_type=='MONOCHROME':
            self._writer.setCompression(mir.Compression_LZW)
        elif color_type=='RGB':
            self._writer.setCompression(mir.Compression_JPEG)
        # 两个版本间存在一些命名不同
        # self._writer.setCompression(mir.Compression_JPEG)#(mir.Compression_LZW)
        self._writer.setJPEGQuality(95)
        self._writer.setDataType(mir.DataType_UChar)
        self._writer.setInterpolation(mir.Interpolation_NearestNeighbor)
        # color_type: Monochrome, RGB, RGBA, Indexed
        color_type = {
            'MONOCHROME': mir.ColorType_Monochrome,
            'RGB': mir.ColorType_RGB,
            'RGBA': mir.ColorType_RGBA,
            'INDEXED': mir.ColorType_Indexed,
        }[self.color_type.upper()]

        self._writer.setColorType(color_type)
        self._writer.writeImageInformation(self.W, self.H)
        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(self.spacing)
        pixel_size_vec.push_back(self.spacing)
        self._writer.setSpacing(pixel_size_vec)


    def write(self, tile: np.ndarray, x: int, y: int):
        assert tile.shape[0] == tile.shape[1] == self.tile_size, f'要求写入数与维度数对齐{tile.shape} -- {self.tile_size}'
        self._writer.writeBaseImagePartToLocation(tile.flatten().astype('uint8'), x=int(x), y=int(y))

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     if not exc_type and not exc_val and not exc_tb:
    #         self._writer.finishImage()
    #     else:
    #         traceback.print_exc()
    #         # 否则删掉重来 -- 实测没啥用
    #         self._writer.finishImage()
    #         os.remove(self.output_path)
    #     return False
    def close(self):
        self._writer.finishImage()
