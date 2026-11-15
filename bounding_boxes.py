from collections import namedtuple
import math
import numpy as np

class BoundingBox:
    def __init__(self, left, right, bottom, top):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
    
    @property
    def width(self):
        return self.right-self.left+1

    @property
    def height(self):
        return self.top-self.bottom+1

    def crop_image(self, image: np.typing.ArrayLike, height_axis=0, width_axis=1):
        assert len(image.shape) >= 2, f"image.shape=[{','.join(image.shape)}] must have at least 2 dimensions for width and height"

        height = image.shape[height_axis]
        width = image.shape[width_axis]
        assert self.right < width, f"right={self.right} must be less than image_width={width}"
        assert self.top < height, f"top={self.top} must be less than image_height={height}"

        total_dims = len(image.shape)

        nd_index = [None for _ in range(total_dims)]
        nd_index[width_axis] = slice(self.left, self.right+1)
        nd_index[height_axis] = slice(self.bottom, self.top+1)
        crop = image[*nd_index]
        return crop

    def __str__(self):
        return f"BoundingBox(x={self.left}..={self.right},y={self.bottom}..={self.top},dx={self.width},dy={self.height})"

    def __repr__(self):
        return self.__str__()

class BoundingBoxSplitter:
    @staticmethod
    def load_from_json(json: dict):
        parent = BoundingBox(
            json["parent"]["left"],
            json["parent"]["right"],
            json["parent"]["bottom"],
            json["parent"]["top"],
        )
        splitter = BoundingBoxSplitter(parent)
        splitter.total_columns = json["total_columns"]
        splitter.total_rows = json["total_rows"]
        splitter.overlap_x = json["overlap_x"]
        splitter.overlap_y = json["overlap_y"]
        return splitter

    def __init__(self, parent=BoundingBox(0,0,0,0)):
        self.parent = parent
        self.total_columns = 0
        self.total_rows = 0
        self.overlap_x = 0
        self.overlap_y = 0
    
    def get_child_boxes(self):
        # X = total columns
        # Y = total rows
        # W = bbox width
        # H = bbox height
        # w = crop width
        # h = crop height
        # kx = overlay x
        # ky = overlay y

        # W = X*w - (X-1)*kx*w
        crop_width = self.parent.width/(self.total_columns - (self.total_columns-1)*self.overlap_x)
        # H = Y*h - (Y-1)*ky*h
        crop_height = self.parent.height/(self.total_rows - (self.total_rows-1)*self.overlap_y)

        stride_x = crop_width*(1-self.overlap_x)
        stride_y = crop_height*(1-self.overlap_y)

        for x in range(self.total_columns):
            for y in range(self.total_rows):
                left = math.floor(self.parent.left + x*stride_x)
                right = math.floor(self.parent.left + x*stride_x + crop_width - 1)
                bottom = math.floor(self.parent.bottom + y*stride_y)
                top = math.floor(self.parent.bottom + y*stride_y + crop_height - 1)
                yield BoundingBox(left, right, bottom, top)
    
    def split_image(self, image: np.typing.ArrayLike, height_axis=0, width_axis=1):
        assert len(image.shape) >= 2, f"image.shape=[{','.join(image.shape)}] must have at least 2 dimensions for width and height"

        height = image.shape[height_axis]
        width = image.shape[width_axis]
        assert self.parent.right < width, f"parent.right={self.parent.right} must be less than image_width={width}"
        assert self.parent.top < height, f"parent.top={self.parent.top} must be less than image_height={height}"

        bboxes = self.get_child_boxes()
        for bbox in bboxes:
            yield bbox.crop_image(image, height_axis=height_axis, width_axis=width_axis)
