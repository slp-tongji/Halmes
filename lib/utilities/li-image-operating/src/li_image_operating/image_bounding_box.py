import textwrap
from typing import Literal
import typing

import tomlkit
import tomlkit.exceptions
import tomlkit.items

from li_image_operating.image_point import ImagePoint
from li_image_operating.in_memory_image import InMemoryImage

class ImageBoundingBox:
    def __init__(self, left_top: ImagePoint, right_bottom: ImagePoint) -> None:
        if not left_top.have_same_image_size(right_bottom):
            raise ValueError("The two given points do not share a same image size.")

        if left_top.x > right_bottom.x:
            raise ValueError(f"The two given points are not the left top corner and right bottom corner: "
                             f"left_top.x ({left_top.x}) > right_bottom.x ({right_bottom.x}).")
        if left_top.y > right_bottom.y:
            raise ValueError(f"The two given points are not the left top corner and right bottom corner: "
                             f"left_top.x ({left_top.y}) > right_bottom.y ({right_bottom.y}).")

        self.__min = left_top
        self.__max = right_bottom

    @staticmethod
    def from_x_y(x1: float, y1: float, 
                 x2: float, y2: float, 
                 image_width: int, image_height: int):
        if x1 > x2:
            temp: float = x1
            x1 = x2
            x2 = temp
        
        if y1 > y2:
            temp = y1
            y1 = y2
            y2 = temp

        return ImageBoundingBox(ImagePoint(x1, y1, image_width, image_height),
                                ImagePoint(x2, y2, image_width, image_height))

    @property
    def left_top(self) -> ImagePoint:
        return self.__min

    @property
    def right_bottom(self) -> ImagePoint:
        return self.__max

    def get_right_top(self) -> ImagePoint:
        return self.left_top.create_point_on_same_image(self.right_bottom.x, 
                                                        self.left_top.y)

    def get_left_bottom(self) -> ImagePoint:
        return self.left_top.create_point_on_same_image(self.left_top.x, 
                                                        self.right_bottom.y)

    @property
    def box_width(self) -> float: 
        return self.right_bottom.x - self.left_top.x

    @property
    def box_height(self) -> float:
        return self.right_bottom.y - self.left_top.y

    def box_area(self):
        return self.box_width * self.box_height
    
    @property
    def box_size(self):
        return self.box_width, self.box_height

    @property
    def image_width(self) -> float: 
        return self.left_top.image_width

    @property
    def image_height(self) -> float: 
        return self.left_top.image_height

    @property
    def image_size(self): 
        return self.left_top.image_size

    def resize_image(self, image_width: int, image_height: int, keep_absolute_x_y: bool = False):
        return ImageBoundingBox(self.left_top.resize_image(image_width, image_height, keep_absolute_x_y), 
                                self.right_bottom.resize_image(image_width, image_height, keep_absolute_x_y))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageBoundingBox):
            return False
        return self.left_top == other.left_top and self.right_bottom == other.right_bottom

    def __repr__(self) -> str:
        return (f"ImageBoundingBox(\n"
                f"    {textwrap.indent(repr(self.left_top), "    ")},\n"
                f"    {textwrap.indent(repr(self.right_bottom), "    ")}\n"
                f")")
    
    def plot_on(self, image: InMemoryImage, color: Literal["red"] = "red", width: int = 4) -> InMemoryImage:
        import PIL
        import PIL.Image
        import PIL.ImageDraw

        image_pillow: PIL.Image.Image = image.pillow
        if image_pillow.size != self.image_size:
            raise ValueError("The given image doesn't match the image size of the bounding box.")
        draw: PIL.ImageDraw.ImageDraw = PIL.ImageDraw.Draw(image_pillow)
        draw.rectangle((self.left_top.xy, self.right_bottom.xy), outline=color, width=width)
        return InMemoryImage.from_pillow(image_pillow)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {self.left_top.xy} -- {self.right_bottom.xy}>"
    
    @staticmethod
    @tomlkit.register_encoder
    def __tomlkit_register(obj: typing.Any, 
                           _parent: tomlkit.items.Item | None = None, 
                           _sort_keys: bool = False):
        if not isinstance(obj, ImageBoundingBox):
            raise tomlkit.exceptions.ConvertError(f"Unsupported type '{type(obj)}'.")
        table: tomlkit.items.InlineTable = tomlkit.inline_table()
        table.add("left_top", obj.left_top.xy)
        table.add("right_bottom", obj.right_bottom.xy)
        table.add("image_size", obj.image_size)
        return table
