import typing

import tomlkit
import tomlkit.exceptions
import tomlkit.items


class ImagePoint:
    def __init__(self, x: float, y: float, image_width: int, image_height: int) -> None:
        self.__x = x
        self.__y = y
        self.__image_width = image_width
        self.__image_height = image_height
    
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def xy(self):
        return self.x, self.y

    @property
    def x_int(self):
        return round(self.x)

    @property
    def y_int(self):
        return round(self.y)

    @property
    def xy_int(self):
        return self.x_int, self.y_int

    @property
    def image_width(self):
        return self.__image_width
    
    @property
    def image_height(self):
        return self.__image_height

    @property
    def image_size(self):
        return self.image_width, self.image_height

    @property
    def x_relative(self):
        return self.x / self.image_width
    
    @property
    def y_relative(self):
        return self.y / self.image_height
    
    @property
    def xy_relative(self):
        return self.x_relative, self.y_relative

    def resize_image(self, image_width: int, image_height: int, keep_absolute_x_y: bool = False):
        if keep_absolute_x_y:
            return ImagePoint(self.x, self.y, image_width, image_height)

        return ImagePoint(self.x * image_width / self.image_width, 
                          self.y * image_height / self.image_height, 
                          image_width, image_height)
    
    def create_point_on_same_image(self, x: float, y: float):
        return ImagePoint(x, y, self.image_width, self.image_height)

    def have_same_image_size(self, other: 'ImagePoint'):
        return (self.image_width == other.image_width and 
                self.image_height == other.image_height)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImagePoint):
            return False
        if not self.have_same_image_size(other):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.image_width, self.image_height))

    def __repr__(self) -> str:
        return (f"ImagePoint(\n"
                f"    x={self.x}, y={self.y},\n"
                f"    image_width={self.image_width}, image_height={self.image_height}\n"
                f")")
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {self.xy}>"

    @staticmethod
    @tomlkit.register_encoder
    def __tomlkit_register(obj: typing.Any, 
                           _parent: tomlkit.items.Item | None = None, 
                           _sort_keys: bool = False):
        if not isinstance(obj, ImagePoint):
            raise tomlkit.exceptions.ConvertError(f"Unsupported type '{type(obj)}'.")
        table: tomlkit.items.InlineTable = tomlkit.inline_table()
        table.add("xy", obj.xy)
        table.add("image_size", obj.image_size)
        return table
