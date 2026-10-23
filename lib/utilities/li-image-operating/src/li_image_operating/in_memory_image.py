import io
from pathlib import Path
import typing

from mime_bytes import MimeBytes

try:
    import PIL.Image
    import numpy
    import matplotlib.figure
except ImportError:
    pass


class InMemoryImage:
    def __init__(self, image: MimeBytes) -> None:
        super().__init__()
        self.__mime_bytes = image
        self.__pillow: 'PIL.Image.ImageFile.ImageFile | None' = None
        self.__cv: 'numpy.ndarray | None' = None
        self.__base64: str | None = None
    
    @property
    def mime_bytes(self):
        return self.__mime_bytes
    
    @property
    def pillow(self):
        import PIL
        import PIL.Image
        
        if self.__pillow is None:
            b: io.BytesIO
            with self.mime_bytes.to_bytes_io() as b:
                self.__pillow = PIL.Image.open(b)
                self.__pillow.load()
        return self.__pillow.copy()

    @property
    def cv_bgr(self):
        import numpy
        import cv2
        from cv2.typing import MatLike

        if self.__cv is None:
            result: MatLike | None = cv2.imdecode(numpy.frombuffer(self.mime_bytes.bytes_, numpy.uint8), 
                                                  cv2.IMREAD_COLOR)
            if result is None:
                raise RuntimeError("Failed to decode the image block as a cv2 image. Is it an image file?")
            
            self.__cv = typing.cast(numpy.ndarray, result)
            self.__cv.flags.writeable = False
        return self.__cv

    @property
    def base64(self):
        if self.__base64 is None:
            self.__base64 = self.mime_bytes.to_base64()
        return self.__base64

    @property
    def data_url(self):
        return f"data:{self.mime_bytes.mime};base64,{self.base64}"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, InMemoryImage):
            return False
        return self.mime_bytes.__eq__(value.mime_bytes)

    def __hash__(self) -> int:
        return self.mime_bytes.__hash__()

    @staticmethod
    def from_base64(base64_: str | bytes, mime: str = "application/octet-stream"):
        result: InMemoryImage = InMemoryImage(MimeBytes.from_base64(base64_, mime))
        if isinstance(base64_, str):
            result.__base64 = base64_
        return result
    
    @staticmethod
    def from_data_url(url: str):
        return InMemoryImage(MimeBytes.from_data_url(url))
    
    @staticmethod
    def from_bytes(bytes_: bytes, mime: str = "application/octet-stream"):
        return InMemoryImage(MimeBytes(mime, bytes_))

    @staticmethod
    def from_buffered_io(buffered_io: io.BufferedIOBase, mime: str = "application/octet-stream"):
        return InMemoryImage(MimeBytes.from_buffered_io(buffered_io, mime))

    @staticmethod
    def from_file(path: Path | str, 
                  mime_force: str | None = None, 
                  mime_default: str = "application/octet-stream"):
        return InMemoryImage(MimeBytes.from_file(path, mime_force, mime_default))
    
    @staticmethod
    def from_pillow(image: 'PIL.Image.Image'):
        b: io.BytesIO
        with io.BytesIO() as b:
            image.save(b, "png")
            return InMemoryImage(MimeBytes("image/png", b.getvalue()))
    
    @staticmethod
    def from_matplotlib(image: 'matplotlib.figure.Figure'):
        b: io.BytesIO
        with io.BytesIO() as b:
            image.savefig(b, format="png")
            return InMemoryImage(MimeBytes("image/png", b.getvalue()))
    
    @staticmethod
    def from_cv_bgr(array: 'numpy.ndarray'):
        import cv2

        success: bool
        success, array = cv2.imencode(".png", array)
        if not success:
            raise RuntimeError("Failed to encode the array to a image file. Is it a cv2 image array?")
        result: InMemoryImage = InMemoryImage(MimeBytes("image/png", array.tobytes()))
        result.__cv = array.copy()
        result.__cv.flags.writeable = False
        return result
    