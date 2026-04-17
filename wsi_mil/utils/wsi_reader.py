from pathlib import Path

from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None

try:
    from tiffslide import TiffSlide
except ImportError:
    TiffSlide = None


class OpenSlideWrapper:
    def __init__(self, slide):
        self.slide = slide

    @property
    def dimensions(self):
        return self.slide.dimensions

    def get_thumbnail(self, size):
        return self.slide.get_thumbnail(size)

    def read_region(self, location, level, size):
        return self.slide.read_region(location, level, size)

    def close(self):
        self.slide.close()


class TiffSlideWrapper:
    def __init__(self, slide):
        self.slide = slide

    @property
    def dimensions(self):
        return self.slide.dimensions

    def get_thumbnail(self, size):
        return self.slide.get_thumbnail(size)

    def read_region(self, location, level, size):
        return self.slide.read_region(location, level, size)

    def close(self):
        self.slide.close()


def open_wsi(path: str):
    p = str(path)
    suffix = Path(path).suffix.lower()

    # 1) tif/tiff 优先 tiffslide
    if suffix in {".tif", ".tiff"}:
        if TiffSlide is not None:
            return TiffSlideWrapper(TiffSlide(p))
        if openslide is not None:
            return OpenSlideWrapper(openslide.OpenSlide(p))
        raise ImportError(
            "打开 tif/tiff 需要 tiffslide 或 openslide。建议安装: pip install tiffslide imagecodecs"
        )

    # 2) svs / ndpi / mrxs 等优先 openslide
    if openslide is not None:
        try:
            return OpenSlideWrapper(openslide.OpenSlide(p))
        except Exception:
            pass

    # 3) 兜底尝试 tiffslide
    if TiffSlide is not None:
        try:
            return TiffSlideWrapper(TiffSlide(p))
        except Exception:
            pass

    raise RuntimeError(f"无法打开 WSI 文件: {path}")