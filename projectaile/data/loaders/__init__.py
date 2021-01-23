from .image import IMAGE_LOADER
from .audio import AUDIO_LOADER
from .text import TEXT_LOADER
from .structured import STRUCTURED_LOADER
from .loader import LOADER

loaders = {
    'image': IMAGE_LOADER,
    'audio': AUDIO_LOADER,
    'text': TEXT_LOADER,
    'structured': STRUCTURED_LOADER
}