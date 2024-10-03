import functools
import hashlib
from typing import TypeVar

from pydantic import BaseModel as OriginalBaseModel
from wirerope.rope import RopeCore

T = TypeVar("T")


class BaseModel(OriginalBaseModel):
    """A pydantic BaseModel with extra functionalities."""

    class Config:
        extra = "forbid"
        ignored_types = (
            functools.cached_property,
            functools._lru_cache_wrapper,
            RopeCore,
        )

    def __hash__(self) -> int:
        # Generate SHA1 hash
        sha1_hash = hashlib.sha1(self.model_dump_json().encode())
        # Convert the first 8 bytes of the hash to an integer
        return int.from_bytes(sha1_hash.digest()[:8], byteorder="big")
