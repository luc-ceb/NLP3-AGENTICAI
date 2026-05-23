from typing import Optional
from pydantic import BaseModel, Field


class Titles(BaseModel):
    canonical: str = Field(..., description="The DB key (may have underscores instead of spaces)")
    normalized: str = Field(..., description="The normalized title, may have spaces instead of underscores")
    display: str = Field(..., description="The title as it should be displayed to the user")


class Thumbnail(BaseModel):
    source: str = Field(..., description="Thumbnail image URI")
    width: int = Field(..., description="Thumbnail width")
    height: int = Field(..., description="Thumbnail height")


class OriginalImage(BaseModel):
    source: str = Field(..., description="Original image URI")
    width: int = Field(..., description="Original image width")
    height: int = Field(..., description="Original image height")


class Coordinates(BaseModel):
    lat: float = Field(..., description="The latitude")
    lon: float = Field(..., description="The longitude")


class Summary(BaseModel):
    titles: Titles

    # Deprecated fields
    title: Optional[str] = Field(None, description="Deprecated: Use titles.normalized instead")
    displaytitle: Optional[str] = Field(None, description="Deprecated: Use titles.display instead")

    pageid: Optional[int] = Field(None, description="The page ID")
    extract: str = Field(..., description="First several sentences of an article in plain text")
    extract_html: Optional[str] = Field(None, description="First several sentences of an article in HTML")
    thumbnail: Optional[Thumbnail] = None
    originalimage: Optional[OriginalImage] = None
    lang: str = Field(..., description="The page language code, e.g., en")
    dir: str = Field(..., description="The page language direction code, e.g., ltr")
    timestamp: Optional[str] = Field(None, description="ISO 8601 last edit time")
    description: Optional[str] = Field(None, description="Wikidata description for the page")
    coordinates: Optional[Coordinates] = None
