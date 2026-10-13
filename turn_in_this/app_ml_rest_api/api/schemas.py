from typing import List, Literal

from pydantic import BaseModel, Field

YesNo = Literal["yes", "no"]


class HouseFeatures(BaseModel):
    area: int = Field(..., gt=0, description="Площа, кв. фути", examples=[7420])
    bedrooms: int = Field(..., ge=0, le=20, examples=[4])
    bathrooms: int = Field(..., ge=0, le=20, examples=[2])
    stories: int = Field(..., ge=1, le=10, examples=[3])
    mainroad: YesNo = Field(..., description="Біля головної дороги")
    guestroom: YesNo
    basement: YesNo
    hotwaterheating: YesNo
    airconditioning: YesNo
    parking: int = Field(..., ge=0, le=10, description="Кількість паркомісць")
    prefarea: YesNo = Field(..., description="Престижний район")
    furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]


class PredictionResponse(BaseModel):
    predicted_price: float


class BatchRequest(BaseModel):
    items: List[HouseFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: List[float]
