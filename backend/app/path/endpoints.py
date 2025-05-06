from fastapi import APIRouter

from .schema import PathRequest, PathResponse
from .service import path_service

router = APIRouter(prefix="/path")


@router.post(
    "/",
    response_model=PathResponse,
    summary="Calculate path.",
    status_code=200,
)
async def calculate_path(request: PathRequest):
    return await path_service.calculate_path(request)
