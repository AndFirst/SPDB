from app.path.endpoints import router as path_router
from fastapi.routing import APIRouter

router = APIRouter(prefix="/v1")

# /path
router.include_router(path_router)
