from app.api import router
from app.config import settings
from app.exceptions import Error, RequestValidationError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def create_app() -> FastAPI:
    app = FastAPI()
    configure_cors(app)
    add_exception_handlers(app)
    app.include_router(router)
    return app


async def exception_handler(request: Request, exc: Error) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": type(exc).__name__, "detail": exc.message},
        headers=exc.headers,
    )


def configure_cors(app: FastAPI) -> None:
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


def add_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Error, exception_handler)
    app.add_exception_handler(RequestValidationError, exception_handler)


app: FastAPI = create_app()
