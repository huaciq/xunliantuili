from fastapi import APIRouter

from app.api import auth, health, projects, registry, resources, runs, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(projects.router)
api_router.include_router(resources.router)
api_router.include_router(runs.router)
api_router.include_router(registry.router)
