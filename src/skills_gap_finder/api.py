"""REST API. Run with: uv run skills-gap serve, then open http://127.0.0.1:8000/docs"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import rag
from .service import SkillsService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = SkillsService()  # loads the model and index once, not per request
    yield


app = FastAPI(
    title="Skills Gap Finder",
    description="Semantic search over ESCO skills, skills-gap analysis and RAG development plans.",
    lifespan=lifespan,
)


def get_service(request: Request) -> SkillsService:
    return request.app.state.service


Service = Annotated[SkillsService, Depends(get_service)]


class GapRequest(BaseModel):
    role: str = Field(examples=["data scientist"])
    statements: list[str] = Field(
        min_length=1,
        max_length=30,
        description="Short sentences describing the person's work",
        examples=[["I write SQL queries to pull sales data", "I build Power BI dashboards for executives"]],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/roles")
def roles(service: Service) -> list[str]:
    return service.roles()


@app.get("/search")
def search(
    service: Service,
    q: str = Query(min_length=2, max_length=500, description="What the person does, in plain words"),
    k: int = Query(5, ge=1, le=50),
) -> list[dict]:
    return service.search(q, k)


@app.post("/gap")
def gap_analysis(body: GapRequest, service: Service) -> dict:
    try:
        return service.gap_analysis(body.statements, body.role)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=error.args[0]) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/plan")
def development_plan(body: GapRequest, service: Service) -> dict:
    analysis = gap_analysis(body, service)
    context = rag.build_context(analysis, service.describe)
    return {"essential_coverage": analysis["essential_coverage"], **rag.generate_plan(context)}
