from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from services.profile_service import ProfileAnalysisError, analyze_profile, ensure_profile_card


router = APIRouter(tags=["api"])


class AnalyzeRequest(BaseModel):
    username: str = Field(..., min_length=1, description="GitHub username to analyze")


@router.post("/analyze")
async def analyze(request: Request, payload: AnalyzeRequest) -> JSONResponse:
    try:
        result = await analyze_profile(request.app, payload.username)
    except ProfileAnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return JSONResponse(result)


@router.get("/api/profile/{username}")
async def raw_profile(request: Request, username: str) -> JSONResponse:
    try:
        result = await analyze_profile(request.app, username)
    except ProfileAnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return JSONResponse(result["raw_data"])


@router.get("/card/{username}")
async def profile_card(request: Request, username: str) -> FileResponse:
    try:
        result = await ensure_profile_card(request.app, username)
    except ProfileAnalysisError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    card_path = Path(result["card_path"])
    if not card_path.exists():
        raise HTTPException(status_code=404, detail="Profile card has not been generated yet.")

    return FileResponse(card_path, media_type="image/png", filename=f"{result['username']}_card.png")


@router.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": bool(request.app.state.model_bundle),
    }
