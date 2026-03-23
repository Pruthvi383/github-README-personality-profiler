from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from services.profile_service import ProfileAnalysisError, analyze_profile


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        "index.html",
        {"request": request, "error": None, "username": ""},
    )


@router.post("/", response_class=HTMLResponse)
async def submit_home(request: Request, username: str = Form(...)) -> RedirectResponse:
    normalized = username.strip().lstrip("@")
    if not normalized:
        return request.app.state.templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please enter a GitHub username.", "username": ""},
            status_code=400,
        )
    destination = f"/profile/{quote(normalized)}"
    return RedirectResponse(url=destination, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/profile/{username}", response_class=HTMLResponse)
async def profile_page(request: Request, username: str) -> HTMLResponse:
    try:
        result = await analyze_profile(request.app, username)
    except ProfileAnalysisError as exc:
        return request.app.state.templates.TemplateResponse(
            "index.html",
            {"request": request, "error": exc.message, "username": username},
            status_code=exc.status_code,
        )

    context = {
        "result": result,
        "user": result["raw_data"]["user"],
        "classification": result["classification"],
        "features": result["features"],
        "stats": result["features"]["raw_features"],
        "percentiles": result["features"]["percentiles"],
    }
    context["request"] = request
    return request.app.state.templates.TemplateResponse("result.html", context)
