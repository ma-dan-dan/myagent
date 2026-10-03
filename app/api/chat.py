from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.agent.adapter import LLMServiceUnavailable
from app.intent.classifier import IntentClassificationError
from app.memory.context_manager import ContextBudgetExceeded
from app.schemas.chat import ChatRequest, ChatResponse, IntentMetrics
from app.services.chat_service import ChatService


router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    service: ChatService = http_request.app.state.chat_service
    try:
        return service.chat(request.user_id, request.session_id, request.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IntentClassificationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ContextBudgetExceeded as exc:
        raise HTTPException(status_code=413, detail="当前问题过长，无法安全构造上下文，请缩短当前问题后重试。") from exc
    except LLMServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="聊天服务暂时不可用，请稍后重试。") from exc


@router.get("/intent-metrics", response_model=IntentMetrics)
def intent_metrics(user_id: str, provider: str | None = None, http_request: Request = None) -> IntentMetrics:
    service: ChatService = http_request.app.state.chat_service
    try:
        return service.intent_metrics(user_id, provider)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
