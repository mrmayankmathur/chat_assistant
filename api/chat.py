from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status

from api.crm import crm_router, get_crm_webhook_service
from services.chat import ChatService
from services.chat_schemas import ChatRequest, Reply
from services.handoff import CrmWebhookService

router = APIRouter()


def get_chat_service() -> ChatService:
    raise NotImplementedError("ChatService dependency must be configured on the FastAPI app")


def get_principal_tenant_id(
    x_tenant_id: Annotated[int | None, Header(alias="X-Tenant-Id")] = None,
) -> int | None:
    return x_tenant_id


@router.post("/v1/chat", response_model=Reply)
async def post_v1_chat(
    payload: ChatRequest,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
    principal_tenant_id: Annotated[int | None, Depends(get_principal_tenant_id)] = None,
) -> Reply:
    if principal_tenant_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authenticated tenant_id",
        )
    if payload.tenant_id is not None and payload.tenant_id != principal_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request body tenant_id does not match authenticated principal",
        )
    return await chat_service.handle_chat(
        payload,
        principal_tenant_id=principal_tenant_id,
    )


def create_app(
    chat_service: ChatService | None = None,
    crm_webhook_service: CrmWebhookService | None = None,
    **kwargs: Any,
) -> FastAPI:
    app = FastAPI(title="WISMO Chat Assistant", **kwargs)
    app.include_router(router)
    app.include_router(crm_router)
    if chat_service is not None:
        app.dependency_overrides[get_chat_service] = lambda: chat_service
    if crm_webhook_service is not None:
        app.dependency_overrides[get_crm_webhook_service] = lambda: crm_webhook_service
    return app
