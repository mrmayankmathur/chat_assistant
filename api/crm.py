from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from services.handoff import CrmWebhookService

crm_router = APIRouter()


def get_crm_webhook_service() -> CrmWebhookService:
    raise NotImplementedError("CrmWebhookService dependency must be configured on the FastAPI app")


@crm_router.post("/v1/crm/{webhook_uuid}")
async def post_v1_crm_webhook(
    webhook_uuid: uuid.UUID,
    request: Request,
    crm_webhook_service: Annotated[CrmWebhookService, Depends(get_crm_webhook_service)],
    x_crm_signature: Annotated[str | None, Header(alias="X-Crm-Signature")] = None,
    x_hub_signature_256: Annotated[str | None, Header(alias="X-Hub-Signature-256")] = None,
    x_signature: Annotated[str | None, Header(alias="X-Signature")] = None,
) -> dict[str, Any]:
    raw_body = await request.body()
    signature = x_crm_signature or x_hub_signature_256 or x_signature

    try:
        return crm_webhook_service.handle_webhook(
            webhook_uuid=webhook_uuid,
            raw_body=raw_body,
            signature=signature,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
