from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from api.chat import create_app, get_chat_service
from app.core import hash_phone_e164, normalize_email, normalize_phone_e164
from app.models import (
    AgentConfig,
    BusinessHours,
    CustomerOrder,
    FailureReason,
    MessageRole,
    Policy,
    PolicyType,
    Shipment,
    ShipmentEvent,
    ShipmentStatus,
    Tenant,
    Tonality,
)
from app.seed import (
    CUSTOMER_A_EMAIL,
    CUSTOMER_A_PHONE_E164,
    CUSTOMER_B_AWB,
    CUSTOMER_B_EMAIL,
    CUSTOMER_B_PHONE_E164,
    TENANT_ID,
)
from app.settings import Settings
from evals.agent_simulator import LiveGpt6LunaClient, LocalEvalAnthropicClient
from repositories import (
    InMemoryChatRepository,
    InMemoryOrderRepository,
    InMemoryThreadRepository,
    clear_tenant_config_cache,
)
from services import (
    ChatRequest,
    ChatService,
    IdentityService,
    OrderService,
    Reply,
    TrackingService,
    is_within_business_hours,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

SANDBOX_SETTINGS = Settings(
    phone_hmac_secret=os.getenv("PHONE_HMAC_SECRET", "wismo-dev-hmac-secret-key"),
    default_phone_region=os.getenv("DEFAULT_PHONE_REGION", "US"),
    llm_model="gpt-6-luna",
)

# Monday 14:00 UTC (10:00 AM America/New_York -> Inside 09:00-18:00 business hours)
INSIDE_HOURS_NOW = datetime(2026, 10, 19, 14, 0, 0, tzinfo=UTC)
# Sunday 03:00 UTC (Saturday 23:00 PM America/New_York -> Outside business hours)
OUTSIDE_HOURS_NOW = datetime(2026, 10, 18, 3, 0, 0, tzinfo=UTC)


class BusinessHoursToggleRequest(BaseModel):
    inside_business_hours: bool


class CreateDemoOrderRequest(BaseModel):
    persona_id: str = "alice"
    sku: str = "SERUM-VITC"
    name: str = "Vitamin C Glow Serum"
    price_cents: int = 3800
    payment_mode: str = "PREPAID"


STOREFRONT_CATALOG: list[dict[str, Any]] = [
    {
        "sku": "SERUM-VITC",
        "name": "Vitamin C Glow Serum",
        "subtitle": "15% L-Ascorbic Acid + Ferulic + Kakadu Plum",
        "price_cents": 3800,
        "price_display": "$38.00",
        "badge": "Bestseller",
        "rating": "4.9 (1,420)",
        "accent": "#d97706",
    },
    {
        "sku": "CREAM-HYDRA",
        "name": "Deep Hydra Barrier Cream",
        "subtitle": "5-Ceramide Complex + Squalane + Ectoin",
        "price_cents": 4200,
        "price_display": "$42.00",
        "badge": "Clinically Tested",
        "rating": "4.8 (980)",
        "accent": "#0284c7",
    },
    {
        "sku": "CLEANSER-GENTLE",
        "name": "Oat Milk Gentle Cleanser",
        "subtitle": "Colloidal Oatmeal + Panthenol B5",
        "price_cents": 2600,
        "price_display": "$26.00",
        "badge": "Sensitive Skin",
        "rating": "4.9 (740)",
        "accent": "#0d9488",
    },
    {
        "sku": "MASK-CLAY",
        "name": "Pink Kaolin Detox Mask",
        "subtitle": "Australian Pink Clay + Niacinamide + Willow Bark",
        "price_cents": 3200,
        "price_display": "$32.00",
        "badge": "Pore Refining",
        "rating": "4.7 (530)",
        "accent": "#e11d48",
    },
    {
        "sku": "OIL-ROSEHIP",
        "name": "Cold-Pressed Rosehip Oil",
        "subtitle": "100% Organic Rosa Canina Seed Oil",
        "price_cents": 4500,
        "price_display": "$45.00",
        "badge": "Organic",
        "rating": "4.9 (1,110)",
        "accent": "#9333ea",
    },
]


class InteractiveSandbox:
    """Self-contained interactive WISMO playground with live state & telemetry inspection."""

    def __init__(self) -> None:
        self.inside_business_hours: bool = True
        self.llm_mode: str = "simulator"
        self._next_demo_order_seq: int = 1007
        self.thread_repo: InMemoryThreadRepository
        self.order_repo: InMemoryOrderRepository
        self.chat_repo: InMemoryChatRepository
        self.chat_service: ChatService
        self.reset()

    @property
    def current_now(self) -> datetime:
        return INSIDE_HOURS_NOW if self.inside_business_hours else OUTSIDE_HOURS_NOW

    def _build_llm_client(self) -> Any:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        live_env = os.getenv("WISMO_LIVE_LLM", "1").strip().lower()
        use_live = live_env not in ("0", "false", "no")
        in_pytest = bool(os.getenv("PYTEST_CURRENT_TEST"))
        if openai_key and use_live and not in_pytest:
            self.llm_mode = "GPT 6 Luna · Live (gpt-6-luna)"
            self.llm_execution_mode = "live"
            return LiveGpt6LunaClient(
                api_key=openai_key,
                base_url=SANDBOX_SETTINGS.openai_base_url,
            )
        self.llm_mode = "GPT 6 Luna (gpt-6-luna)"
        self.llm_execution_mode = "simulator"
        return LocalEvalAnthropicClient()

    def reset(self) -> None:
        clear_tenant_config_cache()
        self._next_demo_order_seq = 1007
        now = INSIDE_HOURS_NOW
        today = now.date()

        alice_phone = normalize_phone_e164(
            CUSTOMER_A_PHONE_E164, default_region=SANDBOX_SETTINGS.default_phone_region
        )
        bob_phone = normalize_phone_e164(
            CUSTOMER_B_PHONE_E164, default_region=SANDBOX_SETTINGS.default_phone_region
        )
        alice_hash = hash_phone_e164(alice_phone, secret=SANDBOX_SETTINGS.phone_hmac_secret)
        bob_hash = hash_phone_e164(bob_phone, secret=SANDBOX_SETTINGS.phone_hmac_secret)

        tenant = Tenant(
            tenant_id=TENANT_ID,
            name="Lumina Botanics",
            timezone="America/New_York",
            features={
                "whatsapp_enabled": True,
                "web_chat_enabled": True,
                "cancel_enabled": True,
                "cancel_before_pickup": True,
            },
        )
        agent_cfg = AgentConfig(
            agent_id=1,
            tenant_id=TENANT_ID,
            brand_prompt="You are the warm, clear support assistant for Lumina Botanics.",
            tonality=Tonality.FRIENDLY,
            is_active=True,
            version=1,
            soft_deleted=False,
        )
        policies = [
            Policy(
                policy_id=1,
                tenant_id=TENANT_ID,
                policy_type=PolicyType.CANCELLATION,
                text="Orders can only be cancelled before carrier pickup.",
                windows={
                    "allow_statuses": [
                        int(ShipmentStatus.ORDER_PLACED),
                        int(ShipmentStatus.MANIFESTED),
                        int(ShipmentStatus.PICKUP_FAILED),
                    ]
                },
                soft_deleted=False,
            ),
            Policy(
                policy_id=2,
                tenant_id=TENANT_ID,
                policy_type=PolicyType.REFUND,
                text="Refunds to original payment method take 5-7 business days after cancellation.",
                windows={"refund_sla_business_days": 7},
                soft_deleted=False,
            ),
        ]
        bh = BusinessHours(
            tenant_id=TENANT_ID,
            weekly={
                "mon": [{"open": "09:00", "close": "18:00"}],
                "tue": [{"open": "09:00", "close": "18:00"}],
                "wed": [{"open": "09:00", "close": "18:00"}],
                "thu": [{"open": "09:00", "close": "18:00"}],
                "fri": [{"open": "09:00", "close": "18:00"}],
            },
            overrides=[],
            soft_deleted=False,
        )

        orders = [
            # Alice Order 1: ORDER_PLACED (Eligible for 2-step pre-pickup cancellation)
            CustomerOrder(
                order_pk=1001,
                tenant_id=TENANT_ID,
                order_ref="LUM-1001",
                phone_hash=alice_hash,
                email=CUSTOMER_A_EMAIL,
                payment_mode="PREPAID",
                placed_at=now - timedelta(hours=6),
                cancelled_at=None,
                items=[
                    {
                        "sku": "SERUM-VITC",
                        "name": "Vitamin C Glow Serum",
                        "qty": 1,
                        "price_cents": 3800,
                    }
                ],
                soft_deleted=False,
            ),
            # Alice Order 2: IN_TRANSIT with EDD breached (2 days overdue -> proactive apology)
            CustomerOrder(
                order_pk=1002,
                tenant_id=TENANT_ID,
                order_ref="LUM-1002",
                phone_hash=alice_hash,
                email=CUSTOMER_A_EMAIL,
                payment_mode="PREPAID",
                placed_at=now - timedelta(days=6),
                cancelled_at=None,
                items=[
                    {
                        "sku": "CREAM-HYDRA",
                        "name": "Deep Hydra Barrier Cream",
                        "qty": 2,
                        "price_cents": 8400,
                    }
                ],
                soft_deleted=False,
            ),
            # Alice Order 3: OUT_FOR_DELIVERY (Post-pickup -> cancellation refused)
            CustomerOrder(
                order_pk=1003,
                tenant_id=TENANT_ID,
                order_ref="LUM-1003",
                phone_hash=alice_hash,
                email=CUSTOMER_A_EMAIL,
                payment_mode="COD",
                placed_at=now - timedelta(days=3),
                cancelled_at=None,
                items=[
                    {
                        "sku": "CLEANSER-GENTLE",
                        "name": "Oat Milk Gentle Cleanser",
                        "qty": 1,
                        "price_cents": 2600,
                    }
                ],
                soft_deleted=False,
            ),
            # Alice Order 4: FAILED_DELIVERY (Customer Unavailable)
            CustomerOrder(
                order_pk=1004,
                tenant_id=TENANT_ID,
                order_ref="LUM-1004",
                phone_hash=alice_hash,
                email=CUSTOMER_A_EMAIL,
                payment_mode="PREPAID",
                placed_at=now - timedelta(days=4),
                cancelled_at=None,
                items=[
                    {
                        "sku": "MASK-CLAY",
                        "name": "Pink Kaolin Detox Mask",
                        "qty": 1,
                        "price_cents": 3200,
                    }
                ],
                soft_deleted=False,
            ),
            # Alice Order 5: CANCELLED COD Order (Refund not applicable for COD)
            CustomerOrder(
                order_pk=1005,
                tenant_id=TENANT_ID,
                order_ref="LUM-1005",
                phone_hash=alice_hash,
                email=CUSTOMER_A_EMAIL,
                payment_mode="COD",
                placed_at=now - timedelta(days=5),
                cancelled_at=now - timedelta(days=5) + timedelta(hours=2),
                items=[
                    {
                        "sku": "TONER-PHA",
                        "name": "PHA Pore Refining Toner",
                        "qty": 1,
                        "price_cents": 2900,
                    }
                ],
                soft_deleted=False,
            ),
            # Bob Order 6: DELIVERED (Foreign order to Alice -> AWB-CUSTB-9001 / LUM-1006)
            CustomerOrder(
                order_pk=1006,
                tenant_id=TENANT_ID,
                order_ref="LUM-1006",
                phone_hash=bob_hash,
                email=CUSTOMER_B_EMAIL,
                payment_mode="PREPAID",
                placed_at=now - timedelta(days=7),
                cancelled_at=None,
                items=[
                    {
                        "sku": "OIL-ROSEHIP",
                        "name": "Cold-Pressed Rosehip Oil",
                        "qty": 1,
                        "price_cents": 4500,
                    }
                ],
                soft_deleted=False,
            ),
        ]

        shipments = [
            Shipment(
                id=2001,
                order_pk=1001,
                awb="AWB-CUSTA-1001",
                carrier_code="DELHIVERY",
                is_reverse=False,
                status=ShipmentStatus.ORDER_PLACED,
                edd=today + timedelta(days=4),
                picked_up_at=None,
                delivered_at=None,
            ),
            Shipment(
                id=2002,
                order_pk=1002,
                awb="AWB-CUSTA-1002",
                carrier_code="BLUEDART",
                is_reverse=False,
                status=ShipmentStatus.IN_TRANSIT,
                edd=today - timedelta(days=2),
                picked_up_at=now - timedelta(days=5),
                delivered_at=None,
            ),
            Shipment(
                id=2003,
                order_pk=1003,
                awb="AWB-CUSTA-1003",
                carrier_code="DELHIVERY",
                is_reverse=False,
                status=ShipmentStatus.OUT_FOR_DELIVERY,
                edd=today,
                picked_up_at=now - timedelta(days=2),
                delivered_at=None,
            ),
            Shipment(
                id=2004,
                order_pk=1004,
                awb="AWB-CUSTA-1004",
                carrier_code="BLUEDART",
                is_reverse=False,
                status=ShipmentStatus.FAILED_DELIVERY,
                edd=today,
                picked_up_at=now - timedelta(days=3),
                delivered_at=None,
            ),
            Shipment(
                id=2005,
                order_pk=1005,
                awb="AWB-CUSTA-1005",
                carrier_code="DELHIVERY",
                is_reverse=False,
                status=ShipmentStatus.CANCELLED,
                edd=None,
                picked_up_at=None,
                delivered_at=None,
            ),
            Shipment(
                id=2006,
                order_pk=1006,
                awb=CUSTOMER_B_AWB,
                carrier_code="BLUEDART",
                is_reverse=False,
                status=ShipmentStatus.DELIVERED,
                edd=today - timedelta(days=3),
                picked_up_at=now - timedelta(days=6),
                delivered_at=now - timedelta(days=3),
            ),
        ]

        events = [
            ShipmentEvent(
                shipment_id=2001,
                event_at=now - timedelta(hours=6),
                status=ShipmentStatus.ORDER_PLACED,
                failure_reason=FailureReason.NONE,
                location="Newark Warehouse",
                raw_remark="Order received at fulfillment center",
            ),
            ShipmentEvent(
                shipment_id=2002,
                event_at=now - timedelta(days=1),
                status=ShipmentStatus.IN_TRANSIT,
                failure_reason=FailureReason.NONE,
                location="Chicago Sort Facility",
                raw_remark="In transit to destination hub - weather delay",
            ),
            ShipmentEvent(
                shipment_id=2003,
                event_at=now - timedelta(hours=3),
                status=ShipmentStatus.OUT_FOR_DELIVERY,
                failure_reason=FailureReason.NONE,
                location="Brooklyn Local Hub",
                raw_remark="Out for delivery with courier",
            ),
            ShipmentEvent(
                shipment_id=2004,
                event_at=now - timedelta(hours=2),
                status=ShipmentStatus.FAILED_DELIVERY,
                failure_reason=FailureReason.CUSTOMER_UNAVAILABLE,
                location="Brooklyn, NY",
                raw_remark="Consignee unavailable; rang bell twice",
            ),
            ShipmentEvent(
                shipment_id=2005,
                event_at=now - timedelta(days=5) + timedelta(hours=2),
                status=ShipmentStatus.CANCELLED,
                failure_reason=FailureReason.NONE,
                location="Newark Warehouse",
                raw_remark="Cancelled before carrier pickup",
            ),
            ShipmentEvent(
                shipment_id=2006,
                event_at=now - timedelta(days=3),
                status=ShipmentStatus.DELIVERED,
                failure_reason=FailureReason.NONE,
                location="Boston, MA",
                raw_remark="Delivered to front desk",
            ),
        ]

        self.thread_repo = InMemoryThreadRepository()
        self.order_repo = InMemoryOrderRepository(orders=orders, shipments=shipments)
        self.chat_repo = InMemoryChatRepository(
            tenant=tenant,
            agent_config=agent_cfg,
            channel_guidelines={
                "whatsapp": "Keep replies concise and mobile-friendly for WhatsApp.",
                "web": "Provide structured order details clearly for web chat.",
            },
            policies=policies,
            business_hours=bh,
            shipments=shipments,
            events=events,
        )

        identity_service = IdentityService(
            repo_or_session=self.thread_repo, settings=SANDBOX_SETTINGS
        )
        order_service = OrderService(repo_or_session=self.order_repo)
        tracking_service = TrackingService()
        client = self._build_llm_client()

        self.chat_service = ChatService(
            identity_service=identity_service,
            order_service=order_service,
            tracking_service=tracking_service,
            chat_repo=self.chat_repo,
            anthropic_client=client,
            settings=SANDBOX_SETTINGS,
        )

    async def handle_sandbox_chat(
        self,
        payload: ChatRequest,
        *,
        principal_tenant_id: int,
    ) -> Reply:
        return await self.chat_service.handle_chat(
            payload,
            principal_tenant_id=principal_tenant_id,
            now=self.current_now,
        )

    def create_demo_order(self, req: CreateDemoOrderRequest) -> dict[str, Any]:
        now = self.current_now
        today = now.date()
        seq = self._next_demo_order_seq
        self._next_demo_order_seq += 1

        is_bob = req.persona_id.strip().lower() == "bob"
        raw_phone = CUSTOMER_B_PHONE_E164 if is_bob else CUSTOMER_A_PHONE_E164
        email = CUSTOMER_B_EMAIL if is_bob else CUSTOMER_A_EMAIL
        norm_phone = normalize_phone_e164(
            raw_phone, default_region=SANDBOX_SETTINGS.default_phone_region
        )
        phone_hash = hash_phone_e164(norm_phone, secret=SANDBOX_SETTINGS.phone_hmac_secret)

        order_ref = f"LUM-{seq}"
        awb = f"AWB-DEMO-{seq}"
        shipment_id = 1000 + seq

        new_order = CustomerOrder(
            order_pk=seq,
            tenant_id=TENANT_ID,
            order_ref=order_ref,
            phone_hash=phone_hash,
            email=email,
            payment_mode=req.payment_mode.upper(),
            placed_at=now - timedelta(minutes=2),
            cancelled_at=None,
            items=[
                {
                    "sku": req.sku,
                    "name": req.name,
                    "qty": 1,
                    "price_cents": req.price_cents,
                }
            ],
            soft_deleted=False,
        )
        new_shipment = Shipment(
            id=shipment_id,
            order_pk=seq,
            awb=awb,
            carrier_code="DELHIVERY",
            is_reverse=False,
            status=ShipmentStatus.ORDER_PLACED,
            edd=today + timedelta(days=3),
            picked_up_at=None,
            delivered_at=None,
        )
        new_event = ShipmentEvent(
            shipment_id=shipment_id,
            event_at=now - timedelta(minutes=1),
            status=ShipmentStatus.ORDER_PLACED,
            failure_reason=FailureReason.NONE,
            location="Newark Fulfillment Center",
            raw_remark="Order placed via Lumina Botanics online storefront",
        )

        self.order_repo._orders.insert(0, new_order)
        self.order_repo._shipments.append(new_shipment)
        self.chat_repo.shipments.append(new_shipment)
        self.chat_repo.events.append(new_event)

        return {
            "created_order_ref": order_ref,
            "awb": awb,
            "state": self.snapshot_state(),
        }

    def snapshot_state(self) -> dict[str, Any]:
        shipments_by_order: dict[int, Shipment] = {
            s.order_pk: s for s in self.chat_repo.shipments
        }
        events_by_shipment: dict[int, list[ShipmentEvent]] = {}
        for ev in self.chat_repo.events:
            events_by_shipment.setdefault(ev.shipment_id, []).append(ev)

        alice_hash = hash_phone_e164(
            normalize_phone_e164(CUSTOMER_A_PHONE_E164, default_region="US"),
            secret=SANDBOX_SETTINGS.phone_hmac_secret,
        )
        today = self.current_now.date()
        orders_view: list[dict[str, Any]] = []
        for o in self.order_repo._orders:
            sh = shipments_by_order.get(o.order_pk)
            status_name = ShipmentStatus(int(sh.status)).name if sh is not None else "UNKNOWN"
            is_alice = o.phone_hash == alice_hash
            owner_label = (
                "Alice (+14155550101)"
                if is_alice
                else "Bob (+14155550102)"
            )
            latest_ev = None
            if sh is not None and sh.id in events_by_shipment:
                latest_ev = sorted(
                    events_by_shipment[sh.id], key=lambda e: e.event_at
                )[-1]
            item = o.items[0] if o.items else {}
            price_cents = int(item["price_cents"]) if item.get("price_cents") is not None else None
            qty = int(item["qty"]) if item.get("qty") is not None else None
            sku = item.get("sku")
            edd_breached = bool(
                sh is not None
                and sh.edd is not None
                and status_name not in ("DELIVERED", "CANCELLED")
                and (today - sh.edd).days >= 2
            )
            can_cancel = status_name in ("ORDER_PLACED", "MANIFESTED", "PICKUP_FAILED")
            orders_view.append(
                {
                    "order_ref": o.order_ref,
                    "owner_id": "alice" if is_alice else "bob",
                    "owner": owner_label,
                    "payment_mode": o.payment_mode,
                    "sku": sku,
                    "item": item.get("name"),
                    "qty": qty,
                    "price_cents": price_cents,
                    "price_display": f"${price_cents / 100:.2f}" if price_cents is not None else None,
                    "awb": sh.awb if sh else None,
                    "carrier": sh.carrier_code if sh else None,
                    "status": status_name,
                    "edd": sh.edd.isoformat() if (sh and sh.edd) else None,
                    "edd_breached": edd_breached,
                    "can_cancel": can_cancel,
                    "placed_at": o.placed_at.isoformat() if o.placed_at else None,
                    "cancelled_at": o.cancelled_at.isoformat() if o.cancelled_at else None,
                    "last_event_at": latest_ev.event_at.isoformat() if latest_ev else None,
                    "last_location": latest_ev.location if latest_ev else None,
                    "last_remark": latest_ev.raw_remark if latest_ev else None,
                }
            )

        llm_calls_view: list[dict[str, Any]] = []
        for idx, c in enumerate(self.chat_repo.llm_calls, start=1):
            llm_calls_view.append(
                {
                    "id": idx,
                    "message_id": c.message_id,
                    "step": c.step,
                    "model": c.model,
                    "stop_reason": c.stop_reason,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cache_read_tokens": c.cache_read_tokens,
                    "cache_write_tokens": c.cache_write_tokens,
                    "latency_ms": c.latency_ms,
                    "tool_calls": c.tool_calls or [],
                }
            )

        outbox_view = [
            {
                "event_type": entry.event_type,
                "tenant_id": entry.tenant_id,
                "thread_id": entry.thread_id,
                "payload": entry.payload,
                "created_at": entry.created_at.isoformat(),
            }
            for entry in self.chat_repo.outbox
        ]

        messages_view = [
            {
                "message_id": m.message_id,
                "thread_id": m.thread_id,
                "channel": m.channel,
                "role": "customer" if int(m.role) == int(MessageRole.CUSTOMER) else "assistant",
                "content": m.content,
                "payload": m.payload,
            }
            for m in self.chat_repo.messages
        ]

        is_bh = is_within_business_hours(
            self.chat_repo.tenant,
            self.chat_repo.business_hours,
            now=self.current_now,
        )

        return {
            "tenant": {
                "tenant_id": self.chat_repo.tenant.tenant_id,
                "name": self.chat_repo.tenant.name,
                "timezone": self.chat_repo.tenant.timezone,
                "current_simulated_time": self.current_now.isoformat(),
                "is_business_hours": is_bh,
                "llm_mode": self.llm_mode,
                "llm_execution_mode": self.llm_execution_mode,
                "configured_model": SANDBOX_SETTINGS.llm_model,
                "telemetry_is_estimated": self.llm_execution_mode == "simulator",
                "data_mode": "shared_in_memory_sandbox",
            },
            "personas": [
                {
                    "id": "alice",
                    "label": "Alice Customer (5 orders: Placed, Delayed, OFD, Failed, Cancelled COD)",
                    "phone": CUSTOMER_A_PHONE_E164,
                    "email": CUSTOMER_A_EMAIL,
                },
                {
                    "id": "bob",
                    "label": "Bob Buyer (1 order: LUM-1006 / AWB-CUSTB-9001 Delivered)",
                    "phone": CUSTOMER_B_PHONE_E164,
                    "email": CUSTOMER_B_EMAIL,
                },
            ],
            "catalog": STOREFRONT_CATALOG,
            "orders": orders_view,
            "llm_calls": llm_calls_view,
            "outbox": outbox_view,
            "messages": messages_view,
        }


class _SandboxChatServiceProxy:
    """Routes FastAPI `/v1/chat` calls through the active `InteractiveSandbox`."""

    def __init__(self, sandbox: InteractiveSandbox) -> None:
        self._sandbox = sandbox

    async def handle_chat(
        self,
        request: ChatRequest,
        *,
        principal_tenant_id: int | None = None,
        now: datetime | None = None,
    ) -> Reply:
        return await self._sandbox.chat_service.handle_chat(
            request,
            principal_tenant_id=principal_tenant_id,
            now=now or self._sandbox.current_now,
        )


def build_playground_app() -> tuple[FastAPI, InteractiveSandbox]:
    sandbox = InteractiveSandbox()
    proxy_service = _SandboxChatServiceProxy(sandbox)
    application = create_app(chat_service=proxy_service)  # type: ignore[arg-type]

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/v1/sandbox/state")
    async def get_sandbox_state() -> dict[str, Any]:
        return sandbox.snapshot_state()

    @application.post("/v1/sandbox/reset")
    async def post_sandbox_reset() -> dict[str, Any]:
        sandbox.reset()
        return sandbox.snapshot_state()

    @application.post("/v1/sandbox/business-hours")
    async def post_sandbox_business_hours(
        body: BusinessHoursToggleRequest,
    ) -> dict[str, Any]:
        sandbox.inside_business_hours = body.inside_business_hours
        return sandbox.snapshot_state()

    @application.post("/v1/sandbox/create-order")
    async def post_sandbox_create_order(
        body: CreateDemoOrderRequest,
    ) -> dict[str, Any]:
        return sandbox.create_demo_order(body)

    return application, sandbox


app, sandbox = build_playground_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=False)
