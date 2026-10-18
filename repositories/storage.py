from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import (
    CustomerOrder,
    FailureReason,
    Shipment,
    ShipmentStatus,
    StatusCopy,
    Thread,
)


@runtime_checkable
class ThreadRepositoryProtocol(Protocol):
    def find_by_email(self, tenant_id: int, email: str) -> Thread | None: ...

    def find_by_phone_hash(self, tenant_id: int, phone_hash: str) -> Thread | None: ...

    def create(
        self,
        *,
        tenant_id: int,
        phone_hash: str | None,
        email: str | None,
        expires_at: datetime,
        migrated_from_thread_id: int | None = None,
    ) -> Thread: ...

    def save(self, thread: Thread) -> Thread: ...


class SqlAlchemyThreadRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def find_by_email(self, tenant_id: int, email: str) -> Thread | None:
        normalized = email.strip().lower()
        if not normalized:
            return None
        stmt = (
            select(Thread)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                Thread.email == normalized,
            )
            .order_by(Thread.expires_at.desc(), Thread.thread_id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def find_by_phone_hash(self, tenant_id: int, phone_hash: str) -> Thread | None:
        cleaned = phone_hash.strip()
        if not cleaned:
            return None
        stmt = (
            select(Thread)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                Thread.phone_hash == cleaned,
            )
            .order_by(Thread.expires_at.desc(), Thread.thread_id.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)

    def is_phone_claimed_by_other_identity(
        self,
        tenant_id: int,
        phone_hash: str,
        *,
        current_thread_id: int | None,
        current_email: str | None,
    ) -> bool:
        other_thread = self.session.scalar(
            select(Thread)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                Thread.phone_hash == phone_hash,
                *(
                    [Thread.thread_id != current_thread_id]
                    if current_thread_id is not None
                    else []
                ),
            )
            .limit(1)
        )
        if other_thread is not None:
            return True

        if current_email is not None:
            conflicting_order = self.session.scalar(
                select(CustomerOrder)
                .where(
                    CustomerOrder.tenant_id == tenant_id,
                    CustomerOrder.soft_deleted.is_(False),
                    CustomerOrder.phone_hash == phone_hash,
                    CustomerOrder.email.is_not(None),
                    CustomerOrder.email != current_email.lower(),
                )
                .limit(1)
            )
            if conflicting_order is not None:
                return True
        return False

    def is_email_claimed_by_other_identity(
        self,
        tenant_id: int,
        email: str,
        *,
        current_thread_id: int | None,
        current_phone_hash: str | None,
    ) -> bool:
        normalized = email.strip().lower()
        other_thread = self.session.scalar(
            select(Thread)
            .where(
                Thread.tenant_id == tenant_id,
                Thread.soft_deleted.is_(False),
                Thread.email == normalized,
                *(
                    [Thread.thread_id != current_thread_id]
                    if current_thread_id is not None
                    else []
                ),
            )
            .limit(1)
        )
        if other_thread is not None:
            return True

        if current_phone_hash is not None:
            conflicting_order = self.session.scalar(
                select(CustomerOrder)
                .where(
                    CustomerOrder.tenant_id == tenant_id,
                    CustomerOrder.soft_deleted.is_(False),
                    CustomerOrder.email == normalized,
                    CustomerOrder.phone_hash.is_not(None),
                    CustomerOrder.phone_hash != current_phone_hash,
                )
                .limit(1)
            )
            if conflicting_order is not None:
                return True
        return False

    def create(
        self,
        *,
        tenant_id: int,
        phone_hash: str | None,
        email: str | None,
        expires_at: datetime,
        migrated_from_thread_id: int | None = None,
    ) -> Thread:
        thread = Thread(
            tenant_id=tenant_id,
            phone_hash=phone_hash,
            email=email,
            expires_at=expires_at,
            migrated_from_thread_id=migrated_from_thread_id,
            soft_deleted=False,
        )
        self.session.add(thread)
        self.session.flush()
        return thread

    def save(self, thread: Thread) -> Thread:
        self.session.add(thread)
        self.session.flush()
        return thread


class InMemoryThreadRepository:
    def __init__(self, threads: Sequence[Thread] | None = None) -> None:
        self._threads: list[Thread] = list(threads) if threads else []
        self._next_id: int = (
            max((t.thread_id for t in self._threads if t.thread_id is not None), default=1000) + 1
        )

    @property
    def threads(self) -> list[Thread]:
        return self._threads

    def find_by_email(self, tenant_id: int, email: str) -> Thread | None:
        normalized = email.strip().lower()
        if not normalized:
            return None
        matches = [
            t
            for t in self._threads
            if t.tenant_id == tenant_id
            and not getattr(t, "soft_deleted", False)
            and t.email is not None
            and t.email.lower() == normalized
        ]
        if not matches:
            return None
        matches.sort(key=lambda t: (t.expires_at, t.thread_id or 0), reverse=True)
        return matches[0]

    def find_by_phone_hash(self, tenant_id: int, phone_hash: str) -> Thread | None:
        cleaned = phone_hash.strip()
        if not cleaned:
            return None
        matches = [
            t
            for t in self._threads
            if t.tenant_id == tenant_id
            and not getattr(t, "soft_deleted", False)
            and t.phone_hash == cleaned
        ]
        if not matches:
            return None
        matches.sort(key=lambda t: (t.expires_at, t.thread_id or 0), reverse=True)
        return matches[0]

    def create(
        self,
        *,
        tenant_id: int,
        phone_hash: str | None,
        email: str | None,
        expires_at: datetime,
        migrated_from_thread_id: int | None = None,
    ) -> Thread:
        thread = Thread(
            thread_id=self._next_id,
            tenant_id=tenant_id,
            phone_hash=phone_hash,
            email=email,
            expires_at=expires_at,
            migrated_from_thread_id=migrated_from_thread_id,
            soft_deleted=False,
        )
        self._next_id += 1
        self._threads.append(thread)
        return thread

    def save(self, thread: Thread) -> Thread:
        if thread.thread_id is None:
            thread.thread_id = self._next_id
            self._next_id += 1
        if thread not in self._threads:
            self._threads.append(thread)
        return thread


@runtime_checkable
class OrderRepositoryProtocol(Protocol):
    def list_for_customer(
        self,
        *,
        tenant_id: int,
        phone_hash: str | None,
        email: str | None,
        since: datetime,
        limit: int = 10,
    ) -> Sequence[CustomerOrder]: ...

    def get_by_order_ref(self, tenant_id: int, order_ref: str) -> CustomerOrder | None: ...

    def get_by_awb(self, tenant_id: int, awb: str) -> CustomerOrder | None: ...


class SqlAlchemyOrderRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_for_customer(
        self,
        *,
        tenant_id: int,
        phone_hash: str | None,
        email: str | None,
        since: datetime,
        limit: int = 10,
    ) -> Sequence[CustomerOrder]:
        ownership_clauses = []
        if phone_hash is not None and phone_hash.strip():
            ownership_clauses.append(CustomerOrder.phone_hash == phone_hash.strip())
        if email is not None and email.strip():
            ownership_clauses.append(CustomerOrder.email == email.strip().lower())
        if not ownership_clauses:
            return []

        stmt = (
            select(CustomerOrder)
            .where(
                CustomerOrder.tenant_id == tenant_id,
                CustomerOrder.soft_deleted.is_(False),
                CustomerOrder.placed_at >= since,
                or_(*ownership_clauses),
            )
            .order_by(CustomerOrder.placed_at.desc(), CustomerOrder.order_pk.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def get_by_order_ref(self, tenant_id: int, order_ref: str) -> CustomerOrder | None:
        stmt = select(CustomerOrder).where(
            CustomerOrder.tenant_id == tenant_id,
            CustomerOrder.soft_deleted.is_(False),
            CustomerOrder.order_ref == order_ref,
        )
        return self.session.scalar(stmt)

    def get_by_awb(self, tenant_id: int, awb: str) -> CustomerOrder | None:
        stmt = (
            select(CustomerOrder)
            .join(Shipment, Shipment.order_pk == CustomerOrder.order_pk)
            .where(
                CustomerOrder.tenant_id == tenant_id,
                CustomerOrder.soft_deleted.is_(False),
                Shipment.awb == awb,
            )
            .order_by(CustomerOrder.placed_at.desc(), CustomerOrder.order_pk.desc())
            .limit(1)
        )
        return self.session.scalar(stmt)


class InMemoryOrderRepository:
    def __init__(
        self,
        orders: Sequence[CustomerOrder] | None = None,
        shipments: Sequence[Shipment] | None = None,
    ) -> None:
        self._orders: list[CustomerOrder] = list(orders) if orders else []
        self._shipments: list[Shipment] = list(shipments) if shipments else []

    def list_for_customer(
        self,
        *,
        tenant_id: int,
        phone_hash: str | None,
        email: str | None,
        since: datetime,
        limit: int = 10,
    ) -> Sequence[CustomerOrder]:
        cleaned_phone = phone_hash.strip() if phone_hash and phone_hash.strip() else None
        normalized_email = email.strip().lower() if email and email.strip() else None
        if cleaned_phone is None and normalized_email is None:
            return []
        matched = [
            o
            for o in self._orders
            if o.tenant_id == tenant_id
            and not getattr(o, "soft_deleted", False)
            and o.placed_at >= since
            and (
                (cleaned_phone is not None and o.phone_hash == cleaned_phone)
                or (
                    normalized_email is not None
                    and o.email is not None
                    and o.email.lower() == normalized_email
                )
            )
        ]
        matched.sort(key=lambda o: (o.placed_at, o.order_pk or 0), reverse=True)
        return matched[:limit]

    def get_by_order_ref(self, tenant_id: int, order_ref: str) -> CustomerOrder | None:
        for order in self._orders:
            if (
                order.tenant_id == tenant_id
                and not getattr(order, "soft_deleted", False)
                and order.order_ref == order_ref
            ):
                return order
        return None

    def get_by_awb(self, tenant_id: int, awb: str) -> CustomerOrder | None:
        order_pks = {s.order_pk for s in self._shipments if s.awb == awb}
        if not order_pks:
            return None
        candidates = [
            order
            for order in self._orders
            if order.tenant_id == tenant_id
            and not getattr(order, "soft_deleted", False)
            and order.order_pk in order_pks
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda o: (o.placed_at, o.order_pk or 0), reverse=True)
        return candidates[0]


@runtime_checkable
class StatusCopyRepositoryProtocol(Protocol):
    def get_copy_map(
        self,
        tenant_id: int,
        status: ShipmentStatus,
        failure_reason: FailureReason,
    ) -> dict[tuple[int, int, int], str]: ...


class SqlAlchemyStatusCopyRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_copy_map(
        self,
        tenant_id: int,
        status: ShipmentStatus,
        failure_reason: FailureReason,
    ) -> dict[tuple[int, int, int], str]:
        stmt = select(StatusCopy).where(
            StatusCopy.tenant_id.in_([tenant_id, 0]),
            StatusCopy.status == int(status),
            StatusCopy.failure_reason.in_([int(failure_reason), int(FailureReason.NONE)]),
            StatusCopy.soft_deleted.is_(False),
        )
        rows = self.session.scalars(stmt).all()
        return {
            (int(r.tenant_id), int(r.status), int(r.failure_reason)): r.customer_text
            for r in rows
        }


class InMemoryStatusCopyRepository:
    def __init__(self, copies: Sequence[StatusCopy] | dict[tuple[int, int, int], str] | None = None) -> None:
        if isinstance(copies, dict):
            self._map: dict[tuple[int, int, int], str] = {
                (int(k[0]), int(k[1]), int(k[2])): v for k, v in copies.items()
            }
        elif copies is not None:
            self._map = {
                (int(c.tenant_id), int(c.status), int(c.failure_reason)): c.customer_text
                for c in copies
                if not getattr(c, "soft_deleted", False)
            }
        else:
            self._map = {}

    def get_copy_map(
        self,
        tenant_id: int,
        status: ShipmentStatus,
        failure_reason: FailureReason,
    ) -> dict[tuple[int, int, int], str]:
        allowed_tenants = {int(tenant_id), 0}
        target_status = int(status)
        allowed_reasons = {int(failure_reason), int(FailureReason.NONE)}
        return {
            k: v
            for k, v in self._map.items()
            if k[0] in allowed_tenants and k[1] == target_status and k[2] in allowed_reasons
        }
