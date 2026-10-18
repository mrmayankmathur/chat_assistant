from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.core import normalize_and_hash_phone, normalize_email
from app.models import Thread
from app.settings import Settings, get_settings
from repositories.storage import (
    InMemoryThreadRepository,
    SqlAlchemyThreadRepository,
    ThreadRepositoryProtocol,
)

THREAD_TTL_DAYS = 30


class IdentityService:
    def __init__(
        self,
        repo_or_session: ThreadRepositoryProtocol | Session | None = None,
        *,
        settings: Settings | None = None,
        default_region: str | None = None,
    ) -> None:
        if isinstance(repo_or_session, Session):
            self.repo: ThreadRepositoryProtocol = SqlAlchemyThreadRepository(repo_or_session)
        elif repo_or_session is None:
            self.repo = InMemoryThreadRepository()
        else:
            self.repo = repo_or_session
        self.settings = settings or get_settings()
        self.default_region = default_region or self.settings.default_phone_region

    def _can_latch_phone(
        self,
        *,
        tenant_id: int,
        matched: Thread,
        phone_hash: str,
        phone_thread: Thread | None,
    ) -> bool:
        if phone_thread is not None and phone_thread.thread_id != matched.thread_id:
            return False
        checker = getattr(self.repo, "is_phone_claimed_by_other_identity", None)
        if callable(checker) and checker(
            tenant_id,
            phone_hash,
            current_thread_id=matched.thread_id,
            current_email=matched.email,
        ):
            return False
        return True

    def _can_latch_email(
        self,
        *,
        tenant_id: int,
        matched: Thread,
        normalized_email: str,
        email_thread: Thread | None,
    ) -> bool:
        if email_thread is not None and email_thread.thread_id != matched.thread_id:
            return False
        checker = getattr(self.repo, "is_email_claimed_by_other_identity", None)
        if callable(checker) and checker(
            tenant_id,
            normalized_email,
            current_thread_id=matched.thread_id,
            current_phone_hash=matched.phone_hash,
        ):
            return False
        return True

    def resolve_thread(
        self,
        tenant_id: int,
        phone: str | None,
        email: str | None,
        *,
        now: datetime | None = None,
    ) -> Thread:
        """Resolve or create a customer thread for a tenant.

        - Normalizes phone to E.164 with `phonenumbers` (default region IN), then HMAC-SHA256s it.
        - Matches on email first, then phone_hash.
        - On a match, latches the missing identifier onto the thread only if it does not overwrite
          an existing value AND does not collide with another thread/customer identity in the tenant.
        - If the matched thread has expired (expires_at < now), creates a new thread with migrated_from_thread_id.
        - New threads expire 30 days after creation.
        """
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)

        phone_hash: str | None = None
        if phone is not None and phone.strip():
            phone_hash = normalize_and_hash_phone(
                phone,
                default_region=self.default_region,
                secret=self.settings.phone_hmac_secret,
            )

        normalized_email: str | None = None
        if email is not None and email.strip():
            normalized_email = normalize_email(email)

        if phone_hash is None and normalized_email is None:
            raise ValueError("At least one of phone or email must be provided")

        # 1. Look up existing threads by email and phone_hash within the tenant.
        email_thread: Thread | None = None
        if normalized_email is not None:
            email_thread = self.repo.find_by_email(tenant_id, normalized_email)

        phone_thread: Thread | None = None
        if phone_hash is not None:
            phone_thread = self.repo.find_by_phone_hash(tenant_id, phone_hash)

        # Match on email first, then phone.
        matched: Thread | None = email_thread if email_thread is not None else phone_thread

        expires_at = current_now + timedelta(days=THREAD_TTL_DAYS)

        if matched is None:
            return self.repo.create(
                tenant_id=tenant_id,
                phone_hash=phone_hash,
                email=normalized_email,
                expires_at=expires_at,
                migrated_from_thread_id=None,
            )

        # Compute latched identifiers without ever overwriting an existing different value
        # or colliding with another customer's thread/identity in the same tenant.
        latched_phone_hash = matched.phone_hash
        if (
            latched_phone_hash is None
            and phone_hash is not None
            and self._can_latch_phone(
                tenant_id=tenant_id,
                matched=matched,
                phone_hash=phone_hash,
                phone_thread=phone_thread,
            )
        ):
            latched_phone_hash = phone_hash

        latched_email = matched.email
        if (
            latched_email is None
            and normalized_email is not None
            and self._can_latch_email(
                tenant_id=tenant_id,
                matched=matched,
                normalized_email=normalized_email,
                email_thread=email_thread,
            )
        ):
            latched_email = normalized_email

        matched_expires_at = matched.expires_at
        if matched_expires_at is not None and matched_expires_at.tzinfo is None:
            matched_expires_at = matched_expires_at.replace(tzinfo=UTC)

        # If the matched thread has expired (expires_at < now), create a new thread and set migrated_from_thread_id.
        if matched_expires_at is not None and matched_expires_at < current_now:
            return self.repo.create(
                tenant_id=tenant_id,
                phone_hash=latched_phone_hash,
                email=latched_email,
                expires_at=expires_at,
                migrated_from_thread_id=matched.thread_id,
            )

        # Latch missing non-conflicting identifier onto the active thread.
        changed = False
        if matched.phone_hash is None and latched_phone_hash is not None:
            matched.phone_hash = latched_phone_hash
            changed = True
        if matched.email is None and latched_email is not None:
            matched.email = latched_email
            changed = True

        if changed:
            return self.repo.save(matched)
        return matched
