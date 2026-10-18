from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.models import BusinessHours, Tenant

WEEKDAY_KEYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKDAY_ALIASES: dict[str, int] = {
    "mon": 0,
    "monday": 0,
    "0": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "1": 1,
    "wed": 2,
    "wednesday": 2,
    "2": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "3": 3,
    "fri": 4,
    "friday": 4,
    "4": 4,
    "sat": 5,
    "saturday": 5,
    "5": 5,
    "sun": 6,
    "sunday": 6,
    "6": 6,
}
WEEK_MINUTES = 7 * 1440  # 10080


def _parse_hhmm(value: str | time) -> time:
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    parts = str(value).strip().split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    second = int(parts[2]) if len(parts) > 2 else 0
    return time(hour=hour, minute=minute, second=second)


def _time_to_minutes(value: str | time) -> int:
    if isinstance(value, str) and value.strip() == "24:00":
        return 1440
    t = _parse_hhmm(value)
    return t.hour * 60 + t.minute


def _extract_shift_open_close(shift: Any) -> tuple[str | time, str | time]:
    if isinstance(shift, (list, tuple)) and len(shift) >= 2:
        return shift[0], shift[1]
    if isinstance(shift, Mapping):
        if "open" in shift and "close" in shift:
            return shift["open"], shift["close"]
        if "start" in shift and "end" in shift:
            return shift["start"], shift["end"]
    if hasattr(shift, "open") and hasattr(shift, "close"):
        return getattr(shift, "open"), getattr(shift, "close")
    if hasattr(shift, "start") and hasattr(shift, "end"):
        return getattr(shift, "start"), getattr(shift, "end")
    raise ValueError(f"Unsupported shift format: {shift!r}")


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip())
    raise ValueError(f"Unsupported date value: {value!r}")


def _extract_weekly_and_overrides(
    cfg: BusinessHours | Mapping[str, Any] | Any,
) -> tuple[Mapping[Any, Any], Any]:
    if cfg is None:
        return {}, []
    if isinstance(cfg, Mapping):
        weekly = cfg.get("weekly")
        overrides = cfg.get("overrides", [])
        if weekly is None and any(str(k).lower() in WEEKDAY_ALIASES for k in cfg.keys()):
            weekly = cfg
        return (weekly if isinstance(weekly, Mapping) else {}), overrides
    weekly = getattr(cfg, "weekly", {}) or {}
    overrides = getattr(cfg, "overrides", [])
    return (weekly if isinstance(weekly, Mapping) else {}), (overrides if overrides is not None else [])


def _shifts_for_weekday(weekly: Mapping[Any, Any], weekday: int) -> list[tuple[time, time]]:
    raw_shifts: list[Any] = []
    for k, v in weekly.items():
        day_idx = k if isinstance(k, int) else WEEKDAY_ALIASES.get(str(k).strip().lower())
        if day_idx == weekday and isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            raw_shifts.extend(v)

    parsed: list[tuple[time, time]] = []
    for item in raw_shifts:
        o_raw, c_raw = _extract_shift_open_close(item)
        parsed.append((_parse_hhmm(o_raw), _parse_hhmm(c_raw)))
    return parsed


def _normalize_override_entries(overrides: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(overrides, Mapping):
        for date_key, val in overrides.items():
            d = _parse_date(date_key)
            if val is None or val is False:
                entries.append(
                    {
                        "start_date": d,
                        "end_date": d,
                        "kind": "CLOSED",
                        "shifts": [],
                        "message": None,
                    }
                )
            elif isinstance(val, Mapping):
                kind_raw = str(val.get("kind") or val.get("type") or val.get("mode") or "").upper()
                is_closed = bool(val.get("closed") is True or kind_raw == "CLOSED")
                raw_shifts = val.get("shifts") if "shifts" in val else val.get("windows", [])
                entries.append(
                    {
                        "start_date": _parse_date(val.get("start_date") or val.get("date") or d),
                        "end_date": _parse_date(val.get("end_date") or val.get("date") or d),
                        "kind": "CLOSED" if is_closed else (kind_raw or "CUSTOM"),
                        "shifts": [] if is_closed else list(raw_shifts or []),
                        "message": val.get("message"),
                    }
                )
            elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                entries.append(
                    {
                        "start_date": d,
                        "end_date": d,
                        "kind": "CUSTOM",
                        "shifts": list(val),
                        "message": None,
                    }
                )
        return entries

    if isinstance(overrides, Sequence) and not isinstance(overrides, (str, bytes)):
        for item in overrides:
            if isinstance(item, Mapping):
                d_single = item.get("date")
                start_raw = item.get("start_date") or item.get("from") or d_single
                end_raw = item.get("end_date") or item.get("to") or d_single or start_raw
                start_d = _parse_date(start_raw)
                end_d = _parse_date(end_raw)
                kind_raw = str(item.get("kind") or item.get("type") or item.get("mode") or "").upper()
                is_closed = bool(item.get("closed") is True or kind_raw == "CLOSED")
                raw_shifts = item.get("shifts") if "shifts" in item else item.get("windows", [])
                msg = item.get("message")
            else:
                d_single = getattr(item, "date", None)
                start_raw = getattr(item, "start_date", None) or d_single
                end_raw = getattr(item, "end_date", None) or d_single or start_raw
                start_d = _parse_date(start_raw)
                end_d = _parse_date(end_raw)
                kind_raw = str(
                    getattr(item, "kind", None)
                    or getattr(item, "type", None)
                    or getattr(item, "mode", None)
                    or ""
                ).upper()
                is_closed = bool(getattr(item, "closed", False) is True or kind_raw == "CLOSED")
                raw_shifts = getattr(item, "shifts", None)
                if raw_shifts is None:
                    raw_shifts = getattr(item, "windows", [])
                msg = getattr(item, "message", None)

            entries.append(
                {
                    "start_date": start_d,
                    "end_date": end_d,
                    "kind": "CLOSED" if is_closed else (kind_raw or "CUSTOM"),
                    "shifts": [] if is_closed else list(raw_shifts or []),
                    "message": msg,
                }
            )
    return entries


def _resolve_day_schedule(
    weekly: Mapping[Any, Any],
    override_entries: Sequence[dict[str, Any]],
    target_date: date,
) -> tuple[list[tuple[time, time]], str | None, bool]:
    """Return (shifts_for_day, override_message, has_override) for target_date.

    If several overrides cover the day, the last matching one wins.
    CLOSED means no shifts.
    """
    winning_override: dict[str, Any] | None = None
    for entry in override_entries:
        if entry["start_date"] <= target_date <= entry["end_date"]:
            winning_override = entry

    if winning_override is not None:
        msg = winning_override.get("message")
        if winning_override["kind"] == "CLOSED":
            return ([], msg, True)
        shifts: list[tuple[time, time]] = []
        for s in winning_override.get("shifts", []):
            o_raw, c_raw = _extract_shift_open_close(s)
            shifts.append((_parse_hhmm(o_raw), _parse_hhmm(c_raw)))
        return (shifts, msg, True)

    return (_shifts_for_weekday(weekly, target_date.weekday()), None, False)


class _IsOpenDescriptor:
    def __get__(
        self,
        instance: BusinessHoursService | None,
        owner: type[BusinessHoursService],
    ) -> Any:
        target = instance if instance is not None else owner()
        return target._is_open


class _ValidateDescriptor:
    def __get__(
        self,
        instance: BusinessHoursService | None,
        owner: type[BusinessHoursService],
    ) -> Any:
        target = instance if instance is not None else owner()
        return target._validate


class _SaveDescriptor:
    def __get__(
        self,
        instance: BusinessHoursService | None,
        owner: type[BusinessHoursService],
    ) -> Any:
        target = instance if instance is not None else owner()
        return target._save


class BusinessHoursService:
    """Deterministic business-hours evaluation and validation service."""

    def _is_open(
        self,
        cfg: BusinessHours | Mapping[str, Any] | Any,
        tz: str | ZoneInfo,
        at_utc: datetime,
    ) -> tuple[bool, datetime | None, str | None]:
        """Evaluate whether `at_utc` falls inside an open shift in `tz`.

        Returns `(open: bool, shift_end_utc: datetime | None, message: str | None)`.
        """
        if cfg is None or getattr(cfg, "soft_deleted", False):
            return (False, None, None)

        tz_info = ZoneInfo(tz) if isinstance(tz, str) else tz
        at_utc_aware = at_utc if at_utc.tzinfo is not None else at_utc.replace(tzinfo=UTC)
        at_utc_aware = at_utc_aware.astimezone(UTC)
        at_local = at_utc_aware.astimezone(tz_info)

        today = at_local.date()
        yesterday = today - timedelta(days=1)

        weekly, raw_overrides = _extract_weekly_and_overrides(cfg)
        override_entries = _normalize_override_entries(raw_overrides)

        today_shifts, today_msg, _ = _resolve_day_schedule(weekly, override_entries, today)
        yesterday_shifts, _, _ = _resolve_day_schedule(weekly, override_entries, yesterday)

        candidates: list[tuple[datetime, datetime]] = []

        # 1. Today's same-day shifts (open < close) and today's overnight shifts that have already started (close < open)
        for open_t, close_t in today_shifts:
            if open_t == close_t:
                continue
            if open_t < close_t:
                start_local = datetime.combine(today, open_t, tzinfo=tz_info)
                end_local = datetime.combine(today, close_t, tzinfo=tz_info)
                candidates.append((start_local, end_local))
            else:
                # Overnight shift starting today and ending tomorrow
                start_local = datetime.combine(today, open_t, tzinfo=tz_info)
                end_local = datetime.combine(today + timedelta(days=1), close_t, tzinfo=tz_info)
                if at_local >= start_local:
                    candidates.append((start_local, end_local))

        # 2. Yesterday's overnight shifts whose tail runs into today (close < open)
        for open_t, close_t in yesterday_shifts:
            if close_t < open_t:
                start_local = datetime.combine(yesterday, open_t, tzinfo=tz_info)
                end_local = datetime.combine(today, close_t, tzinfo=tz_info)
                candidates.append((start_local, end_local))

        containing = [
            (start_local, end_local)
            for start_local, end_local in candidates
            if start_local <= at_local < end_local
        ]

        if not containing:
            return (False, None, today_msg)

        # Pick the containing interval with the LATEST end
        _, latest_end_local = max(containing, key=lambda iv: iv[1])
        return (True, latest_end_local.astimezone(UTC), today_msg)

    def _validate(
        self,
        cfg: BusinessHours | Mapping[str, Any] | Any,
        tz: str | ZoneInfo = "UTC",
        now: datetime | None = None,
    ) -> None:
        """Validate business-hours config on save.

        Rules:
        - open != close
        - no overlaps on a 0..10080 weekly minute axis (split overnight shifts at the week boundary)
        - CUSTOM override shifts can't be overnight
        - override end date is not in the past
        """
        tz_info = ZoneInfo(tz) if isinstance(tz, str) else tz
        current_now = now if now is not None else datetime.now(UTC)
        if current_now.tzinfo is None:
            current_now = current_now.replace(tzinfo=UTC)
        today_local = current_now.astimezone(tz_info).date()

        weekly, raw_overrides = _extract_weekly_and_overrides(cfg)

        # 1. Validate weekly shifts on 0..10080 minute axis
        axis_intervals: list[tuple[int, int]] = []
        for k, v in weekly.items():
            day_idx = k if isinstance(k, int) else WEEKDAY_ALIASES.get(str(k).strip().lower())
            if day_idx is None:
                raise ValueError(f"Unknown weekday key: {k!r}")
            if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
                raise ValueError(f"Invalid shifts sequence for {k!r}")

            day_offset = day_idx * 1440
            for raw_shift in v:
                o_raw, c_raw = _extract_shift_open_close(raw_shift)
                open_min = _time_to_minutes(o_raw)
                close_min = _time_to_minutes(c_raw)

                if open_min == close_min:
                    raise ValueError(f"Shift open and close cannot be equal ({o_raw!r} == {c_raw!r})")

                if open_min < close_min:
                    axis_intervals.append((day_offset + open_min, day_offset + close_min))
                else:
                    # Overnight shift (close < open): split at the 10080 week boundary if needed
                    start_axis = day_offset + open_min
                    end_axis = day_offset + 1440 + close_min
                    if end_axis <= WEEK_MINUTES:
                        axis_intervals.append((start_axis, end_axis))
                    else:
                        axis_intervals.append((start_axis, WEEK_MINUTES))
                        wrapped_end = end_axis - WEEK_MINUTES
                        if wrapped_end > 0:
                            axis_intervals.append((0, wrapped_end))

        axis_intervals.sort(key=lambda iv: (iv[0], iv[1]))
        for idx in range(1, len(axis_intervals)):
            prev_start, prev_end = axis_intervals[idx - 1]
            curr_start, curr_end = axis_intervals[idx]
            if curr_start < prev_end:
                raise ValueError(
                    f"Overlapping weekly shifts on 0..10080 axis: "
                    f"[{prev_start}, {prev_end}) overlaps [{curr_start}, {curr_end})"
                )

        # 2. Validate overrides
        override_entries = _normalize_override_entries(raw_overrides)
        for entry in override_entries:
            start_d: date = entry["start_date"]
            end_d: date = entry["end_date"]
            if start_d > end_d:
                raise ValueError(f"Override start_date {start_d} cannot be after end_date {end_d}")
            if end_d < today_local:
                raise ValueError(
                    f"Override end date {end_d} cannot be in the past (today is {today_local})"
                )

            if entry["kind"] != "CLOSED":
                day_intervals: list[tuple[int, int]] = []
                for raw_shift in entry.get("shifts", []):
                    o_raw, c_raw = _extract_shift_open_close(raw_shift)
                    open_min = _time_to_minutes(o_raw)
                    close_min = _time_to_minutes(c_raw)
                    if open_min == close_min:
                        raise ValueError("CUSTOM override shift open and close cannot be equal")
                    if close_min < open_min:
                        raise ValueError("CUSTOM override shifts cannot be overnight")
                    day_intervals.append((open_min, close_min))

                day_intervals.sort(key=lambda iv: (iv[0], iv[1]))
                for idx in range(1, len(day_intervals)):
                    if day_intervals[idx][0] < day_intervals[idx - 1][1]:
                        raise ValueError("Overlapping shifts in CUSTOM override")

    def _save(
        self,
        cfg: BusinessHours | Mapping[str, Any] | Any,
        *,
        tz: str | ZoneInfo = "UTC",
        now: datetime | None = None,
        session: Session | None = None,
        tenant_id: int | None = None,
    ) -> BusinessHours:
        """Validate `cfg` and return/persist a `BusinessHours` model."""
        self._validate(cfg, tz=tz, now=now)

        if isinstance(cfg, BusinessHours):
            model = cfg
        else:
            weekly, overrides = _extract_weekly_and_overrides(cfg)
            resolved_tenant_id = tenant_id
            if resolved_tenant_id is None and isinstance(cfg, Mapping):
                resolved_tenant_id = cfg.get("tenant_id")
            if resolved_tenant_id is None:
                resolved_tenant_id = getattr(cfg, "tenant_id", 1)
            model = BusinessHours(
                tenant_id=int(resolved_tenant_id),
                weekly=dict(weekly),
                overrides=list(overrides) if isinstance(overrides, Sequence) else overrides,
                soft_deleted=False,
            )

        if session is not None:
            session.merge(model)
            session.flush()
        return model

    is_open = _IsOpenDescriptor()
    validate = _ValidateDescriptor()
    save = _SaveDescriptor()


def is_within_business_hours(
    tenant: Tenant,
    business_hours: BusinessHours | None,
    *,
    now: datetime | None = None,
) -> bool:
    """Deterministically evaluate whether `now` in the tenant's timezone is within business hours."""
    if business_hours is None or getattr(business_hours, "soft_deleted", False):
        return True

    current_now = now if now is not None else datetime.now(UTC)
    if current_now.tzinfo is None:
        current_now = current_now.replace(tzinfo=UTC)

    weekly, overrides = _extract_weekly_and_overrides(business_hours)
    if not weekly and not overrides:
        return True

    is_open_flag, _, _ = BusinessHoursService.is_open(
        business_hours,
        tenant.timezone,
        current_now,
    )
    return is_open_flag
