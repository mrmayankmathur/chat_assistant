from enum import IntEnum


class ShipmentStatus(IntEnum):
    ORDER_PLACED = 1
    MANIFESTED = 2
    PICKED_UP = 3
    IN_TRANSIT = 4
    OUT_FOR_DELIVERY = 5
    DELIVERED = 6
    FAILED_DELIVERY = 7
    RTO_INITIATED = 8
    RTO_DELIVERED = 9
    CANCELLED = 10
    LOST = 11
    DAMAGED = 12
    PICKUP_FAILED = 13


class FailureReason(IntEnum):
    NONE = 0
    CUSTOMER_UNAVAILABLE = 1
    ADDRESS_ISSUE = 2
    REFUSED = 3
    COD_NOT_READY = 4
    RESCHEDULE = 5
    ENTRY_RESTRICTED = 6


class ConversationStatus(IntEnum):
    OPEN = 1
    RESOLVED = 2
    ESCALATED = 3
    ABANDONED = 4
    RESOLVED_BY_HUMAN = 5
    STALE = 6


class MessageRole(IntEnum):
    CUSTOMER = 1
    ASSISTANT = 2
    HUMAN_AGENT = 3


class EscalationMode(IntEnum):
    TICKET = 1
    LIVE = 2


class CrmTicketKind(IntEnum):
    TICKET = 1
    LIVE = 2
    LIVE_TURNED_TICKET = 3


class CrmTicketStatus(IntEnum):
    OPEN = 1
    CLOSED = 2


class Tonality(IntEnum):
    PROFESSIONAL = 1
    FRIENDLY = 2
    EMPATHETIC = 3
    CONCISE = 4


class PolicyType(IntEnum):
    SHIPPING = 1
    CANCELLATION = 2
    REFUND = 3
    RETURN = 4


class CrmProvider(IntEnum):
    ZENDESK = 1
    FRESHDESK = 2
    GORGIAS = 3
    INTERCOM = 4
    CUSTOM_WEBHOOK = 5


class ConversationIntent(IntEnum):
    TRACK_ORDER = 1
    EXPLAIN_STATUS = 2
    CANCEL_ORDER = 3
    REFUND_STATUS = 4
    HUMAN_HANDOFF = 5
    OTHER = 99
