"""Структурированные бизнес- и технические ошибки, которые отдаёт конвейер."""


class BasketError(Exception):
    reason_code = "basket_error"

    def __init__(self, message: str, **details: object):
        super().__init__(message)
        self.details = details

    def to_dict(self) -> dict[str, object]:
        return {
            "reason_code": self.reason_code,
            "message": str(self),
            "details": self.details,
        }


class PackageError(BasketError):
    reason_code = "package_error"


class ReadError(BasketError):
    reason_code = "read_error"


class LlmError(BasketError):
    reason_code = "llm_error"


class StructuredOutputError(BasketError):
    reason_code = "structured_output_error"


class LayoutError(BasketError):
    reason_code = "layout_error"


class MeasurementPlanError(BasketError):
    reason_code = "measurement_plan_error"


class NotEvaluableError(BasketError):
    reason_code = "not_evaluable"


# Низкоуровневые имена transform сохранены как алиасы, а не отдельные контракты.
class ProfileError(LayoutError):
    pass


class ValueMapError(LayoutError):
    pass


class PassportError(MeasurementPlanError):
    pass


class MetricUnresolvableError(NotEvaluableError):
    pass


class MetricError(NotEvaluableError):
    pass
