"""Independent validation of a response against supplied or interpreted directives."""
from app.models import Plan, Scenario
from app.validation import replay as validate_plan

TOL = 0.01

def replay(request: dict, response: dict, directives: list[dict] | None = None) -> list[str]:
    try:
        scenario = Scenario.model_validate(request)
        fields = {name: response[name] for name in Plan.model_fields if name in response}
        fields["scenario_id"] = response.get("scenario_id", scenario.scenario_id)
        fields["plan_summary"] = response.get("plan_summary", "Optimizer validation")
        if directives is not None:
            fields["directive_interpretation"] = directives
        plan = Plan.model_validate(fields)
        validate_plan(scenario, plan, tolerance=TOL)
        return []
    except (ValueError, KeyError, TypeError) as exc:
        return ["Schedule failed validation: " + type(exc).__name__]
