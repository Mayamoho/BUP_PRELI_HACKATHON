from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, StrictInt, model_validator


Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Nonnegative = Annotated[Number, Field(ge=0)]
HourIndex = Annotated[StrictInt, Field(ge=0, le=23)]


def _whole_float_to_int(v):
    # accept 3.0 for hour 3 (some JSON encoders emit floats); 3.5, "3" and true stay invalid
    return int(v) if type(v) is float and v.is_integer() else v


RequestHourIndex = Annotated[HourIndex, BeforeValidator(_whole_float_to_int)]
Text = Annotated[str, Field(strict=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RequestModel(StrictModel):
    # Request objects tolerate unknown extra fields (types stay strict): a harness that adds
    # metadata must not turn a valid scenario into a 400.
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class Hour(RequestModel):
    hour: RequestHourIndex
    demand_kwh: Nonnegative
    solar_kwh: Nonnegative
    tariff_bdt_per_kwh: Number


class Battery(RequestModel):
    capacity_kwh: Nonnegative
    initial_energy_kwh: Nonnegative
    minimum_energy_kwh: Nonnegative
    max_charge_kwh_per_hour: Nonnegative
    max_discharge_kwh_per_hour: Nonnegative

    @model_validator(mode="after")
    def check_bounds(self):
        if not self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh:
            raise ValueError("Battery must satisfy minimum <= initial <= capacity")
        return self


class Scenario(RequestModel):
    scenario_id: Text
    operator_notes: Annotated[list[Text], Field(min_length=1, max_length=3)]
    hours: Annotated[list[Hour], Field(min_length=24, max_length=24)]
    battery: Battery

    @model_validator(mode="after")
    def check_scenario(self):
        if {h.hour for h in self.hours} != set(range(24)):
            raise ValueError("hours must contain each hour 0 through 23 exactly once")
        if any(not note.strip() for note in self.operator_notes):
            raise ValueError("operator_notes must not be blank")
        self.hours.sort(key=lambda h: h.hour)
        return self


class Window(StrictModel):
    hours: Annotated[list[HourIndex], Field(min_length=1, max_length=24)]

    @model_validator(mode="after")
    def check_hours(self):
        if self.hours != sorted(set(self.hours)):
            raise ValueError("Directive hours must be unique and ascending")
        return self


class SolarAdjustment(Window):
    factor: Annotated[Number, Field(ge=0, le=1)]


class ReserveAdjustment(Window):
    minimum_energy_kwh: Nonnegative


class GridAdjustment(Window):
    max_grid_kwh: Nonnegative


class Directive(StrictModel):
    note_index: Annotated[StrictInt, Field(ge=0, le=2)]
    applies: StrictBool
    directive_type: Literal[
        "solar_reduction", "minimum_battery_reserve", "no_charge_window",
        "no_discharge_window", "max_grid_window", "no_op",
    ]
    structured_adjustment: SolarAdjustment | ReserveAdjustment | GridAdjustment | Window | None
    explanation: Text

    @model_validator(mode="after")
    def check_adjustment(self):
        expected = {
            "solar_reduction": SolarAdjustment,
            "minimum_battery_reserve": ReserveAdjustment,
            "max_grid_window": GridAdjustment,
            "no_charge_window": Window,
            "no_discharge_window": Window,
            "no_op": type(None),
        }[self.directive_type]
        if type(self.structured_adjustment) is not expected:
            raise ValueError("Adjustment must match directive type exactly")
        if self.applies != (self.directive_type != "no_op"):
            raise ValueError("Only no_op may have applies=false")
        return self


class Interpretation(StrictModel):
    directive_interpretation: Annotated[list[Directive], Field(min_length=1, max_length=3)]

    def validate_for(self, scenario: Scenario):
        directives = self.directive_interpretation
        if [d.note_index for d in directives] != list(range(len(scenario.operator_notes))):
            raise ValueError("Every operator note must occur once in note_index order")
        for directive in directives:
            adjustment = directive.structured_adjustment
            if isinstance(adjustment, ReserveAdjustment):
                if adjustment.minimum_energy_kwh > scenario.battery.capacity_kwh:
                    raise ValueError("Reserve exceeds battery capacity")
        return self


class PlanHour(StrictModel):
    hour: HourIndex
    grid_kwh: Nonnegative
    solar_used_kwh: Nonnegative
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: Nonnegative
    battery_energy_after_kwh: Nonnegative


class Plan(StrictModel):
    scenario_id: Text
    directive_interpretation: list[Directive]
    hourly_plan: Annotated[list[PlanHour], Field(min_length=24, max_length=24)]
    total_grid_kwh: Nonnegative
    total_cost_bdt: Number
    peak_grid_kwh: Nonnegative
    plan_summary: Text
