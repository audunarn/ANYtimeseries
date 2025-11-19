"""Built-in fatigue curve definitions for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

CurveType = Literal["sn", "tn"]


@dataclass(frozen=True)
class FatigueCurveTemplate:
    """Description of a built-in fatigue curve."""

    key: str
    label: str
    curve_type: CurveType
    source: str
    parameters: dict[str, float]
    notes: str = ""
    lm_formula: tuple[float, float] | None = None


CURVE_LIBRARY: tuple[FatigueCurveTemplate, ...] = (
    FatigueCurveTemplate(
        key="dnv_b1",
        label="DNV Class B1 (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 12.164},
        notes="Rolled plate and extruded sections.",
    ),
    FatigueCurveTemplate(
        key="dnv_b2",
        label="DNV Class B2 (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 12.097},
        notes="Extruded sections with weld improvements.",
    ),
    FatigueCurveTemplate(
        key="dnv_c",
        label="DNV Class C (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 12.012},
        notes="Butt welds ground flush.",
    ),
    FatigueCurveTemplate(
        key="dnv_c1",
        label="DNV Class C1 (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.923},
        notes="Machined or rolled sections.",
    ),
    FatigueCurveTemplate(
        key="dnv_d",
        label="DNV Class D (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.821},
        notes="Typical welded plate details.",
    ),
    FatigueCurveTemplate(
        key="dnv_e",
        label="DNV Class E (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.727},
        notes="Plates with welded attachments.",
    ),
    FatigueCurveTemplate(
        key="dnv_f",
        label="DNV Class F (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.591},
        notes="Non-ground transverse attachments.",
    ),
    FatigueCurveTemplate(
        key="dnv_f1",
        label="DNV Class F1 (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.528},
        notes="Stiffener toes with moderate stress concentration.",
    ),
    FatigueCurveTemplate(
        key="dnv_f2",
        label="DNV Class F2 (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.487},
        notes="Stiffener toes with high stress concentration.",
    ),
    FatigueCurveTemplate(
        key="dnv_g",
        label="DNV Class G (air)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.398},
        notes="Highly stressed plate intersections.",
    ),
    FatigueCurveTemplate(
        key="dnv_w1",
        label="DNV Class W1 (seawater, cathodic protection)",
        curve_type="sn",
        source="DNV-RP-C203",
        parameters={"m1": 3.0, "m2": 5.0, "nswitch": 1e7, "loga1": 11.215},
        notes="Seawater submerged details with CP.",
    ),
    FatigueCurveTemplate(
        key="abs_chain_r3",
        label="ABS common stud-link chain",
        curve_type="tn",
        source="ABS Position Mooring Systems",
        parameters={"m1": 3.0, "a1": 1000.0},
        notes="Table 2 tension-tension parameters (tension range as fraction of MBS).",
    ),
    FatigueCurveTemplate(
        key="abs_chain_r4",
        label="ABS common studless link chain",
        curve_type="tn",
        source="ABS Position Mooring Systems",
        parameters={"m1": 3.0, "a1": 316.0},
        notes="Table 2 tension-tension parameters (tension range as fraction of MBS).",
    ),
    FatigueCurveTemplate(
        key="abs_wire_spiral",
        label="ABS six/multi-strand wire rope (corrosion protected)",
        curve_type="tn",
        source="ABS Position Mooring Systems",
        parameters={"m1": 4.09},
        notes="loga1 = 3.20 - 2.79 · Lm (Lm: mean tension to MBS ratio).",
        lm_formula=(3.20, -2.79),
    ),
    FatigueCurveTemplate(
        key="abs_polyester",
        label="ABS polyester rope",
        curve_type="tn",
        source="ABS Position Mooring Systems",
        parameters={"m1": 5.2, "a1": 25000.0},
        notes="Synthetic fibre mooring leg (tension-tension).",
    ),
    FatigueCurveTemplate(
        key="abs_spiral_wire",
        label="ABS spiral strand wire rope (corrosion protected)",
        curve_type="tn",
        source="ABS Position Mooring Systems",
        parameters={"m1": 5.05},
        notes="loga1 = 3.23 - 3.43 · Lm (Lm: mean tension to MBS ratio).",
        lm_formula=(3.23, -3.43),
    ),
    FatigueCurveTemplate(
        key="dnv_chain_r4",
        label="DNV chain Grade R4",
        curve_type="tn",
        source="DNV-OS-E301",
        parameters={"m1": 3.0, "loga1": 9.55},
        notes="DNV OS-E301 chain design curve.",
    ),
    FatigueCurveTemplate(
        key="dnv_wire_spiral",
        label="DNV spiral strand wire",
        curve_type="tn",
        source="DNV-OS-E301",
        parameters={"m1": 4.0, "loga1": 10.25},
        notes="Wire mooring design curve.",
    ),
    FatigueCurveTemplate(
        key="dnv_polyester",
        label="DNV polyester rope",
        curve_type="tn",
        source="DNV-OS-E301",
        parameters={"m1": 5.0, "loga1": 11.1},
        notes="Synthetic rope design curve.",
    ),
)


def curve_templates(curve_type: CurveType | None = None) -> Iterable[FatigueCurveTemplate]:
    """Yield available templates, optionally filtered by curve type."""

    for template in CURVE_LIBRARY:
        if curve_type is None or template.curve_type == curve_type:
            yield template


def find_template(key: str) -> FatigueCurveTemplate | None:
    """Return template matching ``key`` if it exists."""

    for template in CURVE_LIBRARY:
        if template.key == key:
            return template
    return None
