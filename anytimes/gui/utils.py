"""Utility helpers and shared constants for the AnytimeSeries GUI."""
from __future__ import annotations

import re

ORCAFLEX_VARIABLE_MAP = {'Vessel': ['X', 'Y', 'Z', 'Dynamic x', 'Dynamic y', 'Dynamic z', 'Rotation 1', 'Rotation 2', 'Rotation 3', 'Dynamic Rx', 'Dynamic Ry', 'Dynamic Rz', 'Sea surface Z', 'Sea surface clearance', 'Sea velocity', 'Sea X velocity', 'Sea Y velocity', 'Sea Z velocity', 'Sea acceleration', 'Sea X acceleration', 'Sea Y acceleration', 'Sea Z acceleration', 'Disturbed sea surface Z', 'Disturbed sea surface clearance', 'Disturbed sea velocity', 'Disturbed sea X velocity', 'Disturbed sea Y velocity', 'Disturbed sea Z velocity', 'Disturbed sea acceleration', 'Disturbed sea X acceleration', 'Disturbed sea Y acceleration', 'Disturbed sea Z acceleration', 'Air gap', 'Velocity', 'GX velocity', 'GY velocity', 'GZ velocity', 'x velocity', 'y velocity', 'z velocity', 'Angular velocity', 'x angular velocity', 'y angular velocity', 'z angular velocity', 'Acceleration', 'GX acceleration', 'GY acceleration', 'GZ acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Acceleration rel. g', 'x acceleration rel. g', 'y acceleration rel. g', 'z acceleration rel. g', 'Angular acceleration', 'x angular acceleration', 'y angular acceleration', 'z angular acceleration', 'Primary X', 'Primary Y', 'Primary Z', 'Primary rotation 1', 'Primary rotation 2', 'Primary rotation 3', 'Primary velocity', 'Primary x velocity', 'Primary y velocity', 'Primary z velocity', 'Primary angular velocity', 'Primary x angular velocity', 'Primary y angular velocity', 'Primary z angular velocity', 'Primary acceleration', 'Primary x acceleration', 'Primary y acceleration', 'Primary z acceleration', 'Primary angular acceleration', 'Primary x angular acceleration', 'Primary y angular acceleration', 'Primary z angular acceleration', 'Primary LF X', 'Primary LF Y', 'Primary LF Z', 'Primary LF rotation 1', 'Primary LF rotation 2', 'Primary LF rotation 3', 'Primary WF surge', 'Primary WF sway', 'Primary WF heave', 'Primary WF roll', 'Primary WF pitch', 'Primary WF yaw', 'Total force', 'Total Lx force', 'Total Ly force', 'Total Lz force', 'Total moment', 'Total Lx moment', 'Total Ly moment', 'Total Lz moment', 'Connections force', 'Connections Lx force', 'Connections Ly force', 'Connections Lz force', 'Connections moment', 'Connections Lx moment', 'Connections Ly moment', 'Connections Lz moment', 'Connections GX force', 'Connections GY force', 'Connections GZ force', 'Connections GX moment', 'Connections GY moment', 'Connections GZ moment', 'Hydrostatic stiffness force', 'Hydrostatic stiffness Lx force', 'Hydrostatic stiffness Ly force', 'Hydrostatic stiffness Lz force', 'Hydrostatic stiffness moment', 'Hydrostatic stiffness Lx moment', 'Hydrostatic stiffness Ly moment', 'Hydrostatic stiffness Lz moment', 'Morison elements force', 'Morison elements Lx force', 'Morison elements Ly force', 'Morison elements Lz force', 'Morison elements moment', 'Morison elements Lx moment', 'Morison elements Ly moment', 'Morison elements Lz moment', 'Morison element drag force', 'Morison element Lx drag force', 'Morison element Ly drag force', 'Morison element Lz drag force', 'Morison element fluid inertia force', 'Morison element Lx fluid inertia force', 'Morison element Ly fluid inertia force', 'Morison element Lz fluid inertia force', 'Morison element segment proportion wet', 'Morison element segment relative velocity', 'Morison element segment normal relative velocity', 'Morison element segment x relative velocity', 'Morison element segment y relative velocity', 'Morison element segment z relative velocity', 'Morison element segment x drag coefficient', 'Morison element segment y drag coefficient', 'Morison element segment z drag coefficient', 'Morison element segment drag force', 'Morison element segment x drag force', 'Morison element segment y drag force', 'Morison element segment z drag force', 'Morison element segment fluid inertia force', 'Morison element segment x fluid inertia force', 'Morison element segment y fluid inertia force', 'Morison element segment z fluid inertia force', 'Applied force', 'Applied Lx force', 'Applied Ly force', 'Applied Lz force', 'Applied moment', 'Applied Lx moment', 'Applied Ly moment', 'Applied Lz moment', 'Wave (1st order) force', 'Wave (1st order) Lx force', 'Wave (1st order) Ly force', 'Wave (1st order) Lz force', 'Wave (1st order) moment', 'Wave (1st order) Lx moment', 'Wave (1st order) Ly moment', 'Wave (1st order) Lz moment', 'Wave drift (2nd order) force', 'Wave drift (2nd order) Lx force', 'Wave drift (2nd order) Ly force', 'Wave drift (2nd order) Lz force', 'Wave drift (2nd order) moment', 'Wave drift (2nd order) Lx moment', 'Wave drift (2nd order) Ly moment', 'Wave drift (2nd order) Lz moment', 'Sum frequency force', 'Sum frequency Lx force', 'Sum frequency Ly force', 'Sum frequency Lz force', 'Sum frequency moment', 'Sum frequency Lx moment', 'Sum frequency Ly moment', 'Sum frequency Lz moment', 'Added mass & damping force', 'Added mass & damping Lx force', 'Added mass & damping Ly force', 'Added mass & damping Lz force', 'Added mass & damping moment', 'Added mass & damping Lx moment', 'Added mass & damping Ly moment', 'Added mass & damping Lz moment', 'Surface Pressures'], 'Constraint': ['Displacement', 'x', 'y', 'z', 'Angular displacement', 'Rx', 'Ry', 'Rz', 'Velocity', 'x velocity', 'y velocity', 'z velocity', 'Angular velocity', 'x angular velocity', 'y angular velocity', 'z angular velocity', 'Acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Angular acceleration', 'x angular acceleration', 'y angular acceleration', 'z angular acceleration', 'In-frame X', 'In-frame Y', 'In-frame Z', 'In-frame dynamic x', 'In-frame dynamic y', 'In-frame dynamic z', 'In-frame azimuth', 'In-frame declination', 'In-frame gamma', 'In-frame dynamic Rx', 'In-frame dynamic Ry', 'In-frame dynamic Rz', 'In-frame velocity', 'In-frame GX velocity', 'In-frame GY velocity', 'In-frame GZ velocity', 'In-frame x velocity', 'In-frame y velocity', 'In-frame z velocity', 'In-frame angular velocity', 'In-frame x angular velocity', 'In-frame y angular velocity', 'In-frame z angular velocity', 'In-frame acceleration', 'In-frame GX acceleration', 'In-frame GY acceleration', 'In-frame GZ acceleration', 'In-frame x acceleration', 'In-frame y acceleration', 'In-frame z acceleration', 'In-frame angular acceleration', 'In-frame x angular acceleration', 'In-frame y angular acceleration', 'In-frame z angular acceleration', 'In-frame connection force', 'In-frame connection GX force', 'In-frame connection GY force', 'In-frame connection GZ force', 'In-frame connection Lx force', 'In-frame connection Ly force', 'In-frame connection Lz force', 'In-frame connection moment', 'In-frame connection GX moment', 'In-frame connection GY moment', 'In-frame connection GZ moment', 'In-frame connection Lx moment', 'In-frame connection Ly moment', 'In-frame connection Lz moment', 'Out-frame X', 'Out-frame Y', 'Out-frame Z', 'Out-frame dynamic x', 'Out-frame dynamic y', 'Out-frame dynamic z', 'Out-frame azimuth', 'Out-frame declination', 'Out-frame gamma', 'Out-frame dynamic Rx', 'Out-frame dynamic Ry', 'Out-frame dynamic Rz', 'Out-frame velocity', 'Out-frame GX velocity', 'Out-frame GY velocity', 'Out-frame GZ velocity', 'Out-frame x velocity', 'Out-frame y velocity', 'Out-frame z velocity', 'Out-frame angular velocity', 'Out-frame x angular velocity', 'Out-frame y angular velocity', 'Out-frame z angular velocity', 'Out-frame acceleration', 'Out-frame GX acceleration', 'Out-frame GY acceleration', 'Out-frame GZ acceleration', 'Out-frame x acceleration', 'Out-frame y acceleration', 'Out-frame z acceleration', 'Out-frame angular acceleration', 'Out-frame x angular acceleration', 'Out-frame y angular acceleration', 'Out-frame z angular acceleration', 'Out-frame connection force', 'Out-frame connection GX force', 'Out-frame connection GY force', 'Out-frame connection GZ force', 'Out-frame connection Lx force', 'Out-frame connection Ly force', 'Out-frame connection Lz force', 'Out-frame connection moment', 'Out-frame connection GX moment', 'Out-frame connection GY moment', 'Out-frame connection GZ moment', 'Out-frame connection Lx moment', 'Out-frame connection Ly moment', 'Out-frame connection Lz moment', 'Surface Pressures'], '6Dbuoy': ['X', 'Y', 'Z', 'Dynamic x', 'Dynamic y', 'Dynamic z', 'Rotation 1', 'Rotation 2', 'Rotation 3', 'Azimuth', 'Declination', 'Dynamic Rx', 'Dynamic Ry', 'Dynamic Rz', 'Velocity', 'GX velocity', 'GY velocity', 'GZ velocity', 'x velocity', 'y velocity', 'z velocity', 'Angular velocity', 'x angular velocity', 'y angular velocity', 'z angular velocity', 'Acceleration', 'GX acceleration', 'GY acceleration', 'GZ acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Acceleration rel. g', 'x acceleration rel. g', 'y acceleration rel. g', 'z acceleration rel. g', 'Angular acceleration', 'x angular acceleration', 'y angular acceleration', 'z angular acceleration', 'Dry length', 'Wetted volume', 'Sea surface Z', 'Sea surface clearance', 'Sea velocity', 'Sea X velocity', 'Sea Y velocity', 'Sea Z velocity', 'Sea acceleration', 'Sea X acceleration', 'Sea Y acceleration', 'Sea Z acceleration', 'Applied force', 'Applied Lx force', 'Applied Ly force', 'Applied Lz force', 'Applied moment', 'Applied Lx moment', 'Applied Ly moment', 'Applied Lz moment', 'dummy', 'Surface Pressures'], 'Line': ['Tension per 50 mm', 'End force', 'End moment', 'End force Ez angle', 'End force Exy angle', 'End force Ezx angle', 'End force Ezy angle', 'End force azimuth', 'End force declination', 'No moment azimuth', 'No moment declination', 'End Ex force', 'End Ey force', 'End Ez force', 'End Ex moment', 'End Ey moment', 'End Ez moment', 'End Lx force', 'End Ly force', 'End Lz force', 'End Lx moment', 'End Ly moment', 'End Lz moment', 'End GX force', 'End GY force', 'End GZ force', 'End GX moment', 'End GY moment', 'End GZ moment', 'X', 'Y', 'Z', 'Dynamic x', 'Dynamic y', 'Dynamic z', 'Azimuth', 'Declination', 'Gamma', 'Twist', 'Node azimuth', 'Node declination', 'Node gamma', 'Dynamic Rx', 'Dynamic Ry', 'Dynamic Rz', 'Layback', 'Velocity', 'GX velocity', 'GY velocity', 'GZ velocity', 'x velocity', 'y velocity', 'z velocity', 'Acceleration', 'GX acceleration', 'GY acceleration', 'GZ acceleration', 'x acceleration', 'y acceleration', 'z acceleration', 'Acceleration rel. g', 'x acceleration rel. g', 'y acceleration rel. g', 'z acceleration rel. g', 'Effective tension', 'Wall tension', 'Normalised tension', 'Sidewall pressure', 'Total mean axial strain', 'Direct tensile strain', 'Max bending strain', 'Max pipelay von Mises strain', 'Worst ZZ strain', 'ZZ strain', 'Contents density', 'Contents temperature', 'Contents pressure', 'Contents flow rate', 'Contents flow velocity', 'Fluid incidence angle', 'Bend moment', 'x bend moment', 'y bend moment', 'Bend moment component', 'In plane bend moment', 'Out of plane bend moment', 'Curvature', 'Normalised curvature', 'x curvature', 'y curvature', 'Curvature component', 'In plane curvature', 'Out of plane curvature', 'Bend radius', 'x bend radius', 'y bend radius', 'Bend radius component', 'In plane bend radius', 'Out of plane bend radius', 'Shear force', 'x shear force', 'y shear force', 'Shear force component', 'In plane shear force', 'Out of plane shear force', 'Max von Mises stress', 'Bending stress', 'Max bending stress', 'Pm', 'Pb', 'Worst ZZ stress', 'Direct tensile stress', 'Worst hoop stress', 'Max xy shear stress', 'Internal pressure', 'External pressure', 'Net internal pressure', 'von Mises stress', 'RR stress', 'CC stress', 'ZZ stress', 'RC stress', 'RZ stress', 'CZ stress', 'API RP 2RD stress', 'API RP 2RD utilisation', 'API STD 2RD method 1', 'API STD 2RD method 2', 'API RP 1111 LLD', 'API RP 1111 CLD', 'API RP 1111 BEP', 'API RP 1111 max combined', 'DNV OS F101 disp. controlled', 'DNV OS F101 load controlled', 'DNV ST F101 disp. controlled', 'DNV ST F101 load controlled', 'DNV ST F101 simplified strain', 'DNV ST F101 simplified stress', 'DNV ST F101 tension utilisation', 'DNV OS F201 LRFD', 'DNV OS F201 WSD', 'PD 8010 allowable stress check', 'PD 8010 axial compression check', 'PD 8010 bending check', 'PD 8010 torsion check', 'PD 8010 load combinations check', 'PD 8010 bending strain check', 'Line clearance', 'Line centreline clearance', 'Line horizontal centreline clearance', 'Line vertical centreline clearance', 'Whole line clearance', 'Whole line centreline clearance', 'Whole line horizontal centreline clearance', 'Whole line vertical centreline clearance', 'Seabed clearance', 'Vertical seabed clearance', 'Line clash force', 'Line clash impulse', 'Line clash energy', 'Solid contact force', 'Seabed normal penetration/D', 'Seabed normal resistance', 'Seabed normal resistance/D', 'Arc length', 'Expansion factor', 'Ez angle', 'Exy angle', 'Ezx angle', 'Ezy angle', 'Relative velocity', 'Normal relative velocity', 'x relative velocity', 'y relative velocity', 'z relative velocity', 'Strouhal frequency', 'Reynolds number', 'x drag coefficient', 'y drag coefficient', 'z drag coefficient', 'Lift coefficient', 'Drag force', 'Normal drag force', 'x drag force', 'y drag force', 'z drag force', 'Lift force', 'Fluid inertia force', 'x fluid inertia force', 'y fluid inertia force', 'z fluid inertia force', 'Morison force', 'Normal Morison force', 'x Morison force', 'y Morison force', 'z Morison force', 'Sea surface Z', 'Depth', 'Sea surface clearance', 'Proportion wet', 'Sea velocity', 'Sea X velocity', 'Sea Y velocity', 'Sea Z velocity', 'Sea acceleration', 'Sea X acceleration', 'Sea Y acceleration', 'Sea Z acceleration'], 'Environment': ['Elevation', 'Velocity', 'X velocity', 'Y velocity', 'Z velocity', 'Acceleration', 'X acceleration', 'Y acceleration', 'Z acceleration', 'Current speed', 'Current direction', 'Current X velocity', 'Current Y velocity', 'Current Z velocity', 'Current acceleration', 'Current X acceleration', 'Current Y acceleration', 'Current Z acceleration', 'Wind speed', 'Wind direction', 'Wind X velocity', 'Wind Y velocity', 'Wind Z velocity', 'Static pressure', 'Density', 'Surface Pressures', 'Seabed Z'], 'General': ['Time', 'Ramp', 'Implicit solver iteration count', 'Implicit solver time step']}

MATH_FUNCTIONS = [
    "sin()", "cos()", "tan()", "sqrt()", "exp()", "log()", "abs()",
    "min()", "max()", "radians()", "degrees()", "pow(x, y)"
]

def _find_xyz_triples(
    varnames: list[str], warn_if_fallback: bool = True
) -> list[tuple[str, str, str]]:
    """
    Return a list of (x, y, z) triplets found in *varnames*.

    Strategy
    --------
    1.  Look for “perfect” matches that differ **only** by the axis token.
        • Tokens recognised (case-insensitive):
              x, y, z
              xpos, ypos, zpos
              posx, posy, posz
              <anything>_x, _y, _z               (trailing axis)
        • The remaining stem (with the axis part removed) must be identical.

    2.  Any variables not matched in step 1 are grouped naïvely in the
        original order (batch of 3).  If this fallback is used and
        *warn_if_fallback* is True, a warning dialog is shown.

    Returns
    -------
    list of (x_name, y_name, z_name)  – may be empty on failure.
    """
    import re, tkinter.messagebox as mb, itertools, collections as _C

    # ── regex for axis detection ───────────────────────────────────
    X_PAT = re.compile(r"(?:\b|_)(x|xpos|posx)(?:\b|_)?", re.I)
    Y_PAT = re.compile(r"(?:\b|_)(y|ypos|posy)(?:\b|_)?", re.I)
    Z_PAT = re.compile(r"(?:\b|_)(z|zpos|posz)(?:\b|_)?", re.I)

    def _axis(nm: str) -> str | None:
        if X_PAT.search(nm):
            return "x"
        if Y_PAT.search(nm):
            return "y"
        if Z_PAT.search(nm):
            return "z"
        return None

    # ── step 1  • perfect matches  ─────────────────────────────────
    stems: _C.defaultdict[str, dict[str, str]] = _C.defaultdict(dict)

    for nm in varnames:
        ax = _axis(nm)
        if not ax:
            continue
        # strip *only* the first axis occurrence to obtain a stem
        stem = X_PAT.sub("", nm, count=1)
        stem = Y_PAT.sub("", stem, count=1)
        stem = Z_PAT.sub("", stem, count=1)
        stems[stem][ax] = nm

    perfect = [
        (d["x"], d["y"], d["z"]) for d in stems.values() if {"x", "y", "z"} <= d.keys()
    ]

    matched = set(itertools.chain.from_iterable(perfect))
    leftovers = [nm for nm in varnames if nm not in matched]

    # ── step 2  • positional fallback  ─────────────────────────────
    fallback = []
    if leftovers:
        for i in range(0, len(leftovers), 3):
            trio = leftovers[i : i + 3]
            if len(trio) == 3:
                fallback.append(tuple(trio))

        if fallback and warn_if_fallback:
            mb.showwarning(
                "Ambiguous XYZ pairing",
                "One or more triplets were built by simple ordering because "
                "their axis letters could not be identified uniquely.\n\n"
                "Check that the colours / legends in the animation make sense!",
            )

    return perfect + fallback

_USER_PATTERNS = (
    r"_shift0$",
    r"_shiftNZ$",
    r"_shiftCommon",
    r"_f\d+$",
    r"\bmean\(",
    r"\bsqrt_sum_of_squares\(",
    r"×1000$",
    r"÷1000$",
    r"_rad$",
    r"_deg$",
    r"×10$",
    r"÷10$",
    r"×2",
    r"÷2$",
    # ── NEW:  apply-values “p / m / x / d” suffixes ───────────────
    r"_p\d+(?:\.\d+)?(?:_f\d+)?$",
    r"_m\d+(?:\.\d+)?(?:_f\d+)?$",
    r"_x\d+(?:\.\d+)?(?:_f\d+)?$",
    r"_d\d+(?:\.\d+)?(?:_f\d+)?$",
)

_user_regex = re.compile("|".join(_USER_PATTERNS))

_SAFE_RE = re.compile(r"\W")  # "not [A-Za-z0-9_]"

def _safe(name: str) -> str:
    """Return ``name`` converted to a valid Python identifier."""
    s = _SAFE_RE.sub("_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s

def _looks_like_user_var(name: str) -> bool:
    return bool(_user_regex.search(name))

def _parse_search_terms(text: str) -> list[list[str]]:
    """Return a list of search term groups from ``text``.

    The input may contain comma separated terms. If ``",,"`` occurs in the
    text, a literal comma is also included as a search term.  A term enclosed
    in ``!!`` is interpreted as a group of comma separated alternatives which
    will match if *any* of the alternatives are found.

    Example
    -------
    ``"coords, !!x,y,z!!, hor"`` results in ``[['coords'], ['x', 'y', 'z'],
    ['hor']]``.
    """

    text = text.lower()
    include_comma = ",," in text
    placeholders: dict[str, list[str]] = {}

    def _replace(match: re.Match) -> str:
        idx = len(placeholders)
        placeholder = f"\x00{idx}\x00"
        tokens = [t.strip() for t in match.group(1).split(',') if t.strip()]
        placeholders[placeholder] = tokens or [""]
        return placeholder

    # Temporarily replace !!..!! groups with placeholders to avoid splitting
    # them on commas
    text_no_groups = re.sub(r"!!(.*?)!!", _replace, text)

    groups: list[list[str]] = []
    for tok in [t.strip() for t in text_no_groups.split(',') if t.strip()]:
        if tok in placeholders:
            groups.append(placeholders[tok])
        else:
            groups.append([tok])

    if include_comma:
        groups.append([','])

    return groups

def _matches_terms(name: str, terms: list[list[str]]) -> bool:
    """Return ``True`` if ``name`` matches all search ``terms``.

    Each element in ``terms`` is a list of alternatives; at least one
    alternative must be present in ``name`` for the term to match.
    """

    if not terms:
        return True

    name_l = name.lower()
    return all(any(t in name_l for t in group) for group in terms)

def get_object_available_vars(obj, orcaflex_varmap=None):
    """Return the variables available for an OrcaFlex object or type.

    ``obj`` may be either an OrcaFlex object instance or a ``typeName`` string.
    The helper keeps working with bare type names so callers don't have to keep
    references to OrcaFlex COM objects alive longer than necessary.
    """

    type_name = getattr(obj, "typeName", None)
    if type_name is None and isinstance(obj, str):
        type_name = obj

    if orcaflex_varmap is not None:
        if type_name is None:
            return []
        return orcaflex_varmap.get(type_name, [])

    if type_name is None:
        return []

    for attr in ["AvailableTimeHistories", "AvailableDerivedVariables"]:
        if hasattr(obj, attr):
            try:
                vals = getattr(obj, attr)
                if isinstance(vals, (list, tuple)):
                    return list(vals)
                elif hasattr(vals, "__iter__"):
                    return list(vals)
            except Exception:
                continue
    if hasattr(obj, "AvailableVariables"):
        try:
            vals = obj.AvailableVariables
            if isinstance(vals, (list, tuple)):
                return list(vals)
            elif hasattr(vals, "__iter__"):
                return list(vals)
        except Exception:
            pass
    return [k for k in dir(obj) if not k.startswith("__")]

__all__ = [
    'ORCAFLEX_VARIABLE_MAP',
    'MATH_FUNCTIONS',
    '_find_xyz_triples',
    '_safe',
    '_looks_like_user_var',
    '_parse_search_terms',
    '_matches_terms',
    'get_object_available_vars',
]

