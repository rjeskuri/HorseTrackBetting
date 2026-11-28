"""The current v0 fits for the shadow runner (ticket #120).

`data_dir()/models/japan/v0/current.json` names the fit files, {"win": file, "ordering": file,
"nowcast": file}; a relative name resolves against the pointer's own directory. Each file is loaded
through its class's `from_json` (WinCalibration #114, OrderingFit #115, NowcastV0 #116); a nowcast
file whose "model" is "nowcast_c1" is #85's structural nowcast, loaded behind the same draws
interface (`models.nowcast.ShadowNowcast`), so the pointer switches nowcasts. A missing
pointer, win file or nowcast file gives None and one warning, so the runner logs boards and flags
without values; a pointer without an ordering fit gives `ordering=None`. A file that exists but
does not parse raises: a broken deploy is not hidden as a day of null values.

An optional "residual" entry (#125) names a ResidualFit file (base embedded). When present, `win`
is that residual in place of the plain calibration and `ids` gains "residual"; "win" stays the
calibration's fit_id. A named residual file that is missing gives None and one warning like the
others, and one built on another calibration than the named win fit raises.

An optional "member" entry (#717) names the shadow page's horse member per segment,
{"DEBUT": {"fit": file, "k": number}, "MAIDEN": {...}}: a MemberFit file (models/v0/member.py) and
the p/q threshold k the leaderboard ranks best, loaded into `ShadowModels.member` as
{segment: MemberEntry(fit, k)}. No entry (absent or null) leaves it empty, the market only. A named
member file that is missing drops that segment alone, with one warning; one that exists but does
not parse raises, and so do a key other than DEBUT or MAIDEN, a fit of another segment than its
key, and a k that is missing, not a number, not finite or not above 0. The member is display only:
it is not in `ids` and changes no other fit.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from horsetrack.core import paths
from horsetrack.countries.japan.models import nowcast as nowcast_c1
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
from horsetrack.countries.japan.models.v0.member import SEGMENTS, MemberEntry, MemberFit
from horsetrack.countries.japan.models.v0.nowcast import NowcastV0
from horsetrack.countries.japan.models.v0.ordering import OrderingFit
from horsetrack.countries.japan.models.v0.residual import ResidualFit
from horsetrack.countries.japan.models.v0.shadow_values import ShadowModels

log = logging.getLogger(__name__)
OPTIONAL = ("ordering", "residual")


def current_pointer() -> Path:
    """`data_dir()/models/japan/v0/current.json`, the data dir read on every call."""
    return paths.data_dir() / "models" / "japan" / "v0" / "current.json"


def _load_nowcast(text: str):
    """#85's structural nowcast when the file names it, else #116's drift (NowcastV0)."""
    try:
        model = json.loads(text).get("model")
    except (ValueError, AttributeError):
        model = None
    if model == nowcast_c1.MODEL_NAME:
        return nowcast_c1.load_shadow_nowcast(text)
    return NowcastV0.from_json(text)


def _load_member(pointer: Path, entry: object) -> dict[str, MemberEntry]:
    """The pointer's "member" entry as {segment: MemberEntry}; see the module doc."""
    if entry is None:
        return {}
    if not isinstance(entry, dict):
        raise ValueError(f"{pointer}: 'member' must map DEBUT / MAIDEN to {{fit, k}}")
    out: dict[str, MemberEntry] = {}
    for segment, spec in entry.items():
        if segment not in SEGMENTS:
            raise ValueError(f"{pointer}: member segment {segment!r} is not one of {SEGMENTS}")
        if not isinstance(spec, dict) or not isinstance(spec.get("fit"), str):
            raise ValueError(f"{pointer}: member {segment} must be {{fit: file, k: number}}")
        k = spec.get("k")
        if isinstance(k, bool) or not isinstance(k, int | float) or not math.isfinite(k) or k <= 0:
            raise ValueError(f"{pointer}: member {segment} k {k!r} is not a number above 0")
        name = spec["fit"]
        path = Path(name) if Path(name).is_absolute() else pointer.parent / name
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("shadow models: the %s member fit %s is missing; not shown", segment, path)
            continue
        fit = MemberFit.from_json(text)
        if fit.segment != segment:
            raise ValueError(f"{pointer}: the {segment} member {path} is a {fit.segment} fit")
        out[segment] = MemberEntry(fit=fit, k=float(k))
    return out


def load_shadow_models(pointer: Path | None = None) -> ShadowModels | None:
    """The fits the pointer names, or None (with one warning) when the pointer or a needed file is
    missing."""
    pointer = Path(pointer) if pointer is not None else current_pointer()
    try:
        names = json.loads(pointer.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.warning("shadow models: no pointer at %s; values are left null", pointer)
        return None
    if not isinstance(names, dict):
        raise ValueError(f"{pointer}: expected a JSON object naming the fit files")

    texts: dict[str, str | None] = {}
    for kind in ("win", "ordering", "nowcast", "residual"):
        name = names.get(kind)
        if name is None:
            if kind in OPTIONAL:
                texts[kind] = None
                continue
            log.warning("shadow models: %s names no %s fit; values are left null", pointer, kind)
            return None
        path = Path(name) if Path(name).is_absolute() else pointer.parent / name
        try:
            texts[kind] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.warning("shadow models: the %s fit %s is missing; values are left null", kind, path)
            return None

    win = WinCalibration.from_json(texts["win"])
    nowcast = _load_nowcast(texts["nowcast"])
    ordering = None if texts["ordering"] is None else OrderingFit.from_json(texts["ordering"])
    ids = {"win": win.fit_id}
    if ordering is not None:
        ids["ordering"] = ordering.fit_id
    ids["nowcast"] = nowcast.fit_id
    member = _load_member(pointer, names.get("member"))
    if texts["residual"] is None:
        return ShadowModels(win=win, ordering=ordering, nowcast=nowcast, ids=ids, member=member)
    residual = ResidualFit.from_json(texts["residual"])
    if residual.base != win:
        raise ValueError(
            f"{pointer}: the residual {residual.fit_id} is built on calibration "
            f"{residual.base.fit_id}, not on the named win fit {win.fit_id}"
        )
    ids["residual"] = residual.fit_id
    return ShadowModels(win=residual, ordering=ordering, nowcast=nowcast, ids=ids, member=member)
