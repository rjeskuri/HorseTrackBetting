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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from horsetrack.core import paths
from horsetrack.countries.japan.models import nowcast as nowcast_c1
from horsetrack.countries.japan.models.v0.calibration import WinCalibration
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
    if texts["residual"] is None:
        return ShadowModels(win=win, ordering=ordering, nowcast=nowcast, ids=ids)
    residual = ResidualFit.from_json(texts["residual"])
    if residual.base != win:
        raise ValueError(
            f"{pointer}: the residual {residual.fit_id} is built on calibration "
            f"{residual.base.fit_id}, not on the named win fit {win.fit_id}"
        )
    ids["residual"] = residual.fit_id
    return ShadowModels(win=residual, ordering=ordering, nowcast=nowcast, ids=ids)
