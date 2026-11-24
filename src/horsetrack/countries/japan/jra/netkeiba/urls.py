"""netkeiba race ids (ported from HKTrack's `sources/jra/urls.py`). Pure functions.

A race id is 12 ASCII digits: YYYY + venue number (01-10 for JRA) + meeting (kai) + day + race
number. The same pages also link local-racing (NAR) ids with other venue numbers and overseas ids
with letters; only JRA ids pass `jra_race_id`.
"""

from __future__ import annotations

import re

# [0-9], not \d: \d also matches full-width digits
_RACE_ID = re.compile(r"[0-9]{4}(0[1-9]|10)[0-9]{6}")


def jra_race_id(value: object) -> str | None:
    """`value` if it is a JRA race id, else None."""
    return value if isinstance(value, str) and _RACE_ID.fullmatch(value) else None
