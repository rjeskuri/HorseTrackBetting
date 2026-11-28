"""Pre-registered studies on Japan's development data (Epic #72): each freezes its family in the
run registry before it scores anything."""


def source_family(family_id: str, source: str) -> str:
    """The family a run on the store copy `source` belongs to (#404, owner decision (a)): the JRA
    store 'jra' keeps `family_id`, any other copy is its own pre-registered family
    `<family_id>-<source>` (e.g. 'jravan'), so a frozen jra spec is never re-frozen."""
    return family_id if source == "jra" else f"{family_id}-{source}"
