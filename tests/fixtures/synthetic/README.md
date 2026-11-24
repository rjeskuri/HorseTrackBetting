# Synthetic fixtures

The only test fixtures committed to this public repository. Every file here is hand-built,
minimal and made up. None is a scraped or licensed page or data file (JRDB, netkeiba, JRA,
HKJC), or an excerpt of one, and none holds a real person's name or contact details.

## Building a fixture

- Write it by hand with only the structure and fields the test reads: the tags, attributes and
  layout of the source page, filled with invented values (horse names, odds, times).
- Every `.html` or `.htm` file starts with the marker, as its very first bytes (no byte-order
  mark, blank line or doctype before it): `<!-- synthetic: <what it imitates> -->`, for example
  `<!-- synthetic: JRA race card, one race, three runners -->`.
- Keep files small: a race card needs a few runners, not a full field, and no file may exceed
  256 KB.
- Name files after what they imitate (`jra_race_card.html`), never after a source's own data
  file names, such as JRDB's three capital letters, six digits and `.txt`.

## Real pages

Real pages stay outside the tree, under `$HORSETRACK_DATA_DIR/fixtures`. A test that needs one
carries the `local` marker.

## The guard

`scripts/check_public_tree.py` refuses any file under `tests/fixtures/` outside this folder and
any `.html` or `.htm` here without the marker. `tests/test_public_tree.py` runs it on every pull
request.
