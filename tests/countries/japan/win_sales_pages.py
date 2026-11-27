"""Synthetic JRADB 単勝・複勝 odds pages carrying the 発売票数 block, for the Win-sales tests
(ticket #278).

`sales_page` builds the structure of tests/fixtures/synthetic/jra_win_sales_*.html from values: the
race header (date line, the 発走時刻 cell, the race-number image), the refresh line (a clock such
as '12時10分現在', or '最終オッズ'), a two-runner odds table, the votes block and an optional decoy
row outside it. Votes are printed exactly as given (text), so a test can print junk. Pages come
back as cp932 bytes, as the site sends them. Every value is invented.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import jradb_pages as jp


def sales_page(
    day: date | None,
    meeting: str,
    race_no: int | None,
    *,
    state: str | None,
    post: str | None,
    votes: Sequence[tuple[str, str]] | None,
    decoy: str | None = None,
) -> bytes:
    """A 単勝・複勝 page. `day` None leaves the date line out, `race_no` None the race number,
    `state` None the refresh line, `post` None the 発走時刻 cell, `votes` None the whole 発売票数
    block. `decoy` prints a 単勝 row with that text in a table outside the votes block."""
    date_cell = "" if day is None else f'<div class="cell date">{jp.date_line(day, meeting)}</div>'
    post_cell = "" if post is None else f'<div class="cell time"> 発走時刻：<strong>{post}</strong>'
    post_cell += "" if post is None else "</div>"
    number = (
        ""
        if race_no is None
        else f'<div class="race_number"><img src="n.png" alt="{race_no}レース"></div>'
    )
    refresh = (
        ""
        if state is None
        else f'<div class="refresh_line"><div class="cell time"><strong>{state}</strong></div>'
        "</div>"
    )
    header = (
        '<div class="race_header"><div class="left">'
        f'<div class="date_line"><div class="inner">{date_cell}{post_cell}</div></div>'
        f'<div class="race_title"><div class="inner"><div class="num">{number}</div></div></div>'
        f'</div><div class="right set_live_btn">{refresh}</div></div>'
    )
    odds = (
        '<div id="odds_list"><table><tbody>'
        '<tr><td class="waku"><img alt="w" src="w.png"></td><td class="num">1</td>'
        '<td class="horse"><a href="#">h1</a></td><td class="odds_tan"><strong>4.2</strong></td>'
        '<td class="odds_fuku"><span class="inner"><span class="min">1.5</span>'
        '<span class="cap">-</span><span class="max">2.1</span></span></td></tr>'
        "</tbody></table></div>"
    )
    block = ""
    if votes is not None:
        rows = "".join(f'<tr><th scope="row">{label}</th><td>{n}</td></tr>' for label, n in votes)
        block = (
            '<div class="mt20" id="votes"><table class="basic narrow auto">'
            '<caption class="simple title-s"><div class="inner"><div class="main">発売票数</div>'
            f"</div></caption><tbody>{rows}</tbody></table></div>"
        )
    trap = (
        ""
        if decoy is None
        else '<div class="caution"><table><tbody>'
        f"<tr><th>単勝</th><td>{decoy}</td></tr></tbody></table></div>"
    )
    return jp.page(header + odds + trap + block + trap)
