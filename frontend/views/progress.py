"""Live step lists: what My Day is doing right now, in plain words."""

from __future__ import annotations

import html
import time

import streamlit as st

WORKING = ":material/more_horiz:"
DONE = ":material/check_circle:"
NOTE = ":material/info:"


class Steps:
    """A status box that adds one line per step and ticks it off when the step ends.

    `lines` keeps the finished wording so the same list can be shown again,
    collapsed, after Streamlit reruns the page.
    """

    def __init__(self, label: str) -> None:
        self.box = st.status(label, expanded=True)
        self.lines: list[tuple[str, str]] = []
        self._slot = None
        self._text = ""

    def start(self, text: str) -> None:
        self._close()
        # Write through the box, not `with self.box`: leaving that block marks the status complete.
        self._slot = self.box.empty()
        self._text = text
        self._slot.markdown(f"{WORKING} {text}…")

    def done(self, text: str | None = None, *, icon: str = DONE) -> None:
        wording = text or self._text
        if self._slot is None:
            self._slot = self.box.empty()
        self._slot.markdown(f"{icon} {wording}")
        self.lines.append((icon, wording))
        self._slot = None
        self._text = ""

    def note(self, text: str) -> None:
        self._close()
        self.done(text, icon=NOTE)

    def finish(self, label: str, *, ok: bool = True) -> None:
        self._close()
        self.box.update(label=label, state="complete" if ok else "error", expanded=False)

    def _close(self) -> None:
        if self._slot is not None:
            self.done()


def replay(label: str, lines: list[tuple[str, str]]) -> None:
    """The finished step list, folded away, for every rerun after the live one."""
    if not lines:
        return
    with st.status(label, state="complete", expanded=False):
        for icon, text in lines:
            st.markdown(f"{icon} {text}")


def type_out(slot, text: str, *, css_class: str = "answer-line", delay: float = 0.035) -> None:
    """Write a sentence into `slot` a few words at a time."""
    words = text.split(" ")
    shown: list[str] = []
    for index, word in enumerate(words):
        shown.append(word)
        if index % 2 == 1 or index == len(words) - 1:
            cursor = "" if index == len(words) - 1 else " ▍"
            slot.markdown(
                f'<p class="{css_class}">{html.escape(" ".join(shown))}{cursor}</p>',
                unsafe_allow_html=True,
            )
            time.sleep(delay)


def plural(count: int, one: str, many: str | None = None) -> str:
    return f"{count} {one if count == 1 else (many or one + 's')}"
