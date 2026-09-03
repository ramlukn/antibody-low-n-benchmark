"""Load the two TDC antibody developability datasets into tidy frames.

TDC hands back a single ``Antibody`` column holding a *string* that reprs a
two-element sequence container.  Annoyingly the two datasets use different
reprs (a Python list for SAbDab_Chen, a NumPy array for TAP), so we pull the
chains out with a regex over runs of amino-acid letters rather than trusting
``ast.literal_eval``.

Both datasets store the heavy chain first and the light chain second.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import AMINO_ACIDS, CHEN, RAW, TAP, TAP_TARGETS

_CHAIN_RE = re.compile(rf"[{AMINO_ACIDS}]{{20,}}")


def parse_chains(raw: str) -> tuple[str, str]:
    """Extract (heavy, light) from TDC's stringified sequence container."""
    chains = _CHAIN_RE.findall(raw)
    if len(chains) != 2:
        raise ValueError(f"expected 2 chains, found {len(chains)} in {raw[:80]!r}")
    return chains[0], chains[1]


def load_chen() -> pd.DataFrame:
    """SAbDab_Chen: 2,409 antibodies, binary developability label.

    Returns columns: id, heavy, light, y.
    """
    from tdc.single_pred import Develop

    df = Develop(name=CHEN, path=str(RAW)).get_data()
    chains = df["Antibody"].map(parse_chains)
    out = pd.DataFrame(
        {
            "id": df["Antibody_ID"].astype(str),
            "heavy": chains.map(lambda t: t[0]),
            "light": chains.map(lambda t: t[1]),
            "y": df["Y"].astype(int),
        }
    )
    return out.reset_index(drop=True)


def load_tap() -> pd.DataFrame:
    """TAP: 241 antibodies with five continuous developability metrics.

    TDC exposes each metric as a separate "label name" over the same
    sequences, so we load all five and join them into one wide frame.
    Returns columns: id, heavy, light, and one column per TAP target.
    """
    from tdc.single_pred import Develop

    frames = []
    for target in TAP_TARGETS:
        df = Develop(name=TAP, path=str(RAW), label_name=target).get_data()
        frames.append(df.set_index("Antibody_ID")["Y"].rename(target))

    base = Develop(name=TAP, path=str(RAW), label_name=TAP_TARGETS[0]).get_data()
    chains = base["Antibody"].map(parse_chains)
    out = pd.DataFrame(
        {
            "id": base["Antibody_ID"].astype(str),
            "heavy": chains.map(lambda t: t[0]),
            "light": chains.map(lambda t: t[1]),
        }
    ).set_index("id")
    out = out.join(pd.concat(frames, axis=1))
    return out.reset_index()


def load_dataset(name: str) -> pd.DataFrame:
    if name == CHEN:
        return load_chen()
    if name == TAP:
        return load_tap()
    raise KeyError(name)


def all_sequences(frames: list[pd.DataFrame]) -> list[str]:
    """Unique heavy/light chains across datasets, for one-pass embedding."""
    seen: dict[str, None] = {}
    for df in frames:
        for col in ("heavy", "light"):
            for s in df[col]:
                seen.setdefault(s, None)
    return list(seen)
