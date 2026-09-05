import random

from bridge_extractor_v2 import (
    extract_bridge_candidates_v2,
)


SEED = 99


def run_case(name, tools, expected):
    selected = extract_bridge_candidates_v2(
        tools,
        top_k=3,
    )

    print("=" * 72)
    print(name)
    print("=" * 72)

    for i, tool in enumerate(tools, 1):
        print(f"\nTOOL {i}")
        print(tool)

    print()
    print("EXPECTED:", expected)
    print("SELECTED:", selected)
    print(
        "EXPECTED SELECTED:",
        expected in selected,
    )
    print()


def main():
    random.seed(SEED)

    # ---------------------------------------------------------
    # A: Genuine cross-tool bridge
    # ---------------------------------------------------------

    tools_a = [
        """
Observed req-0111
Observed req-0222
Current trace req-0184
""",
        """
Observed req-0333
Observed req-0444
Follow-up trace req-0184
""",
    ]

    run_case(
        "A — TRUE CROSS-TOOL BRIDGE",
        tools_a,
        "req-0184",
    )

    # ---------------------------------------------------------
    # B: False repeated hypothesis.
    #
    # req-0291 repeats across tools, but req-0184 has
    # explicit root-cause evidence.
    # ---------------------------------------------------------

    tools_b = [
        """
Initial suspicion req-0291
Observed req-0111
Root cause request: req-0184
""",
        """
Follow-up suspicion req-0291
Observed req-0333
""",
    ]

    run_case(
        "B — FALSE CROSS-TOOL BRIDGE",
        tools_b,
        "req-0184",
    )

    # ---------------------------------------------------------
    # C: Several cross-tool repeated candidates.
    # Only req-0184 receives explicit causal evidence.
    # ---------------------------------------------------------

    tools_c = [
        """
Observed req-0101
Observed req-0202
Observed req-0184
""",
        """
Observed req-0101
Observed req-0202
Affected request: req-0184
""",
    ]

    run_case(
        "C — MULTIPLE CROSS-TOOL BRIDGES",
        tools_c,
        "req-0184",
    )


if __name__ == "__main__":
    main()
