# -*- coding: utf-8 -*-

"""
Risk-strip mapping of the 11-item clinical simple score

Validation cohort

Rows:
    clinical_simple_score = 0–11

Columns:
    Score
    Deterioration/death, 24–48 h
    Deterioration/death, 48–72 h
    Mean future organ-failure burden, 24–48 h
    Mean future organ-failure burden, 48–72 h

Deterioration/death:
    trajectory state = NEW_ORGAN_FAILURE or DEATH

Future burden:
    burden5_official_24_48
    burden5_official_48_72

All field names and outcome definitions are retained from the original
fig7c(2).py. No new outcome fields are created.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize

from scipy.stats import sem


# ============================================================
# 0. PATHS
# ============================================================

BASE = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2"
)


SCORE_FILE = (
    BASE
    / "simple_score_clinical"
    / "validation_clinical_simple_score.csv"
)


FULL_MATRIX = (
    BASE
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)


OUTDIR = (
    BASE
    / "fig7c(2)"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. COLUMN SETTINGS
# ============================================================

ID_COL = "stay_id"

SCORE_COL = "clinical_simple_score"


TRAJECTORY_24 = "trajectory_state_24_48"

TRAJECTORY_48 = "trajectory_state_48_72"


BURDEN_24 = "burden5_official_24_48"

BURDEN_48 = "burden5_official_48_72"


# ============================================================
# 2. COLORS
# ============================================================

# Observed event risk
RISK_COLOR_24 = "#D95F02"
RISK_COLOR_48 = "#B2182B"

# Mean burden
BURDEN_COLOR_24 = "#4393C3"
BURDEN_COLOR_48 = "#2166AC"


# ============================================================
# 3. FIGURE STYLE
# ============================================================

plt.rcParams.update({

    "font.family":
        "sans-serif",

    "font.sans-serif":
        [
            "Arial",
            "Helvetica",
            "DejaVu Sans"
        ],

    "font.size":
        8,

    "axes.labelsize":
        8,

    "xtick.labelsize":
        7,

    "ytick.labelsize":
        7,

    "legend.fontsize":
        7,

    "axes.linewidth":
        0.7,

    "figure.dpi":
        300,

    "savefig.dpi":
        600,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

    "svg.fonttype":
        "none",
})


# ============================================================
# 4. MAP TRAJECTORY TO DETERIORATION / DEATH EVENT
# ============================================================

def trajectory_to_event(x):

    """
    1:
        NEW_ORGAN_FAILURE
        DEATH

    0:
        RECOVERY
        ICU_DISCHARGE
        STABLE/PERSISTENT

    NaN:
        unknown / missing
    """

    if pd.isna(x):

        return np.nan


    s = (
        str(x)
        .strip()
        .upper()
        .replace(" ", "_")
    )


    # --------------------------------------------------------
    # Event
    # --------------------------------------------------------

    if (
        "NEW_ORGAN_FAILURE" in s
        or
        "DEATH" in s
        or
        "DEAD" in s
        or
        "DIED" in s
        or
        "DETERIOR" in s
    ):

        return 1


    # --------------------------------------------------------
    # Non-event
    # --------------------------------------------------------

    if (
        "RECOVERY" in s
        or
        "RECOVER" in s
        or
        "ICU_DISCHARGE" in s
        or
        "DISCHARGE" in s
        or
        "STABLE" in s
        or
        "PERSISTENT" in s
    ):

        return 0


    return np.nan


# ============================================================
# 5. WILSON 95% CI FOR EVENT RATE
# ============================================================

def wilson_ci(
    event_count,
    n,
    z=1.96
):

    if n == 0:

        return (
            np.nan,
            np.nan
        )


    p = (
        event_count
        /
        n
    )


    denominator = (
        1
        +
        z ** 2
        /
        n
    )


    centre = (

        p

        +

        z ** 2
        /
        (
            2 * n
        )

    ) / denominator


    half_width = (

        z

        *

        np.sqrt(

            (
                p
                *
                (
                    1
                    -
                    p
                )
                /
                n
            )

            +

            (
                z ** 2
                /
                (
                    4
                    *
                    n ** 2
                )
            )
        )

    ) / denominator


    return (

        max(
            0,
            centre
            -
            half_width
        ),

        min(
            1,
            centre
            +
            half_width
        )
    )


# ============================================================
# 6. READ DATA
# ============================================================

print(
    "=" * 80
)

print(
    "SCORE-TO-RISK MAPPING"
)

print(
    "=" * 80
)


score = pd.read_csv(
    SCORE_FILE
)


full = pd.read_csv(
    FULL_MATRIX,
    usecols=[
        ID_COL,
        TRAJECTORY_24,
        TRAJECTORY_48,
        BURDEN_24,
        BURDEN_48
    ]
)


# ============================================================
# 7. CHECK DUPLICATES
# ============================================================

if score[
    ID_COL
].duplicated().any():

    raise ValueError(
        "Duplicated stay_id in score file."
    )


if full[
    ID_COL
].duplicated().any():

    raise ValueError(
        "Duplicated stay_id in full matrix."
    )


# ============================================================
# 8. MERGE
# ============================================================

df = (

    score[
        [
            ID_COL,
            SCORE_COL
        ]
    ]

    .merge(

        full,

        on=ID_COL,

        how="left",

        validate="one_to_one"
    )
)


# ============================================================
# 9. NUMERIC
# ============================================================

df[
    SCORE_COL
] = pd.to_numeric(
    df[
        SCORE_COL
    ],
    errors="coerce"
)


df[
    BURDEN_24
] = pd.to_numeric(
    df[
        BURDEN_24
    ],
    errors="coerce"
)


df[
    BURDEN_48
] = pd.to_numeric(
    df[
        BURDEN_48
    ],
    errors="coerce"
)


# ============================================================
# 10. CREATE EVENT VARIABLES
# ============================================================

df[
    "event_24_48"
] = df[
    TRAJECTORY_24
].map(
    trajectory_to_event
)


df[
    "event_48_72"
] = df[
    TRAJECTORY_48
].map(
    trajectory_to_event
)


# ============================================================
# 11. QC
# ============================================================

print(
    f"\nValidation patients = {len(df):,}"
)


print(
    "\n24–48 h event distribution:"
)

print(
    df[
        "event_24_48"
    ].value_counts(
        dropna=False
    )
)


print(
    "\n48–72 h event distribution:"
)

print(
    df[
        "event_48_72"
    ].value_counts(
        dropna=False
    )
)


print(
    "\nSimple score distribution:"
)

print(
    df[
        SCORE_COL
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# ============================================================
# 12. SCORE RANGE
# ============================================================

# 理论范围0–11
SCORE_LEVELS = list(
    range(
        0,
        12
    )
)


# ============================================================
# 13. SUMMARIZE BY SCORE
# ============================================================

rows = []


for score_value in SCORE_LEVELS:

    sub = df[
        df[
            SCORE_COL
        ]
        ==
        score_value
    ].copy()


    # --------------------------------------------------------
    # 24–48 h event
    # --------------------------------------------------------

    sub_event24 = sub[
        "event_24_48"
    ].dropna()


    n_event24 = len(
        sub_event24
    )


    event_count24 = int(
        sub_event24.sum()
    )


    event_rate24 = (

        event_count24
        /
        n_event24

        if n_event24 > 0

        else np.nan
    )


    lower24, upper24 = wilson_ci(
        event_count24,
        n_event24
    )


    # --------------------------------------------------------
    # 48–72 h event
    # --------------------------------------------------------

    sub_event48 = sub[
        "event_48_72"
    ].dropna()


    n_event48 = len(
        sub_event48
    )


    event_count48 = int(
        sub_event48.sum()
    )


    event_rate48 = (

        event_count48
        /
        n_event48

        if n_event48 > 0

        else np.nan
    )


    lower48, upper48 = wilson_ci(
        event_count48,
        n_event48
    )


    # --------------------------------------------------------
    # 24–48 h burden
    # --------------------------------------------------------

    burden24 = sub[
        BURDEN_24
    ].dropna()


    mean_burden24 = (

        burden24.mean()

        if len(
            burden24
        ) > 0

        else np.nan
    )


    sem_burden24 = (

        sem(
            burden24,
            nan_policy="omit"
        )

        if len(
            burden24
        ) > 1

        else np.nan
    )


    # --------------------------------------------------------
    # 48–72 h burden
    # --------------------------------------------------------

    burden48 = sub[
        BURDEN_48
    ].dropna()


    mean_burden48 = (

        burden48.mean()

        if len(
            burden48
        ) > 0

        else np.nan
    )


    sem_burden48 = (

        sem(
            burden48,
            nan_policy="omit"
        )

        if len(
            burden48
        ) > 1

        else np.nan
    )


    rows.append(
        {

            "simple_score":
                score_value,

            "n_total":
                len(
                    sub
                ),

            # event risk
            "n_event_available_24_48":
                n_event24,

            "event_count_24_48":
                event_count24,

            "event_rate_24_48":
                event_rate24,

            "event_lower95_24_48":
                lower24,

            "event_upper95_24_48":
                upper24,

            "n_event_available_48_72":
                n_event48,

            "event_count_48_72":
                event_count48,

            "event_rate_48_72":
                event_rate48,

            "event_lower95_48_72":
                lower48,

            "event_upper95_48_72":
                upper48,

            # burden
            "n_burden_24_48":
                len(
                    burden24
                ),

            "mean_burden_24_48":
                mean_burden24,

            "sem_burden_24_48":
                sem_burden24,

            "n_burden_48_72":
                len(
                    burden48
                ),

            "mean_burden_48_72":
                mean_burden48,

            "sem_burden_48_72":
                sem_burden48,
        }
    )


summary = pd.DataFrame(
    rows
)


# ============================================================
# 14. REMOVE COMPLETELY EMPTY SCORE LEVELS
# ============================================================

summary_plot = summary[
    summary[
        "n_total"
    ]
    >
    0
].copy()


print(
    "\n"
    + "=" * 80
)

print(
    "SCORE-TO-RISK TABLE"
)

print(
    "=" * 80
)


print(
    summary[
        [
            "simple_score",
            "n_total",
            "event_rate_24_48",
            "event_rate_48_72",
            "mean_burden_24_48",
            "mean_burden_48_72"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE TABLE
# ============================================================

summary.to_csv(

    OUTDIR
    /
    "clinical_simple_score_absolute_risk_mapping.csv",

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 16. RISK-STRIP FIGURE
# ============================================================

# ------------------------------------------------------------------
# IMPORTANT:
# The summary columns below are exactly those created above from the
# original fig7c(2).py definitions. No new outcome field names are used.
#
# Color scaling:
#   event_rate_*        : theoretical range 0–1
#   mean_burden_*       : theoretical range 0–5
#
# For visualization only, burden is divided by 5 so that all four
# outcome columns use a common 0–1 color-intensity scale. The text
# printed inside each cell remains on the ORIGINAL scale.
# ------------------------------------------------------------------

plot_df = summary_plot.copy()


# ============================================================
# 17. COLOR SCALE
# ============================================================

# Light -> dark sequential red/orange palette.
# Darker cells indicate a higher observed risk / burden.
RISK_CMAP = plt.get_cmap(
    "magma_r"
)

RISK_NORM = Normalize(
    vmin=0.0,
    vmax=1.0
)


# ============================================================
# 18. TABLE STRUCTURE
# ============================================================

# Five visible columns:
#   1. score
#   2. deterioration/death 24–48 h
#   3. deterioration/death 48–72 h
#   4. mean burden 24–48 h
#   5. mean burden 48–72 h

# IMPORTANT:
# Rectangle width == height in DATA coordinates is not enough to make
# a visually square tile unless the axes use an equal aspect ratio.
# Therefore the four outcome columns use TILE_SIZE for both width and
# row height, and ax.set_aspect("equal") is applied below.

TILE_SIZE = 0.72
ROW_HEIGHT = TILE_SIZE

COLUMN_WIDTHS = [
    1.02,      # Score column: allowed to be wider than a square tile
    TILE_SIZE, # D/D 24–48: square
    TILE_SIZE, # D/D 48–72: square
    TILE_SIZE, # burden 24–48: square
    TILE_SIZE, # burden 48–72: square
]

X_EDGES = np.concatenate(
    (
        [0.0],
        np.cumsum(
            COLUMN_WIDTHS
        )
    )
)

TABLE_WIDTH = float(
    X_EDGES[-1]
)

N_ROWS = len(
    plot_df
)

TABLE_HEIGHT = N_ROWS * ROW_HEIGHT

# A 12-row square-tile matrix is naturally tall.  This figure ratio
# lets the equal-aspect axes use most of the canvas instead of squeezing
# the tiles into rectangles.
fig, ax = plt.subplots(
    figsize=(
        4.8,
        8.6
    )
)

# Space above the table for grouped column headers.
HEADER_SPACE = 1.38

ax.set_xlim(
    0,
    TABLE_WIDTH
)

ax.set_ylim(
    -HEADER_SPACE,
    TABLE_HEIGHT
)

ax.invert_yaxis()

# Critical: one x-data unit must equal one y-data unit on screen.
# Without this, equal Rectangle width/height still looks rectangular.
ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.axis("off")


# ============================================================
# 19. HELPER FUNCTIONS
# ============================================================

def cell_text_color(
    normalized_value
):

    """Choose white text on dark cells and dark text on light cells."""

    if pd.isna(
        normalized_value
    ):
        return "#555555"

    return (
        "white"
        if normalized_value >= 0.58
        else "#222222"
    )


def draw_cell(
    x0,
    y0,
    width,
    height,
    facecolor,
    edgecolor="white",
    linewidth=1.0,
):

    rect = Rectangle(
        (
            x0,
            y0
        ),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    ax.add_patch(
        rect
    )


# ============================================================
# 20. DRAW DATA ROWS
# ============================================================

for row_index, row in plot_df.reset_index(drop=True).iterrows():

    y0 = row_index * ROW_HEIGHT
    y_center = y0 + ROW_HEIGHT / 2

    score_value = int(
        row[
            "simple_score"
        ]
    )

    n_total = int(
        row[
            "n_total"
        ]
    )

    # --------------------------------------------------------
    # Column 1: simple score + n
    # --------------------------------------------------------

    draw_cell(
        X_EDGES[0],
        y0,
        COLUMN_WIDTHS[0],
        ROW_HEIGHT,
        facecolor="#F7F7F7",
        edgecolor="#E4E4E4",
        linewidth=0.8,
    )

    ax.text(
        X_EDGES[0]
        + COLUMN_WIDTHS[0] * 0.37,
        y_center - 0.085,
        f"{score_value}",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color="#222222",
    )

    ax.text(
        X_EDGES[0]
        + COLUMN_WIDTHS[0] * 0.69,
        y_center + 0.105,
        f"n={n_total:,}",
        ha="center",
        va="center",
        fontsize=5.8,
        color="#777777",
    )

    # --------------------------------------------------------
    # Exact summary values from the original code
    # --------------------------------------------------------

    event24 = row[
        "event_rate_24_48"
    ]

    event48 = row[
        "event_rate_48_72"
    ]

    burden24 = row[
        "mean_burden_24_48"
    ]

    burden48 = row[
        "mean_burden_48_72"
    ]

    # Only for COLOR scaling.
    # The displayed values below remain in their original units.
    normalized_values = [
        event24,
        event48,
        (
            burden24 / 5.0
            if pd.notna(
                burden24
            )
            else np.nan
        ),
        (
            burden48 / 5.0
            if pd.notna(
                burden48
            )
            else np.nan
        ),
    ]

    display_values = [
        (
            f"{event24 * 100:.1f}%"
            if pd.notna(
                event24
            )
            else "NA"
        ),
        (
            f"{event48 * 100:.1f}%"
            if pd.notna(
                event48
            )
            else "NA"
        ),
        (
            f"{burden24:.2f}"
            if pd.notna(
                burden24
            )
            else "NA"
        ),
        (
            f"{burden48:.2f}"
            if pd.notna(
                burden48
            )
            else "NA"
        ),
    ]

    # --------------------------------------------------------
    # Columns 2–5
    # --------------------------------------------------------

    for j in range(
        4
    ):

        x0 = X_EDGES[
            j + 1
        ]

        width = COLUMN_WIDTHS[
            j + 1
        ]

        normalized_value = normalized_values[
            j
        ]

        if pd.isna(
            normalized_value
        ):

            facecolor = "#F2F2F2"

        else:

            # Keep the color mapping inside theoretical limits.
            normalized_value_for_color = float(
                np.clip(
                    normalized_value,
                    0.0,
                    1.0
                )
            )

            facecolor = RISK_CMAP(
                RISK_NORM(
                    normalized_value_for_color
                )
            )

        draw_cell(
            x0,
            y0,
            width,
            ROW_HEIGHT,
            facecolor=facecolor,
            edgecolor="white",
            linewidth=1.15,
        )

        ax.text(
            x0 + width / 2,
            y_center,
            display_values[
                j
            ],
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight=(
                "bold"
                if normalized_value is not np.nan
                and pd.notna(normalized_value)
                and normalized_value >= 0.65
                else "normal"
            ),
            color=cell_text_color(
                normalized_value
            ),
        )


# ============================================================
# 21. COLUMN HEADERS
# ============================================================

# Main header baseline positions.
SUBHEADER_Y = -0.31
GROUPHEADER_Y = -0.93

# Score header
ax.text(
    X_EDGES[0]
    + COLUMN_WIDTHS[0] / 2,
    SUBHEADER_Y,
    "Score",
    ha="center",
    va="center",
    fontsize=8.2,
    fontweight="bold",
    color="#222222",
)

# Group header: Deterioration/death
risk_group_center = (
    X_EDGES[1]
    +
    X_EDGES[3]
) / 2

ax.text(
    risk_group_center,
    GROUPHEADER_Y,
    "Deterioration/death",
    ha="center",
    va="center",
    fontsize=8.3,
    fontweight="bold",
    color="#222222",
)

# Group header: Mean burden
burden_group_center = (
    X_EDGES[3]
    +
    X_EDGES[5]
) / 2

ax.text(
    burden_group_center,
    GROUPHEADER_Y,
    "Mean future organ-failure burden",
    ha="center",
    va="center",
    fontsize=8.3,
    fontweight="bold",
    color="#222222",
)

# Subheaders
subheaders = [
    "24–48 h",
    "48–72 h",
    "24–48 h",
    "48–72 h",
]

for j, label in enumerate(
    subheaders
):

    x_center = (
        X_EDGES[
            j + 1
        ]
        +
        COLUMN_WIDTHS[
            j + 1
        ] / 2
    )

    ax.text(
        x_center,
        SUBHEADER_Y,
        label,
        ha="center",
        va="center",
        fontsize=7.5,
        color="#444444",
    )


# ============================================================
# 22. HEADER RULES AND GROUP DIVIDER
# ============================================================

# Top/bottom rules around headers
ax.plot(
    [
        0,
        TABLE_WIDTH
    ],
    [
        -0.63,
        -0.63
    ],
    color="#333333",
    linewidth=0.7,
    clip_on=False,
)

ax.plot(
    [
        0,
        TABLE_WIDTH
    ],
    [
        0,
        0
    ],
    color="#333333",
    linewidth=0.75,
    clip_on=False,
)

# A slightly stronger divider between event-risk and burden columns.
ax.plot(
    [
        X_EDGES[3],
        X_EDGES[3]
    ],
    [
        -1.12,
        TABLE_HEIGHT
    ],
    color="#D6D6D6",
    linewidth=1.0,
    clip_on=False,
)


# ============================================================
# 23. COLOR-INTENSITY KEY
# ============================================================

# Numeric values are printed in every cell, so this compact key is
# intentionally qualitative rather than a second full colorbar.
KEY_X0 = X_EDGES[1]
KEY_X1 = X_EDGES[5]
KEY_Y = TABLE_HEIGHT + 0.33

# Because y is inverted, place this visually below the table by
# extending the axis slightly.
ax.set_ylim(
    -HEADER_SPACE,
    TABLE_HEIGHT + 0.68
)
ax.invert_yaxis()

# Re-apply after the final y-limits are set.
ax.set_aspect(
    "equal",
    adjustable="box"
)

N_KEY = 100

for i in range(
    N_KEY
):

    frac0 = i / N_KEY
    frac1 = (i + 1) / N_KEY

    x0 = KEY_X0 + (
        KEY_X1 - KEY_X0
    ) * frac0

    width = (
        KEY_X1 - KEY_X0
    ) / N_KEY

    draw_cell(
        x0,
        KEY_Y,
        width,
        0.10,
        facecolor=RISK_CMAP(
            frac0
        ),
        edgecolor=RISK_CMAP(
            frac0
        ),
        linewidth=0,
    )

ax.text(
    KEY_X0,
    KEY_Y + 0.23,
    "Lower",
    ha="left",
    va="center",
    fontsize=6.2,
    color="#666666",
)

ax.text(
    KEY_X1,
    KEY_Y + 0.23,
    "Higher",
    ha="right",
    va="center",
    fontsize=6.2,
    color="#666666",
)

ax.text(
    (
        KEY_X0
        +
        KEY_X1
    ) / 2,
    KEY_Y + 0.23,
    "Outcome level",
    ha="center",
    va="center",
    fontsize=6.2,
    color="#666666",
)


# ============================================================
# 24. LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.055,
    right=0.985,
    bottom=0.035,
    top=0.985,
)


# ============================================================
# 25. SAVE
# ============================================================

PNG = (
    OUTDIR
    /
    "clinical_simple_score_risk_strip_square.png"
)

PDF = (
    OUTDIR
    /
    "clinical_simple_score_risk_strip_square.pdf"
)

SVG = (
    OUTDIR
    /
    "clinical_simple_score_risk_strip_square.svg"
)

fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    PDF,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    SVG,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

plt.close(
    fig
)


print(
    "\n"
    + "=" * 80
)

print(
    "FINISHED"
)

print(
    "=" * 80
)

print(
    "\nOutput table:"
)

print(
    OUTDIR
    /
    "clinical_simple_score_absolute_risk_mapping.csv"
)

print(
    "\nFigure:"
)

print(
    PNG
)

print(
    PDF
)

print(
    SVG
)
