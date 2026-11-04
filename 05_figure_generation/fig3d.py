# -*- coding: utf-8 -*-
"""
Figure 3d
Risk-score-associated ORs for NEW target-organ failure,
related to PCC and OFTP in the CURRENT validation cohort.

Current analysis
----------------
Population:
    Fixed validation cohort, locked to the common longitudinal subset
    with both 24–48 h and 48–72 h official burden available.
    Expected n = 6,246.

Baseline:
    0–6 h

Future:
    24–48 h

Organs:
    Respiratory, Coagulation, Liver, Cardiovascular, Renal
    (CNS excluded)

PCC:
    corr(i_0-6h, j_0-6h)

OFTP:
    P(
        j_24-48h = 1
        |
        i_0-6h = 1,
        j_0-6h = 0
    )

Risk-score-associated OR for each ordered pair i != j:
    Restrict to patients with:
        i_0-6h = 1
        j_0-6h = 0
        j_24-48h observed

    Fit:
        new target failure j_24-48h
        ~ 1-SD higher risk_score

    The returned OR therefore quantifies risk-score-associated
    new target-organ failure within the same source-positive /
    target-at-risk population used by OFTP.

Important
---------
- Missing organ status remains NaN.
- Diagonal/self pairs are undefined and left blank.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, NullFormatter

warnings.filterwarnings("ignore")


# =========================================================
# 0. 路径配置
# =========================================================

BASE = Path(
    r"D:\博士\博士代码\data_analysis\formal_v2"
)

ORGAN_FILE = (
    BASE
    / "feature_matrix_full"
    / "feature_matrix_full_0_6h.csv"
)

RISK_CANDIDATES = [
    BASE / "nipals_model_save" / "validation_24_48_scored.csv",
    BASE / "nipals_model_save" / "validation_baseline_scored.csv",
    BASE / "nipals_outputs_new" / "validation_baseline_scored.csv",
]

OUTDIR = (
    BASE
    / "fig3d"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


def resolve_risk_file():
    for p in RISK_CANDIDATES:
        if p.exists():
            header = pd.read_csv(
                p,
                nrows=0
            ).columns

            if (
                "stay_id" in header
                and
                "risk_score" in header
            ):
                return p

    raise FileNotFoundError(
        "未找到同时包含 stay_id 和 risk_score 的当前 validation scored 文件。已检查：\n"
        + "\n".join(str(p) for p in RISK_CANDIDATES)
    )


RISK_FILE = resolve_risk_file()


# =========================================================
# 1. 列名映射
# =========================================================

COLUMN_MAP = {
    "stay_id": "stay_id",
    "risk_score": "risk_score",

    "resp_base": "resp_fail_6h",
    "coag_base": "coag_fail_6h",
    "liver_base": "liver_fail_6h",
    "cardio_base": "cardio_fail_6h",
    "renal_base": "renal_fail_6h",

    "resp_follow": "resp_failure_24_48",
    "coag_follow": "coag_failure_24_48",
    "liver_follow": "liver_failure_24_48",
    "cardio_follow": "cardio_failure_24_48",
    "renal_follow": "renal_failure_24_48",
}

BURDEN_24_48 = "burden5_official_24_48"
BURDEN_48_72 = "burden5_official_48_72"


# =========================================================
# 2. 基本设置
# =========================================================

ORGANS = [
    "resp",
    "coag",
    "liver",
    "cardio",
    "renal",
]

ORGAN_LABELS = {
    "resp": "Resp",
    "coag": "Coag",
    "liver": "Liver",
    "cardio": "Cardio",
    "renal": "Renal",
}

N_BOOT = 1000
N_PERM = 10000
RANDOM_STATE = 42


mpl.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 6
plt.rcParams["axes.titlesize"] = 6
plt.rcParams["axes.labelsize"] = 6
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================
# 3. 工具函数
# =========================================================

def ensure_cols(df, cols, file_desc="data"):
    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_desc} 缺少以下字段: {missing}"
        )


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    sd = x.std(
        ddof=0
    )

    if (
        pd.isna(
            sd
        )
        or
        sd == 0
    ):
        return x * np.nan

    return (
        x
        -
        x.mean()
    ) / sd


def to_binary_preserve_na(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(
        s,
        errors="coerce"
    )

    out = pd.Series(
        np.nan,
        index=s.index,
        dtype=float
    )

    mask = s.notna()

    out.loc[
        mask
    ] = (
        s.loc[
            mask
        ]
        > 0
    ).astype(float)

    return out


def riskscore_or_within_transition_subgroup(
    y: pd.Series,
    x: pd.Series
) -> float:
    """
    Logistic regression:
        new target failure ~ z_risk_score

    The caller already restricts to:
        source failed at 0–6 h
        target unaffected at 0–6 h
        future target observed

    Returns:
        OR per 1-SD higher risk_score
    """

    tmp = pd.DataFrame({
        "y": y,
        "x": x
    }).dropna().copy()

    tmp = tmp[
        tmp[
            "y"
        ].isin(
            [
                0,
                1
            ]
        )
    ]

    if len(
        tmp
    ) < 20:
        return np.nan

    if tmp[
        "y"
    ].nunique() < 2:
        return np.nan

    X = tmp[
        [
            "x"
        ]
    ].values

    yv = tmp[
        "y"
    ].astype(
        int
    ).values

    try:

        model = LogisticRegression(
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            max_iter=2000
        )

        model.fit(
            X,
            yv
        )

        beta = model.coef_[
            0,
            0
        ]

        return float(
            np.exp(
                beta
            )
        )

    except Exception:

        return np.nan


def compute_pcc_and_oftp(df: pd.DataFrame):
    """
    PCC and OFTP in the SAME validation cohort.
    """

    # ---------- PCC ----------
    pcc_data = pd.DataFrame({
        organ:
            df[
                COLUMN_MAP[
                    f"{organ}_base"
                ]
            ]
        for organ in ORGANS
    })

    pcc = pcc_data[
        ORGANS
    ].corr(
        method="pearson"
    )

    np.fill_diagonal(
        pcc.values,
        np.nan
    )

    # ---------- OFTP ----------
    oftp = pd.DataFrame(
        np.nan,
        index=ORGANS,
        columns=ORGANS,
        dtype=float
    )

    oftp_num = pd.DataFrame(
        0,
        index=ORGANS,
        columns=ORGANS,
        dtype=int
    )

    oftp_den = pd.DataFrame(
        0,
        index=ORGANS,
        columns=ORGANS,
        dtype=int
    )

    for source in ORGANS:

        source_col = COLUMN_MAP[
            f"{source}_base"
        ]

        for target in ORGANS:

            if source == target:
                continue

            target_base_col = COLUMN_MAP[
                f"{target}_base"
            ]

            target_follow_col = COLUMN_MAP[
                f"{target}_follow"
            ]

            at_risk = (
                (df[source_col] == 1)
                &
                (df[target_base_col] == 0)
                &
                df[target_follow_col].notna()
            )

            denom = int(
                at_risk.sum()
            )

            numer = int(
                (
                    at_risk
                    &
                    (df[target_follow_col] == 1)
                ).sum()
            )

            oftp_den.loc[
                source,
                target
            ] = denom

            oftp_num.loc[
                source,
                target
            ] = numer

            if denom > 0:

                oftp.loc[
                    source,
                    target
                ] = (
                    numer
                    /
                    denom
                )

    return (
        pcc,
        oftp,
        oftp_num,
        oftp_den
    )


def fit_line_with_bootstrap(
    x,
    y,
    n_boot=1000,
    seed=42
):
    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    ok = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    x = x[
        ok
    ]

    y = y[
        ok
    ]

    if len(
        x
    ) < 3:

        raise ValueError(
            "有效器官对不足，无法回归。"
        )

    lr = linregress(
        x,
        y
    )

    slope_obs = lr.slope
    intercept_obs = lr.intercept
    r2_obs = lr.rvalue ** 2

    rng = np.random.default_rng(
        seed
    )

    x_grid = np.linspace(
        x.min(),
        x.max(),
        200
    )

    y_boot = []

    n = len(
        x
    )

    for _ in range(
        n_boot
    ):

        idx = rng.choice(
            np.arange(
                n
            ),
            size=n,
            replace=True
        )

        xb = x[
            idx
        ]

        yb = y[
            idx
        ]

        try:

            lr_b = linregress(
                xb,
                yb
            )

            y_boot.append(
                lr_b.intercept
                +
                lr_b.slope
                *
                x_grid
            )

        except Exception:

            continue

    y_boot = np.asarray(
        y_boot
    )

    if y_boot.size == 0:

        ci_low = np.full_like(
            x_grid,
            np.nan,
            dtype=float
        )

        ci_high = np.full_like(
            x_grid,
            np.nan,
            dtype=float
        )

    else:

        ci_low = np.nanpercentile(
            y_boot,
            2.5,
            axis=0
        )

        ci_high = np.nanpercentile(
            y_boot,
            97.5,
            axis=0
        )

    y_hat = (
        intercept_obs
        +
        slope_obs
        *
        x_grid
    )

    perm_slopes = []

    for _ in range(
        N_PERM
    ):

        y_perm = rng.permutation(
            y
        )

        try:

            lr_p = linregress(
                x,
                y_perm
            )

            perm_slopes.append(
                lr_p.slope
            )

        except Exception:

            continue

    perm_slopes = np.asarray(
        perm_slopes
    )

    p_perm = (
        np.sum(
            np.abs(
                perm_slopes
            )
            >=
            abs(
                slope_obs
            )
        )
        +
        1
    ) / (
        len(
            perm_slopes
        )
        +
        1
    )

    return (
        slope_obs,
        intercept_obs,
        r2_obs,
        p_perm,
        x_grid,
        y_hat,
        ci_low,
        ci_high
    )


def pperm_text(p):
    if p < 1e-4:
        return "P$_{perm}$ < 0.0001"

    return (
        f"P$_{{perm}}$ = {p:.4f}"
    )


def make_annot_df(
    df_vals
):
    annot = df_vals.copy()

    for i in annot.index:

        for j in annot.columns:

            v = annot.loc[
                i,
                j
            ]

            if pd.isna(
                v
            ):

                annot.loc[
                    i,
                    j
                ] = ""

            else:

                annot.loc[
                    i,
                    j
                ] = f"{v:.1f}"

    return annot


def add_compact_top_cbar(
    ax,
    cmap,
    norm,
    left_text,
    right_text
):
    cax = inset_axes(
        ax,
        width="40%",
        height="5%",
        loc="upper left",
        bbox_to_anchor=(
            0,
            0.1,
            1,
            1
        ),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array(
        []
    )

    cbar = plt.colorbar(
        sm,
        cax=cax,
        orientation="horizontal"
    )

    cbar.outline.set_visible(
        False
    )

    cbar.set_ticks(
        []
    )

    cax.xaxis.set_major_locator(
        NullLocator()
    )

    cax.xaxis.set_minor_locator(
        NullLocator()
    )

    cax.xaxis.set_major_formatter(
        NullFormatter()
    )

    cax.xaxis.set_minor_formatter(
        NullFormatter()
    )

    cax.get_xaxis().get_offset_text().set_visible(
        False
    )

    cax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False
    )

    cax.text(
        -0.10,
        0.5,
        left_text,
        transform=cax.transAxes,
        ha="right",
        va="center",
        fontsize=6
    )

    cax.text(
        1.10,
        0.5,
        right_text,
        transform=cax.transAxes,
        ha="left",
        va="center",
        fontsize=6
    )

    cax.text(
        1.30,
        0.5,
        "risk-score OR",
        transform=cax.transAxes,
        ha="left",
        va="center",
        fontsize=6
    )

    return cbar


def trim_scatter_spines_to_ticks(
    ax,
    xticks=None,
    yticks=None,
    xpad=0.02
):

    if xticks is not None:

        ax.xaxis.set_major_locator(
            FixedLocator(
                xticks
            )
        )

        ax.set_xlim(
            min(
                xticks
            )
            -
            xpad,
            max(
                xticks
            )
            +
            xpad
        )

    else:

        ax.xaxis.set_major_locator(
            MaxNLocator(
                nbins=5
            )
        )

    if yticks is not None:

        ax.yaxis.set_major_locator(
            FixedLocator(
                yticks
            )
        )

        ax.set_ylim(
            min(
                yticks
            )
            -
            0.2,
            max(
                yticks
            )
            +
            0.2
        )

    else:

        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=5
            )
        )

    ax.figure.canvas.draw()

    xticks_current = np.array(
        ax.get_xticks()
    )

    xticks_current = xticks_current[
        (
            xticks_current
            >=
            ax.get_xlim()[0]
        )
        &
        (
            xticks_current
            <=
            ax.get_xlim()[1]
        )
    ]

    if len(
        xticks_current
    ) >= 2:

        ax.spines[
            "bottom"
        ].set_bounds(
            xticks_current[
                0
            ],
            xticks_current[
                -1
            ]
        )

    yticks_current = np.array(
        ax.get_yticks()
    )

    yticks_current = yticks_current[
        (
            yticks_current
            >=
            ax.get_ylim()[0]
        )
        &
        (
            yticks_current
            <=
            ax.get_ylim()[1]
        )
    ]

    if len(
        yticks_current
    ) >= 2:

        ax.spines[
            "left"
        ].set_bounds(
            yticks_current[
                0
            ],
            yticks_current[
                -1
            ]
        )

    ax.tick_params(
        axis="both",
        length=3,
        width=0.8
    )


def sensible_ticks(
    values,
    default_ticks,
    n=5
):
    vals = pd.to_numeric(
        pd.Series(
            values
        ),
        errors="coerce"
    ).dropna()

    if vals.empty:
        return default_ticks

    vmin = float(
        vals.min()
    )

    vmax = float(
        vals.max()
    )

    if (
        vmin >= min(
            default_ticks
        )
        and
        vmax <= max(
            default_ticks
        )
    ):
        return default_ticks

    lo = max(
        0.0,
        np.floor(
            vmin
            *
            10
        )
        /
        10
    )

    hi = np.ceil(
        vmax
        *
        10
    ) / 10

    if hi <= lo:
        hi = lo + 0.1

    return np.round(
        np.linspace(
            lo,
            hi,
            n
        ),
        2
    ).tolist()


def plot_scatter_panel(
    ax,
    pairs_df,
    x_col,
    x_label,
    xticks,
    color="red",
    yticks=None
):
    plot_df = pairs_df.dropna(
        subset=[
            x_col,
            "log_or"
        ]
    ).copy()

    (
        slope,
        intercept,
        r2,
        pperm,
        xg,
        yh,
        cil,
        cih
    ) = fit_line_with_bootstrap(
        plot_df[
            x_col
        ].values,
        plot_df[
            "log_or"
        ].values,
        n_boot=N_BOOT,
        seed=RANDOM_STATE
    )

    ax.scatter(
        plot_df[
            x_col
        ],
        plot_df[
            "log_or"
        ],
        s=20,
        color=color,
        alpha=0.9
    )

    ax.plot(
        xg,
        yh,
        color=color,
        linewidth=1
    )

    ax.fill_between(
        xg,
        cil,
        cih,
        color=color,
        alpha=0.18
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        "log(OR) per 1-SD risk score"
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    trim_scatter_spines_to_ticks(
        ax,
        xticks=xticks,
        yticks=yticks,
        xpad=0.02
    )

    ax.set_box_aspect(
        1.5
    )

    ax.text(
        0.08,
        0.96,
        f"$R^2$ = {r2:.2f}\n{pperm_text(pperm)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color=color
    )

    return {
        "x_col": x_col,
        "slope": slope,
        "intercept": intercept,
        "R2": r2,
        "Pperm": pperm,
        "n_pairs": len(
            plot_df
        )
    }


# =========================================================
# 4. 读取当前数据并锁定 Validation 6246
# =========================================================

needed_org_cols = [
    COLUMN_MAP[
        "stay_id"
    ],
    BURDEN_24_48,
    BURDEN_48_72,
]

needed_org_cols += [
    COLUMN_MAP[
        f"{o}_base"
    ]
    for o in ORGANS
]

needed_org_cols += [
    COLUMN_MAP[
        f"{o}_follow"
    ]
    for o in ORGANS
]


df_org = pd.read_csv(
    ORGAN_FILE,
    usecols=needed_org_cols,
    low_memory=False
)

df_risk = pd.read_csv(
    RISK_FILE,
    usecols=[
        COLUMN_MAP[
            "stay_id"
        ],
        COLUMN_MAP[
            "risk_score"
        ]
    ],
    low_memory=False
)


ensure_cols(
    df_org,
    needed_org_cols,
    file_desc="feature_matrix_full_0_6h.csv"
)

ensure_cols(
    df_risk,
    [
        COLUMN_MAP[
            "stay_id"
        ],
        COLUMN_MAP[
            "risk_score"
        ]
    ],
    file_desc="RISK_FILE"
)


df_org = (
    df_org
    .drop_duplicates(
        subset=[
            COLUMN_MAP[
                "stay_id"
            ]
        ]
    )
    .copy()
)


df_risk = (
    df_risk
    .drop_duplicates(
        subset=[
            COLUMN_MAP[
                "stay_id"
            ]
        ]
    )
    .copy()
)


df = df_org.merge(
    df_risk[
        [
            COLUMN_MAP[
                "stay_id"
            ],
            COLUMN_MAP[
                "risk_score"
            ]
        ]
    ],
    on=COLUMN_MAP[
        "stay_id"
    ],
    how="inner",
    validate="one_to_one"
).copy()


df[
    BURDEN_24_48
] = pd.to_numeric(
    df[
        BURDEN_24_48
    ],
    errors="coerce"
)

df[
    BURDEN_48_72
] = pd.to_numeric(
    df[
        BURDEN_48_72
    ],
    errors="coerce"
)

df[
    COLUMN_MAP[
        "risk_score"
    ]
] = pd.to_numeric(
    df[
        COLUMN_MAP[
            "risk_score"
        ]
    ],
    errors="coerce"
)


df = df[
    df[
        BURDEN_24_48
    ].notna()
    &
    df[
        BURDEN_48_72
    ].notna()
    &
    df[
        COLUMN_MAP[
            "risk_score"
        ]
    ].notna()
].copy()


print(
    f"Current common validation subset with risk score N = "
    f"{len(df):,}"
)


if len(
    df
) != 6246:

    print(
        f"WARNING: expected N=6,246, "
        f"but current matched N={len(df):,}."
    )


for o in ORGANS:

    df[
        COLUMN_MAP[
            f"{o}_base"
        ]
    ] = to_binary_preserve_na(
        df[
            COLUMN_MAP[
                f"{o}_base"
            ]
        ]
    )

    df[
        COLUMN_MAP[
            f"{o}_follow"
        ]
    ] = to_binary_preserve_na(
        df[
            COLUMN_MAP[
                f"{o}_follow"
            ]
        ]
    )


df[
    "risk_score_z"
] = zscore(
    df[
        COLUMN_MAP[
            "risk_score"
        ]
    ]
)


# =========================================================
# 4.1 PCC / OFTP in SAME validation cohort
# =========================================================

(
    pcc_mat,
    oftp_mat,
    oftp_num,
    oftp_den
) = compute_pcc_and_oftp(
    df
)


pcc_out = pcc_mat.copy()
oftp_out = oftp_mat.copy()


for m in [
    pcc_out,
    oftp_out
]:

    m.index = [
        ORGAN_LABELS[
            o
        ]
        for o in m.index
    ]

    m.columns = [
        ORGAN_LABELS[
            o
        ]
        for o in m.columns
    ]


pcc_out.to_csv(
    OUTDIR
    /
    "figure3d_pcc_matrix_validation_6246.csv",
    encoding="utf-8-sig"
)


oftp_out.to_csv(
    OUTDIR
    /
    "figure3d_oftp_matrix_validation_6246.csv",
    encoding="utf-8-sig"
)


oftp_num.to_csv(
    OUTDIR
    /
    "figure3d_oftp_numerator_validation_6246.csv",
    encoding="utf-8-sig"
)


oftp_den.to_csv(
    OUTDIR
    /
    "figure3d_oftp_denominator_validation_6246.csv",
    encoding="utf-8-sig"
)


# =========================================================
# 5. Risk-score-associated ORs for NEW target failure
# =========================================================

mat_d = pd.DataFrame(
    np.nan,
    index=ORGANS,
    columns=ORGANS,
    dtype=float
)

pairs_d = []


for target in ORGANS:

    target_base_col = COLUMN_MAP[
        f"{target}_base"
    ]

    target_follow_col = COLUMN_MAP[
        f"{target}_follow"
    ]


    for source in ORGANS:

        source_base_col = COLUMN_MAP[
            f"{source}_base"
        ]


        if source == target:

            pairs_d.append({
                "source": source,
                "target": target,
                "is_self_pair": 1,
                "pcc": np.nan,
                "oftp": np.nan,
                "or": np.nan,
                "log_or": np.nan,
                "n_subgroup": 0,
                "n_events": 0,
            })

            continue


        # SAME transition-risk set used by OFTP:
        # source failed, target unaffected at baseline.
        sub = df[
            (
                df[
                    source_base_col
                ]
                ==
                1
            )
            &
            (
                df[
                    target_base_col
                ]
                ==
                0
            )
            &
            df[
                target_follow_col
            ].notna()
            &
            df[
                "risk_score_z"
            ].notna()
        ].copy()


        or_ij = riskscore_or_within_transition_subgroup(
            y=sub[
                target_follow_col
            ],
            x=sub[
                "risk_score_z"
            ]
        )


        mat_d.loc[
            target,
            source
        ] = or_ij


        pairs_d.append({
            "source": source,
            "target": target,
            "is_self_pair": 0,

            "pcc":
                pcc_mat.loc[
                    source,
                    target
                ],

            "oftp":
                oftp_mat.loc[
                    source,
                    target
                ],

            "or":
                or_ij,

            "log_or":
                (
                    np.log(
                        or_ij
                    )
                    if (
                        pd.notna(
                            or_ij
                        )
                        and
                        or_ij > 0
                    )
                    else np.nan
                ),

            "n_subgroup":
                len(
                    sub
                ),

            "n_events":
                int(
                    (
                        sub[
                            target_follow_col
                        ]
                        ==
                        1
                    ).sum()
                ),
        })


pairs_d = pd.DataFrame(
    pairs_d
)


pairs_d_reg = pairs_d[
    pairs_d[
        "source"
    ]
    !=
    pairs_d[
        "target"
    ]
].dropna(
    subset=[
        "log_or"
    ]
).copy()


print(
    "\nRisk-score transition pairs:"
)

print(
    pairs_d_reg[
        [
            "source",
            "target",
            "pcc",
            "oftp",
            "or",
            "log_or",
            "n_subgroup",
            "n_events",
        ]
    ].round(
        3
    )
)


pairs_d.to_csv(
    OUTDIR
    /
    "figure3d_riskscore_pairs_pcc_oftp_all_pairs.csv",
    index=False,
    encoding="utf-8-sig"
)


pairs_d_reg.to_csv(
    OUTDIR
    /
    "figure3d_riskscore_pairs_pcc_oftp_no_self.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 6. OR matrix
# =========================================================

mat_d_out = mat_d.copy()


mat_d_out.index = pd.Index(
    [
        ORGAN_LABELS[
            o
        ]
        +
        " <"
        for o in mat_d_out.index
    ],
    name="24–48 h new target"
)


mat_d_out.columns = pd.Index(
    [
        ORGAN_LABELS[
            o
        ]
        +
        " >"
        for o in mat_d_out.columns
    ],
    name="0–6 h source"
)


mat_d_out.to_csv(
    OUTDIR
    /
    "figure3d_riskscore_new_failure_or_matrix.csv",
    encoding="utf-8-sig"
)


# =========================================================
# 7. Plot: heatmap + PCC scatter + OFTP scatter
# =========================================================

fig = plt.figure(
    figsize=(
        6,
        2
    )
)


gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[
        1.45,
        1.05,
        1.05
    ],
    wspace=0.35
)


ax_hd = fig.add_subplot(
    gs[
        0,
        0
    ]
)

ax_pcc = fig.add_subplot(
    gs[
        0,
        1
    ]
)

ax_oftp = fig.add_subplot(
    gs[
        0,
        2
    ]
)


# ---------- Heatmap ----------
plot_d = mat_d_out.copy()
annot_d = make_annot_df(
    plot_d
)


positive_vals = plot_d.to_numpy(
    dtype=float
)

positive_vals = positive_vals[
    np.isfinite(
        positive_vals
    )
    &
    (
        positive_vals
        >
        0
    )
]


if len(
    positive_vals
) == 0:

    raise ValueError(
        "risk-score OR matrix 没有有效正值。"
    )


heat_vmin = max(
    0.5,
    float(
        np.nanmin(
            positive_vals
        )
    )
)

heat_vmax = float(
    np.nanmax(
        positive_vals
    )
)


if heat_vmax <= heat_vmin:
    heat_vmax = heat_vmin * 1.05


plot_norm_d = LogNorm(
    vmin=heat_vmin,
    vmax=heat_vmax
)


sns.heatmap(
    plot_d,
    ax=ax_hd,
    cmap="rocket_r",
    norm=plot_norm_d,
    annot=annot_d,
    fmt="",
    annot_kws={
        "fontsize": 6
    },
    cbar=False,
    linewidths=0.5,
    linecolor="white",
    square=True
)


ax_hd.set_xticklabels(
    ax_hd.get_xticklabels(),
    rotation=90,
    ha="center"
)


ax_hd.set_yticklabels(
    ax_hd.get_yticklabels(),
    rotation=0,
    va="center"
)


ax_hd.tick_params(
    axis="both",
    length=0
)


add_compact_top_cbar(
    ax=ax_hd,
    cmap=mpl.cm.get_cmap(
        "rocket_r"
    ),
    norm=plot_norm_d,
    left_text=f"{heat_vmin:.1f}",
    right_text=f"{heat_vmax:.1f}"
)


ax_hd.set_title(
    "Risk score-associated ORs\nfor new target-organ failure",
    fontsize=6,
    pad=30
)


# ---------- Scatter: PCC ----------
pcc_ticks = sensible_ticks(
    pairs_d_reg[
        "pcc"
    ],
    default_ticks=[
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3
    ]
)


summary_pcc = plot_scatter_panel(
    ax=ax_pcc,
    pairs_df=pairs_d_reg,
    x_col="pcc",
    x_label="PCC",
    xticks=pcc_ticks,
    color="red",
    yticks=None
)


# ---------- Scatter: OFTP ----------
oftp_ticks = sensible_ticks(
    pairs_d_reg[
        "oftp"
    ],
    default_ticks=[
        0.0,
        0.1,
        0.2,
        0.3
    ]
)


summary_oftp = plot_scatter_panel(
    ax=ax_oftp,
    pairs_df=pairs_d_reg,
    x_col="oftp",
    x_label="OFTP",
    xticks=oftp_ticks,
    color="red",
    yticks=None
)


reg_summary = pd.DataFrame(
    [
        summary_pcc,
        summary_oftp
    ]
)


reg_summary.to_csv(
    OUTDIR
    /
    "figure3d_riskscore_regression_summary_pcc_oftp.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    "\nRegression summary:"
)

print(
    reg_summary.round(
        4
    )
)


# panel label
fig.text(
    0.02,
    0.92,
    "d",
    fontsize=6,
    fontweight="bold",
    va="top"
)


plt.tight_layout(
    rect=[
        0.01,
        0.01,
        1,
        0.96
    ]
)


png_path = (
    OUTDIR
    /
    "figure3d_current_validation_riskscore_pcc_oftp.png"
)

pdf_path = (
    OUTDIR
    /
    "figure3d_current_validation_riskscore_pcc_oftp.pdf"
)

svg_path = (
    OUTDIR
    /
    "figure3d_current_validation_riskscore_pcc_oftp.svg"
)


plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.savefig(
    svg_path,
    bbox_inches="tight"
)

plt.show()
plt.close()


print(
    "\nSaved:"
)

print(
    png_path.resolve()
)

print(
    pdf_path.resolve()
)
