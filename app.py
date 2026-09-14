!pip install -q -U dash plotly

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

df = pd.read_csv("personality_datasert.csv")

df = df.replace(
    ['None', '', 'NA', 'na', 'null'],
    np.nan
)

df.drop_duplicates(inplace=True)


# ============================================================
# 2. ENCODING
# ============================================================

df['Stage_fear'] = df['Stage_fear'].map({
    'Yes': 1,
    'No': 0
})

df['Drained_after_socializing'] = df[
    'Drained_after_socializing'
].map({
    'Yes': 1,
    'No': 0
})

df['Personality'] = df['Personality'].map({
    'Extrovert': 1,
    'Introvert': 0
})

df.dropna(inplace=True)


# ============================================================
# 3. FEATURES / TARGET
# ============================================================

X = df.drop("Personality", axis=1)
y = df["Personality"]


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf"
    )

}


# ============================================================
# 7. TRAIN MODELS + EVALUATION
# ============================================================

results = []
predictions = {}

for name, model in models.items():

    if name in [
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]:

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

    predictions[name] = y_pred

    results.append({

        "Model": name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        )
    })


results_df = pd.DataFrame(results)


# ============================================================
# 8. BEST MODEL
# ============================================================

# ترتيب الأولوية عند حدوث تعادل في الـ Accuracy
model_priority = {
    "Random Forest": 1,
    "Naive Bayes": 2,
    "Decision Tree": 3,
    "KNN": 4,
    "Logistic Regression": 5,
    "SVM": 6
}

# إضافة أولوية لكل Model
results_df["Priority"] = results_df["Model"].map(model_priority)

# اختيار أعلى Accuracy،
# وفي حالة التعادل يتم اختيار الأعلى أولوية
best_index = results_df.sort_values(
    by=["Accuracy", "Priority"],
    ascending=[False, True]
).index[0]

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_accuracy = results_df.loc[
    best_index,
    "Accuracy"
]

print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy}")

# ============================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_model = models["Random Forest"]

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        rf_model.feature_importances_

})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)


# ============================================================
# 10. CANDIDATE RANKING
# ============================================================

candidate_ranking = X_test.copy()

candidate_ranking["Actual Personality"] = (
    y_test.values
)

candidate_ranking["Extrovert Probability"] = (
    rf_model.predict_proba(X_test)[:, 1]
)

candidate_ranking = candidate_ranking.sort_values(
    "Extrovert Probability",
    ascending=False
)

candidate_ranking["Rank"] = range(
    1,
    len(candidate_ranking) + 1
)

candidate_ranking["Personality"] = (
    candidate_ranking["Actual Personality"]
    .map({
        0: "Introvert",
        1: "Extrovert"
    })
)


# ============================================================
# 11. DASH APP
# ============================================================

app = Dash(
    __name__,
    external_stylesheets=[],
    serve_locally=False
)

app.config.suppress_callback_exceptions = True

app.title = "Personality AI | Analytics Command Center"


# ============================================================
# 12. THEME
# ============================================================

DARK_BG = "#060914"

CARD_BG = "rgba(15, 23, 42, 0.72)"

ACCENT_BLUE = "#38bdf8"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_CYAN = "#22d3ee"
ACCENT_GREEN = "#34d399"
ACCENT_PINK = "#f472b6"

TEXT_COLOR = "#f8fafc"
TEXT_MUTED = "#94a3b8"

BORDER = "rgba(148, 163, 184, 0.14)"

PLOTLY_TEMPLATE = "plotly_dark"

COLOR_DISCRETE_SEQUENCE = [
    ACCENT_CYAN,
    ACCENT_PURPLE,
    ACCENT_PINK,
    ACCENT_GREEN,
    "#fbbf24"
]


# ============================================================
# 13. PLOTLY STYLE
# ============================================================

def apply_custom_layout(fig):

    fig.update_layout(

        template=PLOTLY_TEMPLATE,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            family="Arial, sans-serif",
            color=TEXT_COLOR
        ),

        margin=dict(
            l=45,
            r=45,
            t=75,
            b=45
        ),

        title=dict(
            font=dict(
                size=20,
                color=TEXT_COLOR,
                family="Arial, sans-serif"
            ),
            x=0.02,
            xanchor="left"
        ),

        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="rgba(255,255,255,.08)",
            font=dict(
                color=TEXT_COLOR
            )
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                color=TEXT_MUTED
            )
        )
    )

    return fig


# ============================================================
# 14. PAGE STYLE
# ============================================================

PAGE_STYLE = {

    "background": """
        radial-gradient(
            circle at 5% 0%,
            rgba(34,211,238,.13),
            transparent 25%
        ),

        radial-gradient(
            circle at 95% 5%,
            rgba(139,92,246,.16),
            transparent 28%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(56,189,248,.06),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #050816 0%,
            #0a1020 50%,
            #060914 100%
        )
    """,

    "padding": "30px clamp(16px, 4vw, 60px)",

    "fontFamily":
        "Arial, sans-serif",

    "minHeight": "100vh",

    "color": TEXT_COLOR,

    "boxSizing": "border-box"
}


# ============================================================
# 15. CARD STYLE
# ============================================================

CARD_STYLE = {

    "background": """
        linear-gradient(
            145deg,
            rgba(15,23,42,.88),
            rgba(15,23,42,.55)
        )
    """,

    "padding": "25px",

    "borderRadius": "22px",

    "boxShadow": """
        0 18px 50px rgba(0,0,0,.28),
        inset 0 1px 0 rgba(255,255,255,.035)
    """,

    "border":
        f"1px solid {BORDER}",

    "textAlign": "center",

    "flex": "1",

    "margin": "8px",

    "minWidth": "190px",

    "backdropFilter": "blur(18px)",

    "WebkitBackdropFilter":
        "blur(18px)",

    "transition": """
        transform .25s ease,
        border-color .25s ease,
        box-shadow .25s ease
    """
}


# ============================================================
# 16. TABS STYLE
# ============================================================

TAB_STYLE = {

    "backgroundColor":
        "transparent",

    "color":
        TEXT_MUTED,

    "border":
        "none",

    "padding":
        "13px 25px",

    "fontWeight":
        "600",

    "fontSize":
        "13px",

    "cursor":
        "pointer"
}


TAB_SELECTED_STYLE = {

    "background": """
        linear-gradient(
            135deg,
            rgba(56,189,248,.18),
            rgba(139,92,246,.20)
        )
    """,

    "color":
        "#ffffff",

    "border":
        "1px solid rgba(56,189,248,.24)",

    "borderRadius":
        "13px",

    "padding":
        "13px 25px",

    "fontWeight":
        "700",

    "boxShadow":
        "0 8px 28px rgba(56,189,248,.10)",

    "cursor":
        "pointer"
}


# ============================================================
# 17. CUSTOM CSS
# ============================================================

app.index_string = """

<!DOCTYPE html>

<html>

<head>

    {%metas%}

    <title>{%title%}</title>

    {%favicon%}

    {%css%}

    <style>

        * {
            box-sizing: border-box;
        }

        html,
        body {

            margin: 0;
            padding: 0;

            background: #060914;
        }

        body {

            overflow-x: hidden;
        }

        ::selection {

            background:
                rgba(56,189,248,.35);

            color: white;
        }


        /* PAGE */

        .dashboard-shell {

            animation:
                pageIn .55s ease-out;
        }


        @keyframes pageIn {

            from {

                opacity: 0;

                transform:
                    translateY(12px);
            }

            to {

                opacity: 1;

                transform:
                    translateY(0);
            }
        }


        /* KPI */

        .kpi-card {

            position: relative;
            overflow: hidden;
        }

        .kpi-card::before {

            content: "";

            position: absolute;

            width: 130px;
            height: 130px;

            top: -75px;
            right: -55px;

            background:
                rgba(56,189,248,.08);

            border-radius: 50%;

            filter: blur(4px);
        }

        .kpi-card:hover {

            transform:
                translateY(-6px);

            border-color:
                rgba(56,189,248,.30)
                !important;

            box-shadow:

                0 22px 60px rgba(0,0,0,.40),

                0 0 35px
                rgba(56,189,248,.08)
                !important;
        }


        /* GRAPH */

        .dash-graph {

            border-radius: 18px;
            overflow: hidden;
        }


        /* BUTTON */

        .premium-btn {

            position: relative;

            overflow: hidden;

            transition:
                all .25s ease;
        }

        .premium-btn:hover {

            transform:
                translateY(-2px);

            box-shadow:

                0 15px 40px
                rgba(56,189,248,.25)
                !important;
        }

        .premium-btn::after {

            content: "";

            position: absolute;

            top: -100%;
            left: -35%;

            width: 28%;
            height: 300%;

            background:
                rgba(255,255,255,.22);

            transform:
                rotate(25deg);

            transition:
                left .55s ease;
        }

        .premium-btn:hover::after {

            left: 120%;
        }


        /* TABS */

        .dash-tabs-container {

            position: relative;
            z-index: 1000;
        }

        .dash-tabs-container .tab-parent {

            border:
                none !important;
        }

        .dash-tabs-container .tab-container {

            border:
                none !important;
        }

        .dash-tabs-container .tab {

            cursor:
                pointer !important;

            user-select:
                none;

            transition:
                all .2s ease;
        }


        /* INPUTS */

        input[type="number"] {

            outline: none;

            transition:
                border-color .2s ease,
                box-shadow .2s ease,
                transform .2s ease;
        }

        input[type="number"]:focus {

            border-color:
                rgba(56,189,248,.65)
                !important;

            box-shadow:

                0 0 0 4px
                rgba(56,189,248,.08)
                !important;

            transform:
                translateY(-1px);
        }


        /* TABLE */

        table {

            border-spacing:
                0;
        }

        table tbody tr {

            transition:
                background .18s ease;
        }

        table tbody tr:hover {

            background:
                rgba(56,189,248,.055)
                !important;
        }


        /* MOBILE */

        @media (max-width: 850px) {

            .dashboard-shell {

                padding:
                    18px !important;
            }

            .hero-title {

                font-size:
                    38px !important;

                letter-spacing:
                    -1.8px !important;
            }

            .kpi-grid {

                flex-direction:
                    column !important;
            }

            .kpi-card {

                width:
                    100% !important;

                margin:
                    7px 0 !important;
            }

            .prediction-grid {

                grid-template-columns:
                    1fr !important;
            }
        }

    </style>

</head>


<body>

    {%app_entry%}

    <footer>

        {%config%}

        {%scripts%}

        {%renderer%}

    </footer>

</body>

</html>

"""


# ============================================================
# 18. MAIN LAYOUT
# ============================================================

app.layout = html.Div(

    className="dashboard-shell",

    style=PAGE_STYLE,

    children=[


        # ====================================================
        # HERO
        # ====================================================

        html.Div(

            [

                html.Div(

                    "AI  •  BEHAVIORAL INTELLIGENCE",

                    style={

                        "display":
                            "inline-block",

                        "padding":
                            "7px 14px",

                        "borderRadius":
                            "999px",

                        "background":
                            "rgba(34,211,238,.07)",

                        "border":
                            "1px solid rgba(34,211,238,.18)",

                        "color":
                            ACCENT_CYAN,

                        "fontSize":
                            "10px",

                        "fontWeight":
                            "800",

                        "letterSpacing":
                            "2px",

                        "marginBottom":
                            "17px"
                    }
                ),

                html.H1(

                    "Personality Classification",

                    className="hero-title",

                    style={

                        "fontSize":
                            "clamp(42px, 5.2vw, 68px)",

                        "fontWeight":
                            "800",

                        "letterSpacing":
                            "-3px",

                        "lineHeight":
                            "1",

                        "margin":
                            "0",

                        "background": """
                            linear-gradient(
                                100deg,
                                #ffffff 12%,
                                #67e8f9 48%,
                                #a78bfa 90%
                            )
                        """,

                        "WebkitBackgroundClip":
                            "text",

                        "WebkitTextFillColor":
                            "transparent"
                    }
                ),

                html.P(

                    "Machine learning insights  •  "
                    "predictive behavior  •  "
                    "model intelligence",

                    style={

                        "color":
                            TEXT_MUTED,

                        "fontSize":
                            "15px",

                        "marginTop":
                            "15px",

                        "marginBottom":
                            "30px"
                    }
                )

            ],

            style={

                "maxWidth":
                    "1250px",

                "margin":
                    "0 auto"
            }
        ),


        # ====================================================
        # KPI CARDS
        # ====================================================

        html.Div(

            className="kpi-grid",

            style={

                "display":
                    "flex",

                "justifyContent":
                    "space-between",

                "flexWrap":
                    "wrap",

                "marginBottom":
                    "24px",

                "maxWidth":
                    "1250px",

                "marginLeft":
                    "auto",

                "marginRight":
                    "auto"
            },

            children=[

                html.Div(

                    [

                        html.Div(
                            "DATASET",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "letterSpacing": "1.5px",
                                "color": TEXT_MUTED
                            }
                        ),

                        html.H2(
                            f"{len(df):,}",
                            style={
                                "fontSize": "34px",
                                "margin": "8px 0 2px",
                                "fontWeight": "800"
                            }
                        ),

                        html.P(
                            "Total Records",
                            style={
                                "color": TEXT_MUTED,
                                "fontSize": "12px",
                                "margin": "0"
                            }
                        )

                    ],

                    className="kpi-card",

                    style=CARD_STYLE
                ),


                html.Div(

                    [

                        html.Div(
                            "EXTROVERT",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "letterSpacing": "1.5px",
                                "color": ACCENT_CYAN
                            }
                        ),

                        html.H2(
                            f"{(df['Personality'] == 1).sum():,}",
                            style={
                                "fontSize": "34px",
                                "margin": "8px 0 2px",
                                "fontWeight": "800",
                                "color": ACCENT_CYAN
                            }
                        ),

                        html.P(
                            "Profiles classified",
                            style={
                                "color": TEXT_MUTED,
                                "fontSize": "12px",
                                "margin": "0"
                            }
                        )

                    ],

                    className="kpi-card",

                    style=CARD_STYLE
                ),


                html.Div(

                    [

                        html.Div(
                            "INTROVERT",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "letterSpacing": "1.5px",
                                "color": ACCENT_PURPLE
                            }
                        ),

                        html.H2(
                            f"{(df['Personality'] == 0).sum():,}",
                            style={
                                "fontSize": "34px",
                                "margin": "8px 0 2px",
                                "fontWeight": "800",
                                "color": ACCENT_PURPLE
                            }
                        ),

                        html.P(
                            "Profiles classified",
                            style={
                                "color": TEXT_MUTED,
                                "fontSize": "12px",
                                "margin": "0"
                            }
                        )

                    ],

                    className="kpi-card",

                    style=CARD_STYLE
                ),


                html.Div(

                    [

                        html.Div(
                            "TOP MODEL",
                            style={
                                "fontSize": "10px",
                                "fontWeight": "800",
                                "letterSpacing": "1.5px",
                                "color": ACCENT_GREEN
                            }
                        ),

                        html.H2(
                            best_model_name,
                            style={
                                "fontSize": "18px",
                                "margin": "10px 0 2px",
                                "fontWeight": "800",
                                "color": ACCENT_GREEN
                            }
                        ),

                        html.P(
                            f"Accuracy  •  {best_accuracy:.2%}",
                            style={
                                "color": TEXT_MUTED,
                                "fontSize": "12px",
                                "margin": "0"
                            }
                        )

                    ],

                    className="kpi-card",

                    style=CARD_STYLE
                )

            ]
        ),


        # ====================================================
        # MAIN NAVIGATION — ONLY TWO TABS
        # ====================================================

        html.Div(

            className="dash-tabs-container",

            style={

                "maxWidth":
                    "1250px",

                "margin":
                    "0 auto 22px",

                "padding":
                    "6px",

                "borderRadius":
                    "17px",

                "background":
                    "rgba(15,23,42,.60)",

                "border":
                    f"1px solid {BORDER}",

                "boxShadow":
                    "0 12px 35px rgba(0,0,0,.18)",

                "backdropFilter":
                    "blur(16px)"
            },

            children=[

                dcc.Tabs(

                    id="main-tabs",

                    value="analysis",

                    style={
                        "height": "auto",
                        "width": "100%"
                    },

                    children=[

                        dcc.Tab(
                            label="◈  Analysis",
                            value="analysis",
                            style=TAB_STYLE,
                            selected_style=TAB_SELECTED_STYLE
                        ),

                        dcc.Tab(
                            label="✦  Prediction",
                            value="prediction",
                            style=TAB_STYLE,
                            selected_style=TAB_SELECTED_STYLE
                        )

                    ]
                )

            ]
        ),


        # ====================================================
        # CONTENT
        # ====================================================

        html.Div(

            id="main-content",

            style={

                "maxWidth":
                    "1250px",

                "margin":
                    "0 auto",

                "background": """
                    linear-gradient(
                        145deg,
                        rgba(15,23,42,.82),
                        rgba(15,23,42,.57)
                    )
                """,

                "borderRadius":
                    "24px",

                "padding":
                    "clamp(18px, 3vw, 34px)",

                "border":
                    f"1px solid {BORDER}",

                "boxShadow":
                    "0 25px 70px rgba(0,0,0,.30)",

                "backdropFilter":
                    "blur(20px)",

                "minHeight":
                    "520px"
            }
        ),


        # ====================================================
        # FOOTER
        # ====================================================

        html.Div(

            "PERSONALITY AI  /  ANALYTICS COMMAND CENTER",

            style={

                "maxWidth":
                    "1250px",

                "margin":
                    "18px auto 0",

                "textAlign":
                    "right",

                "fontSize":
                    "9px",

                "letterSpacing":
                    "2px",

                "color":
                    "rgba(148,163,184,.42)",

                "fontWeight":
                    "700"
            }
        )

    ]
)


# ============================================================
# 19. MAIN TAB CALLBACK
# ============================================================

@app.callback(

    Output(
        "main-content",
        "children"
    ),

    Input(
        "main-tabs",
        "value"
    )
)

def render_main_tab(tab):


    # ========================================================
    # ANALYSIS
    # ========================================================

    if tab == "analysis":

        # ----------------------------------------------------
        # OVERVIEW GRAPH
        # ----------------------------------------------------

        personality_counts = (

            df["Personality"]

            .map({
                0: "Introvert",
                1: "Extrovert"
            })

            .value_counts()

            .reset_index()
        )

        personality_counts.columns = [
            "Personality",
            "Count"
        ]


        overview_fig = px.pie(

            personality_counts,

            names="Personality",

            values="Count",

            hole=0.6,

            title=
                "Overall Personality Distribution",

            color_discrete_sequence=[
                ACCENT_BLUE,
                ACCENT_PURPLE
            ]
        )

        apply_custom_layout(
            overview_fig
        )


        # ----------------------------------------------------
        # MODEL BENCHMARK
        # ----------------------------------------------------

        models_fig = px.bar(

            results_df,

            x="Model",

            y=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],

            barmode="group",

            title=
                "Algorithm Metrics Benchmark",

            range_y=[0, 1],

            color_discrete_sequence=
                COLOR_DISCRETE_SEQUENCE
        )

        apply_custom_layout(
            models_fig
        )


        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        features_fig = px.bar(

            feature_importance,

            x="Importance",

            y="Feature",

            orientation="h",

            title=
                "Random Forest Relative Feature Importance",

            color="Importance",

            color_continuous_scale=
                "Blues"
        )

        apply_custom_layout(
            features_fig
        )


        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        ranking_display = (

            candidate_ranking[
                [
                    "Rank",
                    "Personality",
                    "Extrovert Probability"
                ]
            ]

            .head(20)
        )


        ranking_fig = px.bar(

            ranking_display,

            x="Rank",

            y="Extrovert Probability",

            hover_data=[
                "Personality"
            ],

            title=
                "Top Candidates by Extrovert Probability",

            color=
                "Extrovert Probability",

            color_continuous_scale=
                "Tealgrn"
        )

        apply_custom_layout(
            ranking_fig
        )


        # ----------------------------------------------------
        # MODEL TABLE
        # ----------------------------------------------------

        model_table_header = [

            html.Th(

                col,

                style={

                    "padding":
                        "12px",

                    "borderBottom":
                        "2px solid #334155",

                    "color":
                        ACCENT_BLUE
                }
            )

            for col in results_df.columns
        ]


        model_table_rows = []


        for _, row in results_df.iterrows():

            model_table_rows.append(

                html.Tr([

                    html.Td(

                        (
                            f"{row[col]:.2%}"
                            if col != "Model"
                            else row[col]
                        ),

                        style={

                            "padding":
                                "12px",

                            "borderBottom":
                                "1px solid #334155",

                            "color":
                                TEXT_COLOR
                        }
                    )

                    for col in results_df.columns

                ])
            )


        # ----------------------------------------------------
        # RANKING TABLE
        # ----------------------------------------------------

        ranking_table_header = [

            html.Th(

                "Rank",

                style={

                    "padding":
                        "12px",

                    "borderBottom":
                        "2px solid #334155",

                    "color":
                        ACCENT_BLUE
                }
            ),

            html.Th(

                "Actual Personality",

                style={

                    "padding":
                        "12px",

                    "borderBottom":
                        "2px solid #334155",

                    "color":
                        ACCENT_BLUE
                }
            ),

            html.Th(

                "Extrovert Confidence",

                style={

                    "padding":
                        "12px",

                    "borderBottom":
                        "2px solid #334155",

                    "color":
                        ACCENT_BLUE
                }
            )
        ]


        ranking_table_rows = []


        for _, row in ranking_display.iterrows():

            ranking_table_rows.append(

                html.Tr([

                    html.Td(

                        row["Rank"],

                        style={

                            "padding":
                                "10px",

                            "borderBottom":
                                "1px solid #334155"
                        }
                    ),

                    html.Td(

                        row["Personality"],

                        style={

                            "padding":
                                "10px",

                            "borderBottom":
                                "1px solid #334155"
                        }
                    ),

                    html.Td(

                        f"{row['Extrovert Probability']:.2%}",

                        style={

                            "padding":
                                "10px",

                            "borderBottom":
                                "1px solid #334155",

                            "fontWeight":
                                "bold",

                            "color":
                                ACCENT_BLUE
                        }
                    )

                ])
            )


        # ====================================================
        # ANALYSIS PAGE
        # ====================================================

        return html.Div([


            # =================================================
            # SECTION 1 — OVERVIEW
            # =================================================

            html.Div([

                html.Div(

                    [

                        html.Div(
                            "01",
                            style={
                                "fontSize": "11px",
                                "fontWeight": "800",
                                "letterSpacing": "2px",
                                "color": ACCENT_CYAN
                            }
                        ),

                        html.H2(
                            "Overview",
                            style={
                                "fontSize": "28px",
                                "margin": "5px 0 5px",
                                "fontWeight": "800"
                            }
                        ),

                        html.P(
                            "High-level view of the personality dataset.",
                            style={
                                "color": TEXT_MUTED,
                                "margin": "0"
                            }
                        )

                    ],

                    style={
                        "marginBottom": "15px"
                    }
                ),

                dcc.Graph(

                    figure=overview_fig,

                    config={
                        "responsive": True,
                        "displayModeBar": False
                    }
                ),

                html.Div([

                    html.H3(
                        "Business Insights",
                        style={
                            "color": ACCENT_BLUE,
                            "marginTop": "10px"
                        }
                    ),

                    html.Ul([

                        html.Li(
                            f"Best performing model is "
                            f"{best_model_name}."
                        ),

                        html.Li(
                            f"Top achieved model accuracy "
                            f"is {best_accuracy:.2%}."
                        ),

                        html.Li(
                            "Behavioral features are strong "
                            "indicators for personality classification."
                        ),

                        html.Li(
                            "Random Forest feature importance "
                            "highlights the strongest behavioral drivers."
                        )

                    ],

                    style={
                        "lineHeight": "2",
                        "color": TEXT_MUTED
                    })

                ],

                style={
                    "padding": "10px 20px 25px"
                })

            ],

            style={
                "borderBottom":
                    f"1px solid {BORDER}",
                "paddingBottom":
                    "25px",
                "marginBottom":
                    "35px"
            }),


            # =================================================
            # SECTION 2 — MODELS
            # =================================================

            html.Div([

                html.Div(

                    [

                        html.Div(
                            "02",
                            style={
                                "fontSize": "11px",
                                "fontWeight": "800",
                                "letterSpacing": "2px",
                                "color": ACCENT_PURPLE
                            }
                        ),

                        html.H2(
                            "Model Performance",
                            style={
                                "fontSize": "28px",
                                "margin": "5px 0 5px",
                                "fontWeight": "800"
                            }
                        ),

                        html.P(
                            "Comparison of all machine learning algorithms.",
                            style={
                                "color": TEXT_MUTED,
                                "margin": "0"
                            }
                        )

                    ],

                    style={
                        "marginBottom": "15px"
                    }
                ),

                dcc.Graph(

                    figure=models_fig,

                    config={
                        "responsive": True,
                        "displayModeBar": False
                    }
                ),

                html.Div([

                    html.H3(

                        f"Leaderboard Winner: "
                        f"{best_model_name}",

                        style={
                            "color":
                                ACCENT_GREEN,
                            "marginTop":
                                "20px"
                        }
                    ),

                    html.P(

                        f"Peak Accuracy Score: "
                        f"{best_accuracy:.2%}",

                        style={
                            "color":
                                TEXT_MUTED
                        }
                    ),

                    html.H4(

                        "Detailed Metrics Table",

                        style={
                            "marginTop":
                                "25px",
                            "marginBottom":
                                "15px"
                        }
                    ),

                    html.Table(

                        [

                            html.Thead(
                                html.Tr(
                                    model_table_header
                                )
                            ),

                            html.Tbody(
                                model_table_rows
                            )

                        ],

                        style={

                            "width":
                                "100%",

                            "textAlign":
                                "center",

                            "borderCollapse":
                                "collapse",

                            "backgroundColor":
                                "rgba(15, 23, 42, 0.4)",

                            "borderRadius":
                                "8px",

                            "overflow":
                                "hidden"
                        }
                    )

                ])

            ],

            style={
                "borderBottom":
                    f"1px solid {BORDER}",
                "paddingBottom":
                    "35px",
                "marginBottom":
                    "35px"
            }),


            # =================================================
            # SECTION 3 — FEATURES
            # =================================================

            html.Div([

                html.Div(

                    [

                        html.Div(
                            "03",
                            style={
                                "fontSize": "11px",
                                "fontWeight": "800",
                                "letterSpacing": "2px",
                                "color": ACCENT_BLUE
                            }
                        ),

                        html.H2(
                            "Feature Intelligence",
                            style={
                                "fontSize": "28px",
                                "margin": "5px 0 5px",
                                "fontWeight": "800"
                            }
                        ),

                        html.P(
                            "Behavioral features ranked by their relative importance.",
                            style={
                                "color": TEXT_MUTED,
                                "margin": "0"
                            }
                        )

                    ],

                    style={
                        "marginBottom": "15px"
                    }
                ),

                dcc.Graph(

                    figure=features_fig,

                    config={
                        "responsive": True,
                        "displayModeBar": False
                    }
                ),

                html.Div([

                    html.H3(

                        f"Top Key Driver: "
                        f"{feature_importance.iloc[0]['Feature']}",

                        style={
                            "color":
                                ACCENT_BLUE
                        }
                    ),

                    html.P(

                        f"Calculated Relative Weight: "
                        f"{feature_importance.iloc[0]['Importance']:.2%}",

                        style={
                            "color":
                                TEXT_MUTED
                        }
                    )

                ],

                style={
                    "marginTop":
                        "10px"
                })

            ],

            style={
                "borderBottom":
                    f"1px solid {BORDER}",
                "paddingBottom":
                    "35px",
                "marginBottom":
                    "35px"
            }),


            # =================================================
            # SECTION 4 — RANKING
            # =================================================

            html.Div([

                html.Div(

                    [

                        html.Div(
                            "04",
                            style={
                                "fontSize": "11px",
                                "fontWeight": "800",
                                "letterSpacing": "2px",
                                "color": ACCENT_GREEN
                            }
                        ),

                        html.H2(
                            "Candidate Ranking",
                            style={
                                "fontSize": "28px",
                                "margin": "5px 0 5px",
                                "fontWeight": "800"
                            }
                        ),

                        html.P(
                            "Profiles ranked by predicted extroversion probability.",
                            style={
                                "color": TEXT_MUTED,
                                "margin": "0"
                            }
                        )

                    ],

                    style={
                        "marginBottom": "15px"
                    }
                ),

                dcc.Graph(

                    figure=ranking_fig,

                    config={
                        "responsive": True,
                        "displayModeBar": False
                    }
                ),

                html.H3(

                    "Top 20 Extrovert Profile Ranking",

                    style={
                        "marginTop":
                            "25px",
                        "marginBottom":
                            "15px"
                    }
                ),

                html.Table(

                    [

                        html.Thead(
                            html.Tr(
                                ranking_table_header
                            )
                        ),

                        html.Tbody(
                            ranking_table_rows
                        )

                    ],

                    style={

                        "width":
                            "100%",

                        "textAlign":
                            "center",

                        "borderCollapse":
                            "collapse",

                        "backgroundColor":
                            "rgba(15, 23, 42, 0.4)",

                        "borderRadius":
                            "8px",

                        "overflow":
                            "hidden"
                    }
                )

            ])

        ])


    # ========================================================
    # PREDICTION
    # ========================================================

    elif tab == "prediction":

        input_fields = []


        for feature in X.columns:

            input_fields.append(

                html.Div(

                    [

                        html.Label(

                            feature,

                            style={

                                "fontWeight":
                                    "600",

                                "marginBottom":
                                    "7px",

                                "display":
                                    "block",

                                "color":
                                    TEXT_COLOR,

                                "fontSize":
                                    "13px"
                            }
                        ),


                        dcc.Input(

                            id={
                                "type":
                                    "input",

                                "feature":
                                    feature
                            },

                            type="number",

                            placeholder=
                                f"Enter {feature}",

                            style={

                                "width":
                                    "100%",

                                "height":
                                    "48px",

                                "padding":
                                    "0 14px",

                                "backgroundColor":
                                    DARK_BG,

                                "border":
                                    "1px solid #334155",

                                "borderRadius":
                                    "10px",

                                "color":
                                    TEXT_COLOR,

                                "boxSizing":
                                    "border-box",

                                "fontSize":
                                    "15px",

                                "fontFamily":
                                    "Arial, sans-serif",

                                "outline":
                                    "none"
                            }
                        )

                    ],

                    style={

                        "width":
                            "100%",

                        "minWidth":
                            "0"
                    }
                )
            )


        return html.Div(

            [

                html.H2(

                    "Inference & Prediction Console",

                    style={

                        "color":
                            ACCENT_BLUE,

                        "margin":
                            "0 0 10px 0",

                        "fontSize":
                            "25px",

                        "fontWeight":
                            "700"
                    }
                ),


                html.P(

                    "Enter the behavioral values below "
                    "to predict the personality type.",

                    style={

                        "color":
                            TEXT_MUTED,

                        "margin":
                            "0 0 28px 0",

                        "fontSize":
                            "14px"
                    }
                ),


                html.Div(

                    input_fields,

                    className="prediction-grid",

                    style={

                        "display":
                            "grid",

                        "gridTemplateColumns":
                            "repeat(3, minmax(220px, 1fr))",

                        "gap":
                            "22px 20px",

                        "width":
                            "100%",

                        "marginBottom":
                            "30px"
                    }
                ),


                html.Button(

                    "Run Prediction Model",

                    id="predict-button",

                    n_clicks=0,

                    className="premium-btn",

                    style={

                        "padding":
                            "14px 32px",

                        "fontSize":
                            "15px",

                        "fontWeight":
                            "700",

                        "cursor":
                            "pointer",

                        "background":
                            "linear-gradient("
                            "135deg,"
                            "#38bdf8,"
                            "#22d3ee)",

                        "color":
                            DARK_BG,

                        "border":
                            "none",

                        "borderRadius":
                            "10px",

                        "marginTop":
                            "5px",

                        "boxShadow":
                            "0 8px 25px "
                            "rgba(56,189,248,.20)"
                    }
                ),


                html.Div(

                    id="prediction-result",

                    style={

                        "marginTop":
                            "25px",

                        "width":
                            "100%"
                    }
                )

            ],

            style={

                "width":
                    "100%",

                "boxSizing":
                    "border-box"
            }
        )


# ============================================================
# 20. PREDICTION CALLBACK
# ============================================================

@app.callback(

    Output(
        "prediction-result",
        "children"
    ),

    Input(
        "predict-button",
        "n_clicks"
    ),

    [

        State(

            {
                "type":
                    "input",

                "feature":
                    feature
            },

            "value"

        )

        for feature in X.columns

    ],

    prevent_initial_call=True
)

def predict_personality(

    n_clicks,
    *values
):

    if not n_clicks:

        return ""


    if any(
        value is None
        for value in values
    ):

        return html.Div(

            "⚠️ Please fill in all required "
            "feature fields before clicking Predict.",

            style={

                "color":
                    "#f43f5e",

                "padding":
                    "15px",

                "backgroundColor":
                    "rgba(244, 63, 94, 0.1)",

                "borderRadius":
                    "8px"
            }
        )


    try:

        new_data = pd.DataFrame(

            [values],

            columns=X.columns
        )

        new_data = new_data.astype(float)


        prediction = rf_model.predict(
            new_data
        )[0]


        probability = (

            rf_model
            .predict_proba(new_data)[0][1]
        )


        personality = (

            "Extrovert"

            if prediction == 1

            else "Introvert"
        )


        accent_color = (

            ACCENT_BLUE

            if personality == "Extrovert"

            else ACCENT_PURPLE
        )


        return html.Div(

            [

                html.Div(

                    [

                        html.Span(

                            "Predicted Outcome: ",

                            style={
                                "fontSize":
                                    "18px",

                                "color":
                                    TEXT_MUTED
                            }
                        ),

                        html.Span(

                            personality,

                            style={
                                "fontSize":
                                    "26px",

                                "fontWeight":
                                    "bold",

                                "color":
                                    accent_color
                            }
                        )

                    ]
                ),


                html.Div(

                    [

                        html.Span(

                            "Model Extroversion Confidence: ",

                            style={
                                "fontSize":
                                    "16px",

                                "color":
                                    TEXT_MUTED
                            }
                        ),

                        html.Span(

                            f"{probability:.2%}",

                            style={
                                "fontSize":
                                    "20px",

                                "fontWeight":
                                    "600",

                                "color":
                                    TEXT_COLOR
                            }
                        )

                    ],

                    style={
                        "marginTop":
                            "8px"
                    }
                )

            ],

            style={

                "padding":
                    "20px",

                "backgroundColor":
                    "rgba(15, 23, 42, 0.6)",

                "borderRadius":
                    "12px",

                "border":
                    f"1px solid {accent_color}"
            }
        )


    except Exception as e:

        return html.Div(

            f"❌ Prediction error: {str(e)}",

            style={

                "color":
                    "#f43f5e",

                "padding":
                    "15px",

                "backgroundColor":
                    "rgba(244, 63, 94, 0.1)",

                "borderRadius":
                    "8px"
            }
        )


# ============================================================
# 21. RUN APP
# ============================================================

import os

server = app.server

if name == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050))
    )
