from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Files are loaded from the same folder as this app.py file.
PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_FOLDER / "disaster_damage_model_local.pkl"
DATA_PATH = PROJECT_FOLDER / "disaster_dataset.csv"


st.set_page_config(
    page_title="Disaster Economic Damage Predictor",
    page_icon="🌍",
    layout="wide",
)


# Clean, professional styling for the page.
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f7fb;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        section[data-testid="stSidebar"] {
            background-color: #102338;
        }
        section[data-testid="stSidebar"] * {
            color: #edf4fb;
        }
        .hero {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            color: white;
            background: linear-gradient(120deg, #102338 0%, #176b78 100%);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            color: white;
            margin: 0.25rem 0 0.5rem 0;
            font-size: 2.3rem;
        }
        .hero p {
            color: #d8e9ef;
            margin: 0;
            font-size: 1.05rem;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e3eaf2;
            border-radius: 16px;
            padding: 1rem 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_disaster_types(data_path):
    dataset = pd.read_csv(data_path)

    if "disaster_type" not in dataset.columns:
        raise ValueError("The dataset is missing the 'disaster_type' column.")

    return sorted(
        dataset["disaster_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


with st.sidebar:
    st.markdown("## 🌍 Disaster Impact")
    st.caption("Economic damage estimation")
    st.markdown("---")
    st.markdown("### About this project")
    st.write(
        "This app connects a trained machine-learning model "
        "to a simple disaster scenario form."
    )
    st.markdown("---")
    st.caption("Local prototype • Random Forest model")


st.markdown(
    """
    <div class="hero">
        <div>DISASTER ANALYTICS</div>
        <h1>Economic Damage Predictor</h1>
        <p>Enter a disaster scenario to estimate its potential economic impact.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Prototype notice: this model was trained on simulated data. "
    "Its estimate is not a verified real-world loss forecast."
)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH.name}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    disaster_types = load_disaster_types(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()

if not disaster_types:
    st.error("No disaster types were found in the dataset.")
    st.stop()


input_column, result_column = st.columns([1, 1.2], gap="large")


with input_column:
    with st.container(border=True):
        st.subheader("Scenario details")
        st.caption("Enter the information for the disaster scenario.")

        with st.form("disaster_prediction_form"):
            disaster_type = st.selectbox(
                "Disaster type",
                options=disaster_types,
            )

            severity = st.slider(
                "Severity (1–5)",
                min_value=1.0,
                max_value=5.0,
                value=4.0,
                step=0.5,
            )

            affected_population = st.number_input(
                "Affected population",
                min_value=1,
                value=25000,
                step=1000,
            )

            infrastructure_vulnerability = st.slider(
                "Infrastructure vulnerability (1–5)",
                min_value=1.0,
                max_value=5.0,
                value=3.5,
                step=0.5,
            )

            submitted = st.form_submit_button(
                "Estimate economic damage",
                type="primary",
                use_container_width=True,
            )


with result_column:
    with st.container(border=True):
        st.subheader("Prediction")
        st.caption("Your model estimate will appear here.")

        if submitted:
            input_data = pd.DataFrame(
                [
                    {
                        "disaster_type": disaster_type,
                        "severity": severity,
                        "affected_population": affected_population,
                        "infrastructure_vulnerability": (
                            infrastructure_vulnerability
                        ),
                    }
                ]
            )

            try:
                prediction = float(model.predict(input_data)[0])

                st.metric(
                    label="Estimated economic damage",
                    value=f"${prediction:,.2f}",
                )
                st.success("Prediction generated.")

                st.markdown("#### Scenario summary")
                summary_left, summary_right = st.columns(2)
                summary_left.write(f"**Disaster:** {disaster_type}")
                summary_left.write(f"**Severity:** {severity:g} / 5")
                summary_right.write(
                    f"**Affected population:** {affected_population:,}"
                )
                summary_right.write(
                    "**Infrastructure vulnerability:** "
                    f"{infrastructure_vulnerability:g} / 5"
                )

            except Exception as error:
                st.error(f"Could not generate a prediction: {error}")
        else:
            st.write("Complete the form and select **Estimate economic damage**.")

st.divider()
st.caption(
    "For demonstration purposes only. Always validate estimates against "
    "reliable real-world data before making decisions."
)