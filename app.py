import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Accident Prediction & Prevention",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        padding: 10px 0 5px 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .severity-fatal   { color: #FF4B4B; font-weight: 700; font-size: 1.4rem; }
    .severity-serious { color: #FF8C00; font-weight: 700; font-size: 1.4rem; }
    .severity-minor   { color: #00C09A; font-weight: 700; font-size: 1.4rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load Data & Model ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("accident_prediction.csv")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le_y = pickle.load(f)
    return model, encoders, le_y

df = load_data()
model, encoders, le_y = load_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car-accident.png", width=80)
    st.markdown("*Road Accident Prediction & Prevention System*")
    st.markdown("---")
    st.markdown("**Dataset Stats**")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("States Covered", df["State Name"].nunique())
    st.metric("Years", f"{df['Year'].min()} – {df['Year'].max()}")
    st.markdown("---")
    st.markdown("**Built with**")
    st.markdown(" Python · Scikit-learn · Streamlit · Plotly")
    st.markdown("---")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚨 Road Accident Prediction & Prevention</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">India Road Safety Analytics Dashboard · AI-Powered Severity Prediction</div>', unsafe_allow_html=True)

# ── KPI Row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Accidents", f"{len(df):,}")
with k2:
    fatal = (df["Accident Severity"] == "Fatal").sum()
    st.metric("Fatal", f"{fatal:,}", delta=f"{fatal/len(df)*100:.1f}%", delta_color="inverse")
with k3:
    serious = (df["Accident Severity"] == "Serious").sum()
    st.metric("Serious", f"{serious:,}", delta=f"{serious/len(df)*100:.1f}%", delta_color="inverse")
with k4:
    minor = (df["Accident Severity"] == "Minor").sum()
    st.metric("Minor", f"{minor:,}", delta=f"{minor/len(df)*100:.1f}%", delta_color="off")
with k5:
    alcohol = (df["Alcohol Involvement"] == "Yes").sum()
    st.metric("Alcohol Involved", f"{alcohol:,}", delta=f"{alcohol/len(df)*100:.1f}%", delta_color="inverse")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗺️ State Analysis", "⏰ Time & Weather", "🔬 Risk Factors", "🤖 Predict Severity"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Accident Severity Distribution")
    c1, c2 = st.columns(2)

    with c1:
        sev_counts = df["Accident Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"Fatal": "#FF4B4B", "Serious": "#FF8C00", "Minor": "#00C09A"}
        fig = px.pie(
            sev_counts, names="Severity", values="Count",
            color="Severity", color_discrete_map=color_map,
            hole=0.45, title="Severity Breakdown"
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=True, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        road_sev = df.groupby(["Road Type", "Accident Severity"]).size().reset_index(name="Count")
        fig2 = px.bar(
            road_sev, x="Road Type", y="Count", color="Accident Severity",
            color_discrete_map=color_map, barmode="group",
            title="Severity by Road Type"
        )
        fig2.update_layout(height=360, xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Yearly Accident Trend")
    yearly = df.groupby(["Year", "Accident Severity"]).size().reset_index(name="Count")
    fig3 = px.line(
        yearly, x="Year", y="Count", color="Accident Severity",
        color_discrete_map=color_map, markers=True,
        title="Accidents per Year by Severity"
    )
    fig3.update_layout(height=340)
    st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        veh_sev = df.groupby(["Vehicle Type Involved", "Accident Severity"]).size().reset_index(name="Count")
        fig4 = px.bar(
            veh_sev, x="Vehicle Type Involved", y="Count", color="Accident Severity",
            color_discrete_map=color_map, barmode="stack",
            title="Accidents by Vehicle Type"
        )
        fig4.update_layout(height=360, xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        loc_sev = df.groupby(["Accident Location Details", "Accident Severity"]).size().reset_index(name="Count")
        fig5 = px.bar(
            loc_sev, x="Accident Location Details", y="Count", color="Accident Severity",
            color_discrete_map=color_map, barmode="group",
            title="Accidents by Location Type"
        )
        fig5.update_layout(height=360)
        st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("State-wise Accident Analysis")

    state_total = df.groupby("State Name").size().reset_index(name="Total Accidents").sort_values("Total Accidents", ascending=False)
    fig_state = px.bar(
        state_total.head(20), x="Total Accidents", y="State Name", orientation="h",
        color="Total Accidents", color_continuous_scale="Reds",
        title="Top 20 States by Total Accidents"
    )
    fig_state.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_state, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        state_fatal = df[df["Accident Severity"] == "Fatal"].groupby("State Name").size().reset_index(name="Fatal Count").sort_values("Fatal Count", ascending=False).head(15)
        fig_sf = px.bar(
            state_fatal, x="Fatal Count", y="State Name", orientation="h",
            color="Fatal Count", color_continuous_scale="OrRd",
            title="Top 15 States by Fatal Accidents"
        )
        fig_sf.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_sf, use_container_width=True)

    with c2:
        state_road = df.groupby(["State Name", "Road Type"]).size().reset_index(name="Count")
        top_states = state_total.head(10)["State Name"].tolist()
        state_road_top = state_road[state_road["State Name"].isin(top_states)]
        fig_sr = px.bar(
            state_road_top, x="State Name", y="Count", color="Road Type",
            barmode="stack", title="Road Type Distribution (Top 10 States)"
        )
        fig_sr.update_layout(height=420, xaxis_tickangle=-30)
        st.plotly_chart(fig_sr, use_container_width=True)

    st.subheader("City-level Breakdown")
    city_counts = df[df["City Name"] != "Unknown"].groupby("City Name").size().reset_index(name="Count").sort_values("Count", ascending=False).head(15)
    if not city_counts.empty:
        fig_city = px.bar(
            city_counts, x="City Name", y="Count",
            color="Count", color_continuous_scale="Blues",
            title="Top Cities by Accident Count (excluding Unknown)"
        )
        fig_city.update_layout(height=350, xaxis_tickangle=-30)
        st.plotly_chart(fig_city, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — TIME & WEATHER
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Time & Weather Patterns")

    df["Hour"] = df["Time of Day"].apply(lambda x: int(str(x).split(":")[0]) if ":" in str(x) else 0)

    c1, c2 = st.columns(2)
    with c1:
        hourly = df.groupby(["Hour", "Accident Severity"]).size().reset_index(name="Count")
        color_map = {"Fatal": "#FF4B4B", "Serious": "#FF8C00", "Minor": "#00C09A"}
        fig_h = px.line(
            hourly, x="Hour", y="Count", color="Accident Severity",
            color_discrete_map=color_map, markers=True,
            title="Accidents by Hour of Day"
        )
        fig_h.update_layout(height=360, xaxis=dict(tickmode="linear", tick0=0, dtick=2))
        st.plotly_chart(fig_h, use_container_width=True)

    with c2:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = df.groupby(["Day of Week", "Accident Severity"]).size().reset_index(name="Count")
        day_counts["Day of Week"] = pd.Categorical(day_counts["Day of Week"], categories=day_order, ordered=True)
        day_counts = day_counts.sort_values("Day of Week")
        fig_d = px.bar(
            day_counts, x="Day of Week", y="Count", color="Accident Severity",
            color_discrete_map=color_map, barmode="group",
            title="Accidents by Day of Week"
        )
        fig_d.update_layout(height=360)
        st.plotly_chart(fig_d, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        mon_counts = df.groupby(["Month", "Accident Severity"]).size().reset_index(name="Count")
        mon_counts["Month"] = pd.Categorical(mon_counts["Month"], categories=month_order, ordered=True)
        mon_counts = mon_counts.sort_values("Month")
        fig_m = px.line(
            mon_counts, x="Month", y="Count", color="Accident Severity",
            color_discrete_map=color_map, markers=True,
            title="Monthly Accident Trend"
        )
        fig_m.update_layout(height=360, xaxis_tickangle=-30)
        st.plotly_chart(fig_m, use_container_width=True)

    with c4:
        weather_sev = df.groupby(["Weather Conditions", "Accident Severity"]).size().reset_index(name="Count")
        fig_w = px.bar(
            weather_sev, x="Weather Conditions", y="Count", color="Accident Severity",
            color_discrete_map=color_map, barmode="stack",
            title="Accidents by Weather Conditions"
        )
        fig_w.update_layout(height=360)
        st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Lighting Conditions Impact")
    light_sev = df.groupby(["Lighting Conditions", "Accident Severity"]).size().reset_index(name="Count")
    fig_l = px.bar(
        light_sev, x="Lighting Conditions", y="Count", color="Accident Severity",
        color_discrete_map=color_map, barmode="group",
        title="Accidents by Lighting Condition"
    )
    fig_l.update_layout(height=340)
    st.plotly_chart(fig_l, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK FACTORS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Driver & Road Risk Factors")
    color_map = {"Fatal": "#FF4B4B", "Serious": "#FF8C00", "Minor": "#00C09A"}

    c1, c2 = st.columns(2)
    with c1:
        fig_age = px.histogram(
            df, x="Driver Age", color="Accident Severity",
            color_discrete_map=color_map, barmode="overlay",
            nbins=25, opacity=0.75,
            title="Driver Age vs Accident Severity"
        )
        fig_age.update_layout(height=360)
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        alc_sev = df.groupby(["Alcohol Involvement", "Accident Severity"]).size().reset_index(name="Count")
        fig_alc = px.bar(
            alc_sev, x="Alcohol Involvement", y="Count", color="Accident Severity",
            color_discrete_map=color_map, barmode="group",
            title="Alcohol Involvement vs Severity"
        )
        fig_alc.update_layout(height=360)
        st.plotly_chart(fig_alc, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_speed = px.box(
            df, x="Accident Severity", y="Speed Limit (km/h)",
            color="Accident Severity", color_discrete_map=color_map,
            title="Speed Limit Distribution by Severity"
        )
        fig_speed.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_speed, use_container_width=True)

    with c4:
        road_cond = df.groupby(["Road Condition", "Accident Severity"]).size().reset_index(name="Count")
        fig_rc = px.bar(
            road_cond, x="Road Condition", y="Count", color="Accident Severity",
            color_discrete_map=color_map, barmode="stack",
            title="Road Condition vs Severity"
        )
        fig_rc.update_layout(height=360, xaxis_tickangle=-20)
        st.plotly_chart(fig_rc, use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    feature_cols = ['State Name','Month','Day of Week','Hour','Number of Vehicles Involved',
                    'Vehicle Type Involved','Weather Conditions','Road Type','Road Condition',
                    'Lighting Conditions','Traffic Control Presence','Speed Limit (km/h)',
                    'Driver Age','Driver Gender','Driver License Status','Alcohol Involvement',
                    'Accident Location Details']
    fi = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_})
    fi = fi.sort_values("Importance", ascending=True)
    fig_fi = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Reds",
        title="Feature Importance — What Drives Severity?"
    )
    fig_fi.update_layout(height=500)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Traffic Control vs Severity")
    tc_sev = df.groupby(["Traffic Control Presence", "Accident Severity"]).size().reset_index(name="Count")
    fig_tc = px.bar(
        tc_sev, x="Traffic Control Presence", y="Count", color="Accident Severity",
        color_discrete_map=color_map, barmode="group",
        title="Traffic Control Presence vs Accident Severity"
    )
    fig_tc.update_layout(height=340)
    st.plotly_chart(fig_tc, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 Predict Accident Severity")
    st.markdown("Enter the road and driver conditions below to get an AI-based severity prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Location & Road**")
        state = st.selectbox("State", sorted(df["State Name"].unique()))
        road_type = st.selectbox("Road Type", df["Road Type"].unique())
        road_cond = st.selectbox("Road Condition", df["Road Condition"].unique())
        accident_loc = st.selectbox("Accident Location", df["Accident Location Details"].unique())
        speed = st.slider("Speed Limit (km/h)", 30, 120, 60)

    with col2:
        st.markdown("**🌦️ Environment**")
        weather = st.selectbox("Weather", df["Weather Conditions"].unique())
        lighting = st.selectbox("Lighting", df["Lighting Conditions"].unique())
        traffic_ctrl = st.selectbox("Traffic Control", ["Signals", "Signs", "Police Checkpost", "None"])
        month = st.selectbox("Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
        day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        hour = st.slider("Hour of Day", 0, 23, 12)

    with col3:
        st.markdown("**🧑 Driver & Vehicle**")
        driver_age = st.slider("Driver Age", 18, 70, 30)
        driver_gender = st.selectbox("Driver Gender", ["Male", "Female"])
        license_status = st.selectbox("License Status", ["Valid", "Expired", "No License", "Unknown"])
        alcohol = st.selectbox("Alcohol Involvement", ["No", "Yes"])
        vehicle = st.selectbox("Vehicle Type", df["Vehicle Type Involved"].unique())
        num_vehicles = st.slider("Number of Vehicles", 1, 10, 2)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Severity", type="primary", use_container_width=True)

    if predict_btn:
        input_dict = {
            "State Name": state, "Month": month, "Day of Week": day,
            "Hour": hour, "Number of Vehicles Involved": num_vehicles,
            "Vehicle Type Involved": vehicle, "Weather Conditions": weather,
            "Road Type": road_type, "Road Condition": road_cond,
            "Lighting Conditions": lighting, "Traffic Control Presence": traffic_ctrl,
            "Speed Limit (km/h)": speed, "Driver Age": driver_age,
            "Driver Gender": driver_gender, "Driver License Status": license_status,
            "Alcohol Involvement": alcohol, "Accident Location Details": accident_loc
        }

        input_df = pd.DataFrame([input_dict])

        cat_cols = ['State Name','Month','Day of Week','Vehicle Type Involved',
                    'Weather Conditions','Road Type','Road Condition','Lighting Conditions',
                    'Traffic Control Presence','Driver Gender','Driver License Status',
                    'Alcohol Involvement','Accident Location Details']

        for col in cat_cols:
            le = encoders[col]
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0

        feature_cols = ['State Name','Month','Day of Week','Hour','Number of Vehicles Involved',
                        'Vehicle Type Involved','Weather Conditions','Road Type','Road Condition',
                        'Lighting Conditions','Traffic Control Presence','Speed Limit (km/h)',
                        'Driver Age','Driver Gender','Driver License Status','Alcohol Involvement',
                        'Accident Location Details']

        X_input = input_df[feature_cols]
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        pred_label = le_y.inverse_transform([pred])[0]
        classes = le_y.classes_

        r1, r2 = st.columns([1, 2])
        with r1:
            if pred_label == "Fatal":
                st.markdown(f'<div style="background:#2a0a0a;border:2px solid #FF4B4B;border-radius:12px;padding:24px;text-align:center"><p style="color:#aaa;margin:0">Predicted Severity</p><p class="severity-fatal" style="font-size:2rem;margin:8px 0">⚠️ FATAL</p><p style="color:#888;font-size:0.85rem">High risk scenario</p></div>', unsafe_allow_html=True)
            elif pred_label == "Serious":
                st.markdown(f'<div style="background:#2a1500;border:2px solid #FF8C00;border-radius:12px;padding:24px;text-align:center"><p style="color:#aaa;margin:0">Predicted Severity</p><p class="severity-serious" style="font-size:2rem;margin:8px 0">⚡ SERIOUS</p><p style="color:#888;font-size:0.85rem">Moderate-high risk</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:#0a2a1a;border:2px solid #00C09A;border-radius:12px;padding:24px;text-align:center"><p style="color:#aaa;margin:0">Predicted Severity</p><p class="severity-minor" style="font-size:2rem;margin:8px 0">✅ MINOR</p><p style="color:#888;font-size:0.85rem">Lower risk scenario</p></div>', unsafe_allow_html=True)

        with r2:
            st.markdown("**Model Confidence (Class Probabilities)**")
            prob_df = pd.DataFrame({"Severity": classes, "Probability": proba})
            color_map2 = {"Fatal": "#FF4B4B", "Serious": "#FF8C00", "Minor": "#00C09A"}
            fig_prob = px.bar(
                prob_df, x="Severity", y="Probability", color="Severity",
                color_discrete_map=color_map2, text="Probability"
            )
            fig_prob.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_prob.update_layout(height=260, showlegend=False, yaxis=dict(tickformat=".0%", range=[0, 1]))
            st.plotly_chart(fig_prob, use_container_width=True)

        # Prevention tips
        st.markdown("---")
        st.markdown("### 🛡️ Prevention Recommendations")
        tips = []
        if alcohol == "Yes":
            tips.append("🚫 **Do not drive under alcohol influence.** This is a leading cause of fatal accidents.")
        if lighting in ["Dark", "Dusk", "Dawn"]:
            tips.append("💡 **Use headlights and drive at reduced speed** in low-visibility conditions.")
        if weather in ["Foggy", "Stormy", "Rainy"]:
            tips.append(f"🌧️ **Adverse weather ({weather}) detected.** Maintain greater following distance and slow down.")
        if road_cond in ["Under Construction", "Damaged"]:
            tips.append(f"🚧 **Road condition is {road_cond}.** Be alert for unexpected hazards and detours.")
        if speed > 80:
            tips.append(f"⚡ **Speed limit of {speed} km/h is high.** Reduce speed, especially at curves and intersections.")
        if accident_loc in ["Curve", "Intersection", "Bridge"]:
            tips.append(f"📍 **{accident_loc} locations are high-risk.** Slow down and check for oncoming traffic.")
        if driver_age < 25:
            tips.append("🎓 **Young driver detected.** Inexperienced drivers should avoid night driving and high-speed roads.")
        if traffic_ctrl == "None":
            tips.append("🚦 **No traffic control present.** Treat all intersections as uncontrolled — yield and proceed carefully.")
        if not tips:
            tips.append("✅ **Relatively safer conditions detected.** Always stay alert and follow traffic rules.")

        for tip in tips:
            st.markdown(f"- {tip}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.85rem'>"
    "RAKSHAK AI · Road Accident Prediction & Prevention System · India · "
    "Random Forest Model · Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True
)
