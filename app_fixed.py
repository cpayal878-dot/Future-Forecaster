
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import uuid
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas



def create_placement_pdf(student_data, prediction, confidence):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Professional navy-blue header
    pdf.setFillColorRGB(0.08, 0.16, 0.32)
    pdf.rect(0, height - 150, width, 150, fill=1, stroke=0)

    pdf.setFillColorRGB(1, 1, 1)
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(45, height - 55, "FUTURE FORECASTER")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(45, height - 80, "Student Placement Assessment Report")

    report_date = datetime.now().strftime("%d %B %Y")
    report_id = str(uuid.uuid4())[:8].upper()

    pdf.setFillColorRGB(1, 1, 1)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(45, height - 130, f"Report Date: {report_date}")
    pdf.drawRightString(width - 45, height - 130, f"Report ID: {report_id}")
    

    pdf.setFont("Helvetica", 9)
    pdf.drawString(45, height - 110, "CAREER INSIGHTS | SKILL DEVELOPMENT")
    # Student name
    student_name = student_data.get("name", "Student")
    pdf.setFillColorRGB(0.2, 0.2, 0.2)
    pdf.setFont("Helvetica", 11)
    pdf.drawString(45, height - 165, f"Prepared for: {student_name}")
    # Result section
    y = height - 195
    pdf.setFillColorRGB(0.92, 0.95, 0.99)
    pdf.roundRect(40, y - 45, width - 80, 75, 10, fill=1, stroke=0)

    pdf.setFillColorRGB(0.08, 0.16, 0.32)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(55, y + 5, "PLACEMENT PREDICTION")

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(55, y - 20, f"Result: {prediction}")
    pdf.drawRightString(
        width - 55, y - 20, f"Model score: {confidence:.1f}%"
    )

    # Student details
    y -= 100
    pdf.setFillColorRGB(0.08, 0.16, 0.32)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(45, y, "Student Performance Details")
    y -= 25

    pdf.setStrokeColorRGB(0.2, 0.45, 0.75)
    pdf.line(45, y, width - 45, y)
    y -= 25

    pdf.setFont("Helvetica", 11)
    for key, value in student_data.items():
        if key.lower() != "name":
            pdf.setFillColorRGB(0.2, 0.2, 0.2)
            pdf.drawString(55, y, str(key))
            pdf.setFillColorRGB(0.05, 0.1, 0.2)
            pdf.drawRightString(width - 55, y, str(value))
            y -= 25

    # Disclaimer
    y -= 20
    pdf.setFillColorRGB(0.35, 0.35, 0.35)
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(45, y, "Important: This is a prototype prediction,")
    pdf.drawString(45, y - 14, "not a guarantee of actual placement.")

    # Footer
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(
        width / 2, 30, "Future Forecaster | Hackathon Prototype"
    )

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()




st.set_page_config(
    page_title=" Future Forecaster| Student Analytics",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111c32);
    color: #f1f5f9;
}
h1, h2, h3, p, label {
    color: #f1f5f9 !important;
}
[data-testid="stMetric"] {
    background: #17243a;
    border: 1px solid #30435f;
    padding: 18px;
    border-radius: 14px;
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
}
.stButton button {
    background: #0284c7;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓Future Forecaster ")
st.caption("STUDENT PLACEMENT PREDICTION & SKILL ANALYTICS")
st.divider()

model_path = Path("placement_model.pkl")

if not model_path.exists():
    st.error("Model file missing. Please run train_model.py first.")
    st.stop()

with open(model_path, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
features = saved["features"]

data_path = Path("students.csv")

if data_path.exists():
    df = pd.read_csv(data_path)
else:
    df = pd.DataFrame(columns=features)

st.subheader("📊 Placement Overview")

if not df.empty:
    predictions = model.predict(df[features])
    placed = int((predictions == 1).sum())
    total = len(df)
    rate = placed / total * 100
    needs_support = total - placed

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students Analyzed", total)
    c2.metric("Predicted Placed", placed)
    c3.metric("Predicted Placement Rate", f"{rate:.1f}%")
    c4.metric("Flagged for Support", needs_support)

    left, right = st.columns(2)

    with left:
        st.subheader("Placement Predictions")
        chart_data = pd.DataFrame({
            "Prediction": ["Placed", "Not Placed"],
            "Students": [
                placed,
                total - placed
            ]
        }).set_index("Prediction")
        st.bar_chart(chart_data)

    with right:
        st.subheader("Skill Score Overview")
        skill_cols = [
            "Aptitude Score",
            "Coding Score",
            "Communication Score"
        ]
        st.bar_chart(df[skill_cols].mean())

    st.caption(
        "Charts show predictions from synthetic demo data, "
        "not verified real-world placement outcomes."
    )
else:
    st.info("No student dataset found. Add students.csv to view analytics.")

st.divider()
st.subheader("🔍 Predict an Individual Student")

with st.form("student_form"):
    student_name = st.text_input("Student Name")
    col1, col2, col3 = st.columns(3)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
    with col2:
        aptitude = st.slider("Aptitude Score", 0, 100, 70)
        coding = st.slider("Coding Score", 0, 100, 65)
    with col3:
        communication = st.slider(
            "Communication Score", 0, 100, 70
        )
        internship = st.selectbox(
            "Internship Experience",
            ["No", "Yes"]
        )

    submitted = st.form_submit_button("✨ Predict Placement")

if submitted:
    student = pd.DataFrame([{
        "CGPA": cgpa,
        "Attendance": attendance,
        "Aptitude Score": aptitude,
        "Coding Score": coding,
        "Communication Score": communication,
        "Internship Experience": 1 if internship == "Yes" else 0
    }])[features]

    prediction = int(model.predict(student)[0])
    probability = float(model.predict_proba(student)[0][1]) * 100

    st.subheader("📌 Prediction Result")

    r1, r2 = st.columns(2)
    with r1:
        if prediction == 1:
            st.success("Model predicts: Placed")
        else:
            st.warning("Model predicts: Not Placed")

    with r2:
        st.metric("Demo Model Score", f"{probability:.1f}%")

    st.progress(int(round(probability)))

   
    tips = []

    if coding < 60:
        tips.append("Practice Python and solve coding problems regularly.")
    if aptitude < 60:
        tips.append("Improve aptitude with daily reasoning and quantitative practice.")
    if communication < 60:
        tips.append("Practice mock interviews and spoken communication.")
    if attendance < 75:
        tips.append("Improve class attendance and maintain consistent learning.")
    if internship == "No":
        tips.append("Explore internships and build practical project experience.")
    if cgpa < 7:
        tips.append("Strengthen subject fundamentals and work toward improving CGPA.")

    if tips:
        st.subheader("🎯 Your Improvement Plan")
        for tip in tips:
            st.write("• " + tip)
    else:
        st.success("Great work! Keep practising and preparing for interviews.")
    # Career Recommendations
    st.subheader("💼 Recommended Career Paths")

    career_roles = []

    if coding >= 70:
        career_roles.append("Junior Python Developer")
    if aptitude >= 65 and communication >= 60:
        career_roles.append("Business Analyst Trainee")
    if aptitude >= 60 and coding >= 50:
        career_roles.append("Junior Data Analyst / Data Analyst Intern")
    if communication >= 65 and coding < 50:
        career_roles.append("Customer Support / Technical Support Trainee")
    if coding < 50 and aptitude < 50:
        career_roles.append("Explore beginner-friendly internships while building core skills")

    if career_roles:
        for role in career_roles:
            st.write("✅ " + role)
    else:
        st.write("Explore entry-level roles based on your interests and improve your core skills.")

    # Skill Gap Analysis
    st.subheader("📊 Your Skill Gap Analysis")

    skills = {
        "Coding": coding,
        "Aptitude": aptitude,
        "Communication": communication,
        "Attendance": attendance,
    }

    weak_skills = []

    for skill, score in skills.items():
        if score < 60:
            weak_skills.append(skill)
            st.write(f"🔴 {skill}: {score} — Needs improvement")
        else:
            st.write(f"🟢 {skill}: {score} — Good progress")

    # Personalized Learning Roadmap
    st.subheader("🗺️ Your Next-Step Roadmap")

    roadmap = {
        "Coding": "Practise Python basics, functions, and small coding problems.",
        "Aptitude": "Practise quantitative aptitude and reasoning for 20 minutes daily.",
        "Communication": "Practise spoken English, introductions, and mock interviews.",
        "Attendance": "Follow a consistent study schedule and attend classes regularly.",
    }

    if weak_skills:
        for skill in weak_skills:
            st.write(f"📌 **{skill}:** {roadmap[skill]}")
    else:
        st.success("Great progress! Build projects and prepare for interviews.")




    st.caption(
        "This is an educational demonstration using synthetic data. "
        "The percentage is not a validated real-world placement probability."
    )

    st.subheader("📄 Download Placement Report")
    student_data = {
    "name": student_name,
    "CGPA": cgpa,
    "Attendance": attendance,
    "Aptitude Score": aptitude,
    "Coding Score": coding,
    "Communication Score": communication,
    "Internship Experience": internship,
}
    pdf_file = create_placement_pdf(
        student_data,
        "Placed" if prediction == 1 else "Not Placed",
        probability
    )
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_file,
        file_name="Future_Forecaster_Placement_Report.pdf",
        mime="application/pdf",
        key="download_pdf_report"
    )

st.divider()
st.caption("Future Forecaster | Hackathon Prototype | Built with Python & Streamlit")

# Multiple Student Analysis
st.divider()
st.header("📂 Upload & Analyze Student Data")

uploaded_file = st.file_uploader(
    "Upload a student dataset (CSV)",
    type=["csv"],
    key="student_csv_upload"
)

if uploaded_file is not None:
    try:
        import pandas as pd

        uploaded_df = pd.read_csv(uploaded_file)

        required_columns = [
            "CGPA",
            "Attendance",
            "Aptitude Score",
            "Coding Score",
            "Communication Score",
            "Internship Experience",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )
        elif uploaded_df.empty:
            st.warning("The uploaded CSV contains no student records.")
        else:
            st.success(
                f"Uploaded successfully: {len(uploaded_df)} student records"
            )

            st.subheader("👀 Preview Student Data")
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            st.subheader("📊 Dataset Summary")
            st.write("Number of students:", len(uploaded_df))
            st.write(
                "Average CGPA:",
                round(pd.to_numeric(uploaded_df["CGPA"], errors="coerce").mean(), 2)
            )

            # Predict placement for every uploaded student
            import pickle

            with open("placement_model.pkl", "rb") as file:
                saved_data = pickle.load(file)

            model = saved_data["model"]
            features = saved_data["features"]

            prediction_df = uploaded_df.copy()

            # Convert Internship Experience from Yes/No to 1/0
            prediction_df["Internship Experience"] = (
                prediction_df["Internship Experience"]
                .replace({
                    "Yes": 1,
                    "No": 0,
                    "yes": 1,
                    "no": 0
                })
            )

            # Ensure all model inputs are numeric
            for col in features:
                prediction_df[col] = pd.to_numeric(
                    prediction_df[col], errors="coerce"
                )

            if prediction_df[features].isnull().any().any():
                st.error(
                    "Some student values are missing or invalid. "
                    "Please check the uploaded CSV."
                )
            else:
                predictions = model.predict(prediction_df[features])
                probabilities = model.predict_proba(
                    prediction_df[features]
                ).max(axis=1)

                prediction_df["Predicted Placement"] = [
                    "Placed" if value == 1 else "Not Placed"
                    for value in predictions
                ]

                prediction_df["Model Confidence (%)"] = (
                    probabilities * 100
                ).round(2)

                st.subheader("🎯 Placement Predictions")
                st.dataframe(
                    prediction_df,
                    use_container_width=True
                )

                st.download_button(
                    "⬇️ Download Predictions Report",
                    data=prediction_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="future_forecaster_predictions.csv",
                    mime="text/csv",
                    key="download_predictions"
                )


                st.divider()
                st.subheader("📊 Placement Analytics Dashboard")

                placed_count = int(
                    (prediction_df["Predicted Placement"] == "Placed").sum()
                )
                not_placed_count = int(
                    (prediction_df["Predicted Placement"] == "Not Placed").sum()
                )
                total_students = len(prediction_df)

                placement_rate = (
                    placed_count / total_students * 100
                    if total_students else 0
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Students", total_students)
                col2.metric("Predicted Placed", placed_count)
                col3.metric(
                    "Predicted Placement Rate",
                    f"{placement_rate:.1f}%"
                )

                chart_df = pd.DataFrame({
                    "Prediction": ["Placed", "Not Placed"],
                    "Students": [placed_count, not_placed_count]
                })

                st.bar_chart(chart_df.set_index("Prediction"))

            st.download_button(
                "⬇️ Download Uploaded Data",
                data=uploaded_df.to_csv(index=False).encode("utf-8"),
                file_name="student_data_report.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
