# UIDAI Data Hackathon 2026  
## Identifying Aadhaar Exclusion Risk Using Enrolment & Update Patterns

📊 Data-Driven Risk Index for Aadhaar Service Exclusion  
🏛️ UIDAI × NIC × MeitY Hackathon 2026  

---

## 📌 Project Overview

Aadhaar is India’s foundational digital identity system enabling access to welfare schemes, financial services, and digital governance.  
Despite high coverage, **service exclusion risks** persist due to enrolment update delays, biometric failures, and age-transition bottlenecks.

This project develops a **State-Level Aadhaar Exclusion Risk Index** using anonymised UIDAI enrolment and update datasets to help policymakers identify **high-risk regions** requiring targeted intervention.

---

## 🎯 Problem Statement

Aadhaar exclusion is not uniformly distributed across India.  
Certain states experience higher stress due to:

- High biometric update failures  
- Administrative overload in demographic corrections  
- Child-to-adult transition challenges  

**Objective:**  
To identify and rank Indian states based on Aadhaar exclusion vulnerability using data-driven indicators.

---

## 🧠 Solution Approach

We construct a **composite Aadhaar Exclusion Risk Index (0–1 scale)** using three core dimensions:

1. **Demographic Update Stress**  
2. **Biometric Update Stress**  
3. **Child Transition Risk**

All indicators are:
- Aggregated at **State × Year** level  
- Normalized using Min-Max Scaling  
- Combined using policy-oriented weights  

---

## 📂 Data Sources

All datasets are **official anonymised UIDAI datasets**:

| Dataset | Description |
|------|------------|
| Aadhaar Enrolment Data | Age-wise enrolment counts |
| Demographic Update Data | Age-wise demographic corrections |
| Biometric Update Data | Age-wise biometric updates |

⚠️ No personally identifiable information (PII) is used.

---

## ⚙️ Methodology

### 1️⃣ Data Ingestion
- Automated ingestion of multiple CSV files
- Temporal alignment using date parsing

### 2️⃣ Data Cleaning & Standardization
- State name normalization (legacy & UT mergers)
- Invalid records removed
- Defensive handling of missing values

### 3️⃣ State-Year Aggregation
All metrics aggregated to:


---

## 🧮 Feature Engineering

Key metrics constructed:

- **Total Enrolment**
- **Demographic Update Rate**
- **Biometric Update Rate**
- **Child Transition Risk**

---

## 📐 Normalization

Min-Max Scaling applied:


X_norm = (X − X_min) / (X_max − X_min)


Ensures:
- Equal comparability across metrics  
- Stable and interpretable composite index  

---

## 🧮 Aadhaar Exclusion Risk Index

Weighted combination:

| Component | Weight |
|--------|--------|
| Demographic Update Stress | 0.30 |
| Biometric Update Stress | 0.30 |
| Child Transition Risk | 0.40 |

Final index normalized to **0–1 range**.

---

## 📊 Outputs

- State-wise Aadhaar Exclusion Risk Ranking  
- Identification of high-risk and low-risk states  
- Distribution analysis of exclusion risk  
- Interactive **Power BI Dashboard**

---

## 📈 Power BI Dashboard

The dashboard presents:
- Overall Aadhaar Exclusion Risk Index
- State-wise comparison
- Risk component breakdown
- Temporal trends
- Policy-ready insights

File:uidai_hackathon_dashboard.pbix

---

## 🗂️ Project Structure

├── api_data_aadhar_enrolment/
├── api_data_aadhar_demographic/
├── api_data_aadhar_biometric/
├── uidai_hackathon_2026.ipynb
├── uidai_hackathon_2026_python_script.py
├── uidai_hackathon_2026.html
├── uidai_hackathon_dashboard.pbix
└── README.md


---

## 🛠️ Technologies Used

- Python  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Jupyter Notebook  
- Power BI  

---

## 📌 Key Insights

- Aadhaar exclusion risk is **systemic**, not population-size driven  
- Several low-enrolment states show **high exclusion vulnerability**  
- Child biometric transition is the **most critical risk pathway**  
- Policy focus should shift from scale to **service quality & update capacity**

---

## 🏛️ Policy Relevance

This analysis supports:
- Targeted Aadhaar infrastructure investment  
- Focused biometric update drives  
- Child transition risk mitigation  
- Evidence-based governance decisions  

---

## ⚠️ Disclaimer

This project is created **solely for academic and hackathon purposes** using anonymised public datasets.  
It does **not** represent official UIDAI policy or operational decisions.

---

## 👤 Author

**Sohom Roy**  
Data Analyst | UIDAI Data Hackathon 2026  

---

## ⭐ Acknowledgements

- Unique Identification Authority of India (UIDAI)  
- National Informatics Centre (NIC)  
- Ministry of Electronics and Information Technology (MeitY)  
