I developed an end-to-end machine learning application that predicts the **probability of loan default** for a credit applicant and makes a **threshold-based loan approval decision**.
The model is trained separately and deployed using a **Streamlit web application** for real-time inference.

## Solution Overview

- **Model**: Decision Tree Classifier  
- **Approach**:
  - End-to-end sklearn `Pipeline`
  - Proper handling of missing values and categorical features
  - Class imbalance handling using `class_weight="balanced"`
  - Evaluation using **ROC-AUC**
  - Threshold-based business decision logic
- **Deployment**: Streamlit app for inference
