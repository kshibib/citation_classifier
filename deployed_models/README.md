# Deployable Model Artifacts

The Streamlit app looks for model artifacts under `deployed_models/linear_svm/` first.

Required files:

- `deployed_models/linear_svm/model.joblib`
- `deployed_models/linear_svm/label_encoder.joblib`

If you retrain the model later, replace those two files with the new artifacts you want the deployed app to use.
