MLFLOW_TRACKING_URI = '../Models/mlruns'
MLFLOW_RUN_ID = "366b3b40b46344edb0a8d00c95a3884c"
CLUSTERS_YAML_PATH = "../Data/Processed/3_skills_clusters.yaml"

# -------------------------------------------------------------------------------

from JobPrediciton import JobPrediction
from flask import Flask, request, jsonify

#------------------------------------------

# Initiate API and JobPrediction object
app = Flask(__name__)
job_model = JobPrediction(mlflow_uri=MLFLOW_TRACKING_URI,
                          run_id=MLFLOW_RUN_ID,
                          clusters_yaml_path=CLUSTERS_YAML_PATH)

# Create Job prediction endpoint 
@app.route('/predict_jobs_probs', methods=['POST'])
def predict_jobs_probs():
    entry_skills = request.get_json()
    predictions = job_model.predict_job_probabilities(entry_skills).to_dict()
    return jsonify(predictions)

# Create Skills recommendation endpoint
@app.route('/recommend_new_skills', methods=['POST'])
def recommend_new_skills():
    request_details = request.get_json()
    entry_skills = request_details['entry_skills']
    target_job = request_details['target_job']

    recommended_skills = job_model.recommend_new_skills(entry_skills, target_job).to_dict()
    return jsonify(recommended_skills)


if __name__ == '__main__':
    app.run(port=5000)