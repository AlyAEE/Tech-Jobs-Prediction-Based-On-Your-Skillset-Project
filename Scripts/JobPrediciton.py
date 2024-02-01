OG_DATA_PKL    =  "data.pkl"
LOG_MODEL_PKL   =  "model.pkl"
LOG_METRICS_PKL =  "metrics.pkl"

#-------------------------------------------------------------

import os
import sklearn
import pickle
import yaml

import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

#-------------------------------------------------------------

class JobPrediction:

    """ A class for predicting the probability of a job from skills"""

    # ========================================================
    # ***********    Initialization Functions    ************
    # ========================================================

    #Constructor
    def __init__(self, mlflow_uri, run_id, clusters_yaml_path):

        # Constants
        self.tracking_uri  = mlflow_uri
        self.run_id        = run_id

        # Retrieve model, features and labels
        mlflow_objs         = self.load_mlflow_objs()
        self.model          = mlflow_objs[0]
        self.features_names = mlflow_objs[1]
        self.targets_names  = mlflow_objs[2]

        # Retrieve skills_cluster Dataframe
        self.skills_clusters_df   = self.load_clusters_config(clusters_yaml_path)

    def load_mlflow_objs():
        """Load artifacts from mlflow run"""
        # Initialize client and experiment
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()

        run = mlflow.get_run(self.run_id)
        artifact_path = run.info.artifact_uri.replace("file:///", "")
    
        # Load data pkl
        data_path = os.path.join(artifact_path, LOG_DATA_PKL)
        with open(data_path, 'rb') as handle:
            data_pkl = pickle.load(handle)

        # Load model
        model_path = os.path.join(artifact_path, LOG_MODEL_PKL)
        with open(model_path, "rb") as handle:
            model_pkl = pickle.load(handle)

        # Return model, features and labels
        return model_pkl["model_object"], \
               data_pkl["features_names"], \
               data_pkl["targets_names"]
    
    def load_clusters_config(self, clusters_yaml_path):
        """Load the Engineered Skills Clusters"""

        # Load skills clusters
        with open(clusters_yaml_path, "r") as handle:
            clusters_config = yaml.safe_load(handle)
        
        # Reformat into Dataframe
        clusters_df = [(cluster_name, cluster_skill)
                   for cluster_name, cluster_skills in clusters_config.items()
                   for cluster_skill in cluster_skills]

        clusters_df = pd.DataFrame(clusters_df, 
                                   columns=["cluster_name", "skill"])
        return clusters_df
    
    # ========================================================
    # ***********           Getters               ************
    # ========================================================

    def get_all_skills(self):
        return self.features_names

    def get_all_jobs(self):
        return self.targets_names
    
    # ========================================================
    # **************    Prediction Functions    **************  
    # ========================================================

    def create_features_array(self, skills_entry):
        """Create the features array from a list of skills entry"""

        # Create clusters features array
        def create_clusters_features(self, skills_entry):
            sample_clusters = self.skills_clusters_df.copy()
            sample_clusters['skills_entry'] = sample_clusters['skill'].isin(skills_entry)
            clusters_features = sample_clusters.groupby("cluster_name")["skills_entry"].sum()
            return clusters_features
        
        # Create skills features array, exclude skills_cluster features
        def create_skills_features(self, skills_entry, exclude_features):
            all_features = pd.Series(self.features_names.copy())
            skills_features = all_features[~all_features.isin(exclude_features)]
            skills_features = pd.Series(skills_features.isin(skills_entry).astype(int).tolist(),
                                        index = skills_features)
            return skills_features
        
        # Combine skills and clusters features, then sort 
        def combine_features(skills_features,clusters_features):
            features = pd.concat([skills_features,clusters_features])
            features = features.loc[self.features_names]
            features =pd.DataFrame([features.values],columns=features.index)
            return features
        
        # ----------------------------
        
        # Method's main
        clusters_features = create_clusters_features(self, skills_entry)
        skills_features   = create_skills_features(self, skills_entry, 
                                                   exclude_features=clusters_features.index)
        
        features_array = combine_features(skills_features,clusters_features)
        return features_array


    def predict_job_probabilities(self, skills_entry):
        """ Predict probabilities of different jobs according to given skills"""
        # Create features array
        features_array = self.create_features_array(skills_entry)

        # Predict probabilities
        predictions = self.model.predict_proba([features_array])
        predictions = [prob[0][1] for prob in predictions] #Keep positive predictions
        predictions = pd.Series(predictions, index=self.targets_names)


    # ========================================================
    # **************    Simulation Functions    **************
    # ========================================================
        
    