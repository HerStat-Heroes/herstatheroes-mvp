import glob
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaseballAnalysis:
    def __init__(self, dataset_path):
        """
        Initialize the BaseballAnalysis class with the dataset path.
        Initialize the necessary attributes and start processing files and building the model.
        """
        self.dataset_path = dataset_path
        self.full_df = None
        self.hit_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.process_files()
        self.build_model()

    def process_files(self):
        """
        Process all JSON files in the dataset path.
        Concatenate them into a single DataFrame and perform initial filtering and calculations.
        """
        all_files = glob.glob(self.dataset_path)
        df_list = []

        for file in all_files:
            df = pd.read_json(file, lines=True)
            df_list.append(df)

        self.full_df = pd.concat(df_list, ignore_index=True)
        self.filter_hits()
        self.calculate_trajectory_length(self.hit_df)
        self.extract_hit_info(self.hit_df)
        self.calculate_contact_point(self.hit_df)
        self.calculate_stress_level(self.hit_df)
        self.calculate_squared_up_rate()
        self.prepare_final_dataframe()

    def filter_hits(self):
        """
        Filter the DataFrame to include only rows that contain hit events.
        """
        def check_hit_event(summary_acts):
            if 'hit' in summary_acts and summary_acts['hit'].get('eventId'):
                return True
            return False

        self.hit_df = self.full_df[self.full_df['summary_acts'].apply(check_hit_event)].reset_index(drop=True)

    def calculate_trajectory_length(self, hit_df):
        """
        Calculate the trajectory length for each hit in the DataFrame.
        The length is computed as the sum of Euclidean distances between consecutive head positions.
        """
        def euclidean_distance(pos1, pos2):
            return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))

        def compute_trajectory_length(records):
            length = 0.0
            for i in range(len(records) - 1):
                length += euclidean_distance(records[i], records[i + 1])
            return length

        hit_bat_df = hit_df['samples_bat']
        results = []

        for hit in hit_bat_df:
            positions = [record['head']['pos'] for record in hit if 'head' in record]
            total_length = compute_trajectory_length(positions)
            results.append(total_length)

        self.length_df = pd.DataFrame(results, columns=['Total_length'])
        return self.length_df

    def extract_hit_info(self, hit_df):
        """
        Extract relevant features for each hit event, such as pitch speed, pitch spin, hit speed, etc.
        """
        def extract_features(row):
            pitch_info = row['summary_acts'].get('pitch', {})
            hit_info = row['summary_acts'].get('hit', {})
            event_info = row['events']
            swing_info = row['samples_bat']

            pitch_event = pitch_info['eventId']
            pitch_result = pitch_info['result']
            pitch_action = pitch_info['action']
            pitch_speed_mph = pitch_info.get('speed', {}).get('mph', 0)
            pitch_spin_rpm = pitch_info.get('spin', {}).get('rpm', 0)
            hit_speed_mph = hit_info.get('speed', {}).get('mph', 0)
            teamid = event_info[0]['teamId']['mlbId']
            personid = event_info[0]['personId']['mlbId']
            eventid = event_info[0]['eventId']

            swing_speed_mph = 0
            for i in range(len(swing_info) - 1):
                if swing_info[i].get('event') == 'Hit':
                    head_pos_hit = swing_info[i]['head']['pos']
                    time_hit = swing_info[i]['time']

                    head_pos_next = swing_info[i - 1]['head']['pos']
                    time_next = swing_info[i - 1]['time']

                    distance = np.sqrt(np.sum((np.array(head_pos_hit) - np.array(head_pos_next)) ** 2))
                    time_diff = time_hit - time_next

                    swing_speed_fps = distance / time_diff if time_diff != 0 else 0
                    swing_speed_mph = swing_speed_fps / 1.46667
                    break

            return pd.Series({
                'Pitch_speed(mph)': pitch_speed_mph,
                'Pitch_spin(rpm)': pitch_spin_rpm,
                'Hit_speed(mph)': hit_speed_mph,
                'Swing_speed(mph)': swing_speed_mph,
                'TeamID': teamid,
                'PersonID': personid,
                'EventID': eventid,
                'Pitch_event': pitch_event,
                'Pitch_result': pitch_result,
                'Pitch_action': pitch_action
            })

        self.hit_info_df = hit_df.apply(extract_features, axis=1)
        return self.hit_info_df

    def calculate_contact_point(self, hit_df):
        """
        Calculate the contact point on the bat for each hit.
        Normalize the distance between the ball and the bat head by the length of the bat.
        """
        def euclidean_distance(pos1, pos2):
            return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))

        def find_closest_time(ball_samples, hit_time):
            times = [sample['time'] for sample in ball_samples]
            closest_time_idx = (np.abs(np.array(times) - hit_time)).argmin()
            return ball_samples[closest_time_idx]['pos']

        hit_ballbat_df = hit_df[['samples_ball', 'samples_bat']]
        results = []

        for index, row in hit_ballbat_df.iterrows():
            bat_samples = row['samples_bat']
            ball_samples = row['samples_ball']
            hit_event = next((event for event in bat_samples if event.get('event') == 'Hit'), None)
            
            if hit_event:
                hit_time = hit_event['time']
                head_pos = hit_event['head']['pos']
                handle_pos = hit_event['handle']['pos']
                closest_ball_pos = find_closest_time(ball_samples, hit_time)
                distance_ball_to_head = euclidean_distance(closest_ball_pos, head_pos)
                bat_length = euclidean_distance(head_pos, handle_pos)

                # Handle zero bat length to avoid division by zero
                if bat_length == 0:
                    normalized_distance = np.nan
                else:
                    normalized_distance = distance_ball_to_head / bat_length

                results.append({'index': index, 'contact_point': normalized_distance})

        results_df = pd.DataFrame(results).set_index('index')
        
        # Ensure indices are compatible for joining
        results_df.index = results_df.index.astype(hit_ballbat_df.index.dtype)
        self.hit_ballbat_df = hit_ballbat_df.join(results_df, on=hit_ballbat_df.index)
        
        return self.hit_ballbat_df['contact_point']


    def calculate_stress_level(self, hit_df):
        """
        Calculate the stress level for each hit based on the game situation (outs, strikes, balls).
        """
        hit_score_df = hit_df['summary_score']
        level_stress_df = pd.DataFrame(index=hit_score_df.index, columns=["stress_level"])

        def calculate_stress_level(hit):
            stress_level = hit['outs']['inning'] + hit['count']['strikes']['plateAppearance'] - hit['count']['balls']['plateAppearance']
            return stress_level

        level_stress_df['stress_level'] = hit_score_df.apply(calculate_stress_level)
        self.level_stress_df = level_stress_df
        return self.level_stress_df

    def calculate_squared_up_rate(self):
        """
        Calculate the squared-up rate for each hit.
        The squared-up rate is the ratio of hit speed to optimal exit velocity.
        """
        MF = 0.2
        self.hit_info_df['Optimal_exit_velocity(mph)'] = MF * self.hit_info_df['Pitch_speed(mph)'] + (1 + MF) * self.hit_info_df['Swing_speed(mph)']
        self.hit_info_df['Squared_up_Rate'] = self.hit_info_df['Hit_speed(mph)'] / self.hit_info_df['Optimal_exit_velocity(mph)']

    def prepare_final_dataframe(self):
        """
        Prepare the final DataFrame for model training and prediction.
        Combine all calculated features into a single DataFrame and drop rows with missing values.
        """
        self.df_predict = pd.concat([self.length_df, self.hit_info_df[
            ['Pitch_speed(mph)', 'Pitch_spin(rpm)', 'Hit_speed(mph)', 'Swing_speed(mph)', 'Optimal_exit_velocity(mph)', 'Squared_up_Rate']],
            self.hit_ballbat_df['contact_point'], self.level_stress_df['stress_level']], axis=1)
        self.df_predict = self.df_predict.dropna()

    def clean_data(self):
        """
        Clean the DataFrame by replacing infinite values with NaN and dropping rows with NaN values.
        """
        self.df_predict.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df_predict.dropna(inplace=True)

    def build_model(self):
        """
        Build and train the Random Forest model to classify squared-up rates.
        """
        self.clean_data()

        X = self.df_predict.drop(columns=['Squared_up_Rate'])
        y = self.df_predict['Squared_up_Rate']

        X_scaled = self.scaler.fit_transform(X)

        bins = [0, 0.5, 0.6, 0.7, 0.8, np.inf]
        labels = [0, 1, 2, 3, 4]
        y_class = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train_class, y_train_class)

    def out_prediction(self, jsonl_file):
        """
        Predict the squared-up rate category for a new JSON file.
        """
        predict_df = pd.read_json(jsonl_file, lines=True)

        length = self.calculate_trajectory_length(predict_df)
        hit_info = self.extract_hit_info(predict_df)
        contact_point = self.calculate_contact_point(predict_df)
        stress_level = self.calculate_stress_level(predict_df)
        MF = 0.2
        hit_info['Optimal_exit_velocity(mph)'] = MF * hit_info['Pitch_speed(mph)'] + (1 + MF) * hit_info['Swing_speed(mph)']

        X = pd.DataFrame([{
            'Total_length': length.values,
            'Pitch_speed(mph)': hit_info['Pitch_speed(mph)'].values,
            'Pitch_spin(rpm)': hit_info['Pitch_spin(rpm)'].values,
            'Hit_speed(mph)': hit_info['Hit_speed(mph)'].values,
            'Swing_speed(mph)': hit_info['Swing_speed(mph)'].values,
            'Optimal_exit_velocity(mph)': hit_info['Optimal_exit_velocity(mph)'].values,
            'contact_point': contact_point.values,
            'stress_level': stress_level.values
        }])

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        category_mapping = {
            0: "below 0.5",
            1: "0.5 to 0.6",
            2: "0.6 to 0.7",
            3: "0.7 to 0.8",
            4: "0.8 to 1"
        }

        Other_info = pd.DataFrame([{
            'TeamID': hit_info['TeamID'].values,
            'PersonID': hit_info['PersonID'].values,
            'HitEventID': hit_info['EventID'].values,
            'PitchEventID': hit_info['Pitch_event'].values,
            'Pitch_result': hit_info['Pitch_result'].values,
            'Pitch_action': hit_info['Pitch_action'].values
        }])

        return X, Other_info, category_mapping[y_pred[0]]


# analysis = BaseballAnalysis('dataset/*.jsonl')
# X, Other_info, category_description = analysis.out_prediction('dataset/12345641_36899.jsonl')

# print("Feature DataFrame (X) details:")
# for column in X.columns:
#     values = X[column].iloc[0]
#     print(f"{column}: {values}")

# print("\nAdditional Information (Other_info) details:")
# for column in Other_info.columns:
#     values = Other_info[column].iloc[0]
#     print(f"{column}: {values}")

# print("\nPredicted Category Description:")
# print(category_description)
