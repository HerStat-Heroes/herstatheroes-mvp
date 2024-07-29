from flask import Flask, jsonify, send_from_directory, request, render_template
import json
import os
import glob
import numpy as np
from video import create_animation_for_data_id
from data_model import BaseballAnalysis  # Adjust the import based on your actual module location

app = Flask(__name__)

LOCAL_DATA_DIR = 'dataset/'

# Initialize your BaseballAnalysis with the path to your dataset
analysis = BaseballAnalysis(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))

def get_json_data_by_id(data_id):
    file_path = os.path.join(LOCAL_DATA_DIR, f'{data_id}.jsonl')
    if not os.path.exists(file_path):
        return None, 'File not found'

    try:
        with open(file_path, 'r') as file:
            json_data = file.read()
    except Exception as e:
        return None, f'Failed to read file: {e}'

    if not json_data.strip():
        return None, 'The data is empty or not in a valid JSON format'
    
    data = []
    try:
        for line in json_data.splitlines():
            if line.strip():
                data.append(json.loads(line))
    except json.JSONDecodeError as e:
        return None, f'Failed to parse JSON data: {e}'
    
    return data, None

def extract_specific_info(data):
    extracted_data = []

    for record in data:
        # Extracting pitch details
        pitch_id = record.get('summary_acts', {}).get('pitch', {}).get('eventId')
        pitch_type = record.get('summary_acts', {}).get('pitch', {}).get('type')
        result = record.get('summary_acts', {}).get('pitch', {}).get('result')
        action = record.get('summary_acts', {}).get('pitch', {}).get('action')
        action = action if action else None

        # Extracting hit details
        hit_event_id = record.get('summary_acts', {}).get('hit', {}).get('eventId', None)
        hitter_id = None
        for event in record.get('events', []):
            if event.get('type') == 'Hit':
                hitter_id = str(event.get('personId', {}).get('mlbId', None))
                break

        extracted_data.append({
            'pitch_id': pitch_id,
            'pitch_type': pitch_type,
            'result': result,
            'action': action,
            'hit_event_id': hit_event_id if hit_event_id else None,
            'hitter_id': hitter_id
        })

    return extracted_data

def get_squared_up_rate_category(value):
    if value < 0.5:
        return "below 0.5"
    elif 0.5 <= value < 0.6:
        return "0.5 to 0.6"
    elif 0.6 <= value < 0.7:
        return "0.6 to 0.7"
    elif 0.7 <= value < 0.8:
        return "0.7 to 0.8"
    else:
        return "0.8 to 1"
    

def build_hit_event_to_file_mapping():
    mapping = {}
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    
    for file in files:
        data_id = os.path.splitext(os.path.basename(file))[0]
        data, error = get_json_data_by_id(data_id)
        if error:
            continue
        
        for record in data:
            hit_event = record.get('summary_acts', {}).get('hit', {}).get('eventId')
            if hit_event:
                mapping[hit_event] = data_id
    
    return mapping

def get_hitter_data(hitter_id):
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    hitter_data = []

    for file in files:
        try:
            X, Other_info, category_description = analysis.out_prediction(file)
            
            if Other_info is not None and not Other_info.empty and 'PersonID' in Other_info.columns:
                if Other_info['PersonID'].iloc[0] == int(hitter_id):
                    jsonl_filename = os.path.basename(file)
                    
                    hitter_data.append({
                        'hitter_id': hitter_id,
                        'name': str(Other_info['PersonID'].iloc[0]),
                        'team': str(Other_info['TeamID'].iloc[0]) if 'TeamID' in Other_info.columns else 'Unknown',
                        'hit_event_id': Other_info['HitEventID'].iloc[0] if 'HitEventID' in Other_info.columns else 'Unknown',
                        'jsonl_filename': jsonl_filename,
                        'total_length': float(X['Total_length'].iloc[0]) if 'Total_length' in X.columns else 0,
                        'swing_speed': float(X['Swing_speed(mph)'].iloc[0]) if 'Swing_speed(mph)' in X.columns else 0,
                        'pitch_speed': float(X['Pitch_speed(mph)'].iloc[0]) if 'Pitch_speed(mph)' in X.columns else 0,
                        'pitch_spin': float(X['Pitch_spin(rpm)'].iloc[0]) if 'Pitch_spin(rpm)' in X.columns else 0,
                        'contact_point': float(X['contact_point'].iloc[0]) if 'contact_point' in X.columns else 0,
                        'stress_level': float(X['stress_level'].iloc[0]) if 'stress_level' in X.columns else 0,
                        'squared_up_rate_category': category_description
                    })
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    return hitter_data

def get_team_data(team_id):
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    team_data = []

    for file in files:
        try:
            X, Other_info, category_description = analysis.out_prediction(file)
            
            if Other_info is not None and not Other_info.empty and 'TeamID' in Other_info.columns:
                if Other_info['TeamID'].iloc[0] == int(team_id):
                    jsonl_filename = os.path.basename(file)
                    
                    team_data.append({
                        'team_id': team_id,
                        'team_name': str(Other_info['TeamID'].iloc[0]),
                        'hit_event_id': Other_info['HitEventID'].iloc[0] if 'HitEventID' in Other_info.columns else 'Unknown',
                        'jsonl_filename': jsonl_filename,
                        'total_length': float(X['Total_length'].iloc[0]) if 'Total_length' in X.columns else 0,
                        'swing_speed': float(X['Swing_speed(mph)'].iloc[0]) if 'Swing_speed(mph)' in X.columns else 0,
                        'pitch_speed': float(X['Pitch_speed(mph)'].iloc[0]) if 'Pitch_speed(mph)' in X.columns else 0,
                        'pitch_spin': float(X['Pitch_spin(rpm)'].iloc[0]) if 'Pitch_spin(rpm)' in X.columns else 0,
                        'contact_point': float(X['contact_point'].iloc[0]) if 'contact_point' in X.columns else 0,
                        'stress_level': float(X['stress_level'].iloc[0]) if 'stress_level' in X.columns else 0,
                        'squared_up_rate_category': category_description
                    })
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    return team_data

@app.route('/data/<data_id>', methods=['GET'])
def get_data(data_id):
    data, error = get_json_data_by_id(data_id)
    if error:
        return jsonify({'error': error}), 404
    return jsonify(data)

@app.route('/pitch_events', methods=['GET'])
def get_pitch_events():
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    pitch_events = []

    for file in files:
        data_id = os.path.splitext(os.path.basename(file))[0]
        data, error = get_json_data_by_id(data_id)
        if error:
            continue

        for record in data:
            pitch_event = record.get('summary_acts', {}).get('pitch', {}).get('eventId')
            if pitch_event:
                pitch_events.append({'eventId': pitch_event, 'fileName': data_id})

    return jsonify(pitch_events)

@app.route('/hit_events', methods=['GET'])
def get_hit_events():
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    hit_events = []

    for file in files:
        data_id = os.path.splitext(os.path.basename(file))[0]
        data, error = get_json_data_by_id(data_id)
        if error:
            continue

        for record in data:
            hit_event = record.get('summary_acts', {}).get('hit', {}).get('eventId')
            if hit_event:
                hit_events.append({'eventId': hit_event, 'fileName': data_id})

    return jsonify(hit_events)

@app.route('/teams', methods=['GET'])
def get_teams():
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    team_ids = set()

    for file in files:
        data_id = os.path.splitext(os.path.basename(file))[0]
        data, error = get_json_data_by_id(data_id)
        if error:
            continue

        for record in data:
            for event in record.get('events', []):
                team_id = event.get('teamId', {}).get('mlbId')
                if team_id:
                    team_ids.add(team_id)

    return jsonify(list(team_ids))

@app.route('/hitters', methods=['GET'])
def get_hitters():
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    hitter_ids = set()

    for file in files:
        data_id = os.path.splitext(os.path.basename(file))[0]
        data, error = get_json_data_by_id(data_id)
        if error:
            continue

        for record in data:
            for event in record.get('events', []):
                hitter_id = event.get('personId', {}).get('mlbId')
                if hitter_id:
                    hitter_ids.add(hitter_id)

    return jsonify(list(hitter_ids))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualization/<data_id>')
def visualization(data_id):
    return render_template('visualization.html', data_id=data_id)

@app.route('/team_profile/<team_id>')
def team_profile(team_id):
    team_data = get_team_data(team_id)
    
    if not team_data:
        return render_template('error.html', error="No data found for this team.")
    
    # Calculate some reference values
    good_data = [d for d in team_data if d['squared_up_rate_category'] == '0.8 to 1']
    avg_trajectory_good = np.mean([d['total_length'] for d in good_data]) if good_data else 0
    avg_swing_speed_good = np.mean([d['swing_speed'] for d in good_data]) if good_data else 0
    avg_contact_point_good = np.mean([d['contact_point'] for d in good_data]) if good_data else 0
    
    return render_template('team_profile.html', 
                           team_data=team_data, 
                           avg_trajectory_good=avg_trajectory_good,
                           avg_swing_speed_good=avg_swing_speed_good,
                           avg_contact_point_good=avg_contact_point_good)

@app.route('/hitter_profile/<hitter_id>')
def hitter_profile(hitter_id):
    hitter_data = get_hitter_data(hitter_id)
    
    if not hitter_data:
        return render_template('error.html', error="No data found for this hitter.")
    
    # Calculate some reference values
    good_data = [d for d in hitter_data if d['squared_up_rate_category'] == '0.8 to 1']
    avg_trajectory_good = np.mean([d['total_length'] for d in good_data]) if good_data else 0
    avg_swing_speed_good = np.mean([d['swing_speed'] for d in good_data]) if good_data else 0
    avg_contact_point_good = np.mean([d['contact_point'] for d in good_data]) if good_data else 0
    
    return render_template('hitter_profile.html', 
                           hitter_data=hitter_data, 
                           avg_trajectory_good=avg_trajectory_good,
                           avg_swing_speed_good=avg_swing_speed_good,
                           avg_contact_point_good=avg_contact_point_good)

@app.route('/video/<data_id>', methods=['POST'])
def video(data_id):
    error = create_animation_for_data_id(data_id, LOCAL_DATA_DIR)
    if error:
        return jsonify({'error': error}), 404
    video_urls = [f'/video/{data_id}/view{i}' for i in range(3)]
    return jsonify({'message': 'Video created successfully', 'videos': video_urls}), 201

@app.route('/video/<data_id>/<view_id>', methods=['GET'])
def get_video(data_id, view_id):
    filename = f'{data_id}_view{view_id}.mp4'
    if not os.path.exists(os.path.join('videos', filename)):
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(directory=os.path.abspath('videos'), path=filename)

@app.route('/delete_videos/<data_id>', methods=['DELETE'])
def delete_videos(data_id):
    video_files = glob.glob(os.path.join('videos', f'{data_id}_view*.mp4'))
    for file in video_files:
        try:
            os.remove(file)
        except Exception as e:
            return jsonify({'error': f'Failed to delete video: {e}'}), 500
    return jsonify({'message': 'Videos deleted successfully'}), 200

@app.route('/extract_info/<data_id>', methods=['GET'])
def extract_info(data_id):
    data, error = get_json_data_by_id(data_id)
    if error:
        return jsonify({'error': error}), 404

    extracted_data = extract_specific_info(data)
    return jsonify(extracted_data)

@app.route('/data_by_team/<team_id>', methods=['GET'])
def get_data_by_team(team_id):
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    matching_data_ids = []

    for file in files:
        data_id = os.path.splitext(os.path.basename(file))[0]
        data, error = get_json_data_by_id(data_id)
        if error:
            continue

        for record in data:
            for event in record.get('events', []):
                if event.get('teamId', {}).get('mlbId') == int(team_id):
                    matching_data_ids.append(data_id)
                    break

    return jsonify(matching_data_ids)

@app.route('/data_by_hitter/<hitter_id>', methods=['GET'])
def get_data_by_hitter(hitter_id):
    files = glob.glob(os.path.join(LOCAL_DATA_DIR, '*.jsonl'))
    matching_data_ids = []

    for file in files:
        data_id = os.path.splitext(os.path.basename(file))[0]
        data, error = get_json_data_by_id(data_id)
        if error:
            continue

        for record in data:
            for event in record.get('events', []):
                if event.get('personId', {}).get('mlbId') == int(hitter_id):
                    matching_data_ids.append(data_id)
                    break

    return jsonify(matching_data_ids)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
