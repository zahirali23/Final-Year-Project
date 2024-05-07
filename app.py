import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model and scaler from disk
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Position mapping from encoded value back to label
position_mapping = {
    0: 'DF',
    1: 'DFFW',
    2: 'DFMF',
    3: 'FW',
    4: 'FWDF',
    5: 'FWMF',
    6: 'GK',
    7: 'MF',
    8: 'MFDF',
    9: 'MFFW'
}


def user_input_features():
    st.sidebar.header('Player Input Features')
    age = st.sidebar.number_input('Age', min_value=16, max_value=40, value=25, step=1)
    minutes_played = st.sidebar.number_input('Minutes Played', min_value=0, max_value=5000, value=1800, step=90)
    shots = st.sidebar.number_input('Shots', min_value=0.0, max_value=200.0, value=17.0, step=0.1, format="%.1f")
    aerial_battles_won = st.sidebar.number_input('Aerial Battles Won %', min_value=0, max_value=1000, value=55, step=1)
    goal_creation = st.sidebar.number_input('Goal Creation', min_value=0, max_value=200, value=12, step=1)
    tackles_made = st.sidebar.number_input('Tackles Made', min_value=0, max_value=500, value=73, step=1)
    tackles_won = st.sidebar.number_input('Tackles Won', min_value=0, max_value=500, value=58, step=1)
    times_team_pressed_opposition = st.sidebar.number_input('Times Team Pressed Opposition', min_value=0, max_value=2000, value=842, step=1)
    blocks = st.sidebar.number_input('Blocks ', min_value=0, max_value=500, value=69, step=1)
    interceptions = st.sidebar.number_input('Interceptions', min_value=0, max_value=500, value=64, step=1)
    clearances = st.sidebar.number_input('Clearances', min_value=0, max_value=500, value=87, step=1)
    fouls_commited = st.sidebar.number_input('Fouls Commited', min_value=0, max_value=500, value=33, step=1)
    fouls_drawn = st.sidebar.number_input('Fouls Drawn', min_value=0, max_value=500, value=29, step=1)
    loose_balls_recovered = st.sidebar.number_input('Loose Balls Recovered', min_value=0, max_value=2000, value=300, step=1)
    touches = st.sidebar.number_input('Touches', min_value=0, max_value=5000, value=2500, step=100)
    successful_dribbles = st.sidebar.number_input('Successful Dribbles', min_value=0, max_value=500, value=80, step=1)
    attempted_dribbles = st.sidebar.number_input('Attempted Dribbles', min_value=0, max_value=1000, value=120, step=1)
    distance_ran = st.sidebar.number_input('Distance Ran', min_value=0, max_value=50000, value=10400, step=100)
    distance_covered_with_ball = st.sidebar.number_input('Distance Covered With Ball', min_value=0, max_value=30000, value=2100, step=100)
    times_dispossessed = st.sidebar.number_input('Times Dispossessed', min_value=0, max_value=500, value=40, step=1)
    passes_received = st.sidebar.number_input('Passes Received', min_value=0, max_value=5000, value=800, step=10)
    # Dropdown for positions using reversed mapping
    position = st.sidebar.selectbox('Position', options=list(position_mapping.values()), index=7)  # Default to 'MF'
    position_encoded = {v: k for k, v in position_mapping.items()}[position]  # Encode position back
    
    data = {
        'Age': age,
        'Minutes Played': minutes_played,
        'Shots': shots,
        'Aerial Battles Won %': aerial_battles_won,
        'Goal Creation': goal_creation,
        'Tackles Made': tackles_made,
        'Tackles Won': tackles_won,
        'Times Team Pressed Opposition': times_team_pressed_opposition,
        'Blocks ': blocks,
        'Interceptions': interceptions,
        'Clearances': clearances,
        'Fouls Commited': fouls_commited,
        'Fouls Drawn': fouls_drawn,
        'Loose Balls Recovered': loose_balls_recovered,
        'Touches': touches,
        'Successful Dribbles': successful_dribbles,
        'Attempted Dribbles': attempted_dribbles,
        'Distance Ran': distance_ran,
        'Distance Covered With Ball': distance_covered_with_ball,
        'Times Dispossessed': times_dispossessed,
        'Passes Received': passes_received,
        'Position_Encoded': position_encoded
    }
    features = pd.DataFrame(data, index=[0])
    return features

def calculate_per_90_features(df):
    df['Full 90s Played'] = df['Minutes Played'] / 90
    per_90_columns = [
        'Shots', 'Goal Creation', 'Tackles Made', 'Tackles Won', 
        'Times Team Pressed Opposition', 'Blocks ', 'Interceptions', 
        'Clearances', 'Fouls Commited', 'Fouls Drawn', 'Loose Balls Recovered', 
        'Touches', 'Successful Dribbles', 'Attempted Dribbles', 
        'Distance Ran', 'Distance Covered With Ball', 'Times Dispossessed', 'Passes Received'
    ]
    for col in per_90_columns:
        df[col + ' per 90'] = df[col] / df['Full 90s Played']

    return df

# Use the functions to prepare data
input_df = user_input_features()
input_df = calculate_per_90_features(input_df)

expected_features = [
    'Age', 
    'Shots per 90', 
    'Aerial Battles Won %', 
    'Goal Creation per 90',
    'Full 90s Played', 
    'Tackles Made per 90', 
    'Tackles Won per 90',
    'Times Team Pressed Opposition per 90', 
    'Blocks  per 90',
    'Interceptions per 90', 
    'Clearances per 90', 
    'Fouls Commited per 90',
    'Fouls Drawn per 90', 
    'Loose Balls Recovered per 90', 
    'Touches per 90',
    'Successful Dribbles per 90', 
    'Attempted Dribbles per 90',
    'Distance Ran per 90', 
    'Distance Covered With Ball per 90',
    'Times Dispossessed per 90', 
    'Passes Received per 90',
    'Position_Encoded'
]

# Prepare model input by selecting only the features expected by the model
model_input = input_df[[col for col in expected_features if col in input_df.columns]]

# Scale the features
model_input_scaled = scaler.transform(model_input)

# Make predictions
prediction = model.predict(model_input_scaled)
prediction_prob = model.predict_proba(model_input_scaled)

# Display results
st.subheader('Prediction Probability')
st.write(prediction_prob)

injury_labels = {0: 'No injury', 1: 'Serious Injuries', 2: 'Moderate Injuries', 3: 'Minor Injuries'}
predicted_label = injury_labels[prediction[0]]

st.subheader('Predicted Injury Type')
st.write(predicted_label)