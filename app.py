import streamlit as st

import jsonlines

import pandas as pd

# Thresholds as per given data

thresholds = {

    "Blood": 80,

    "Yolk": 120,

    "Feces": 130,

    "Bruised": 20,

    "OpenCrack": 20,

    "Crack": 175,

    "Wrinkle": 6500,

    "Feather": 100,

    "CalciumSpot": 100,

    "Dust": 150,

    "Group_Dirty": 20,

    "Group_Damaged": 20,

    "Group_ShellDeviation": 1000

}

# Streamlit UI Setup

st.title("Egg Quality Analysis")

st.subheader("Upload JSONL File to Analyze Egg Data")

# Upload JSONL file

uploaded_file = st.file_uploader("Upload JSONL File", type="jsonl")

# Analysis

if uploaded_file:

    # Load JSONL file

    data = []

    with jsonlines.Reader(uploaded_file) as reader:

        for obj in reader:

            data.append(obj)

    # DataFrame for analysis

    df = pd.DataFrame(data)

    # Convert list of dictionaries to individual columns

    egg_data = []

    for egg in df[0]:

        egg_dict = {item['Label']: item['Value'] for item in egg}

        egg_data.append(egg_dict)

    egg_df = pd.DataFrame(egg_data)

    # Display the uploaded data

    st.subheader("Uploaded Data")

    st.dataframe(egg_df)

    # Analysis Results

    st.subheader("Threshold Analysis")

    total_eggs = len(egg_df)

    rejected_eggs = {}

    close_to_rejection = { "1%": {}, "2%-10%": {} }

    for label, threshold in thresholds.items():

        # Eggs crossing the threshold

        rejected = egg_df[egg_df[label] > threshold]

        rejected_eggs[label] = len(rejected)

        # Eggs close to threshold (1% below)

        just_rejected = egg_df[

            (egg_df[label] <= threshold) & 

            (egg_df[label] > threshold * 0.99)

        ]

        close_to_rejection["1%"][label] = len(just_rejected)

        # Eggs close to threshold (2% to 10% below)

        close_range = egg_df[

            (egg_df[label] <= threshold * 0.99) & 

            (egg_df[label] > threshold * 0.90)

        ]

        close_to_rejection["2%-10%"][label] = len(close_range)

    # Display results

    st.subheader("Eggs Crossing Thresholds")

    st.write(f"Total Eggs Analyzed: {total_eggs}")

    st.table(pd.DataFrame.from_dict(rejected_eggs, orient='index', columns=['Rejected Eggs']))

    st.subheader("Eggs Just Rejected (Within 1% of Threshold)")

    st.table(pd.DataFrame.from_dict(close_to_rejection["1%"], orient='index', columns=['Just Rejected (1%)']))

    st.subheader("Eggs Close to Rejection (2% - 10% Below Threshold)")

    st.table(pd.DataFrame.from_dict(close_to_rejection["2%-10%"], orient='index', columns=['Close to Rejection (2%-10%)'])) 