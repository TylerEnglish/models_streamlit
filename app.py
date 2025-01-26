import streamlit as st
from scripts import load_model, make_prediction, train_models, ensure_model_directory
import pandas as pd
import matplotlib.pyplot as plt
import time

def main():
    st.title("Example Model Simulation")

    ensure_model_directory()

    action = st.sidebar.selectbox("Action", ("Train Models", "Make Prediction"))

    if action == "Train Models":
        if st.button("Train Models"):
            train_models()
            st.success("✅ Models trained and saved.")

    elif action == "Make Prediction":
        model_choice = st.selectbox("Choose a model", (
            "Linear Regression", 
            "Random Forest", 
            "Neural Network", 
            "PyTorch Neural Network", 
            "TensorFlow Neural Network", 
            "Complex PyTorch Model", 
            "Complex TensorFlow Model"
        ))

        example_data = {
            "Example 1": [1.2, 3.4, 5.6, 7.8, 9.0],
            "Example 2": [2.3, 4.5, 6.7, 8.9, 1.1],
            "Example 3": [3.4, 5.6, 7.8, 9.0, 2.2]
        }

        st.subheader("Example Data")
        df_examples = pd.DataFrame(example_data).T
        df_examples.columns = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        st.table(df_examples)

        selected_example = st.selectbox("Choose example data", list(example_data.keys()))

        if st.button("Predict"):
            with st.spinner("Generating prediction..."):
                features = example_data[selected_example]
                model = load_model(model_choice)
                prediction = make_prediction(model, features)
                time.sleep(1)  

            st.success(f"✅ Prediction for {selected_example}: {prediction:.2f}")

            st.subheader("Charts Example")

            fig, ax = plt.subplots()
            ax.bar(["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], features)
            ax.set_ylabel("Feature Values")
            ax.set_title("Input Feature Distribution")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
