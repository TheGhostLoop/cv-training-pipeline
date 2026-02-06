import streamlit as st
import requests
import os
import zipfile
from datetime import datetime
from pipeline.utils.data_utils import find_dataset_root,get_class_distribution,get_recommended_details
import pandas as pd 
import matplotlib.pyplot as plt
from training_script.generator import generate_training_script
from training_script.inference import generate_inference_script

st.set_page_config(page_title="CV Training Pipeline Generator")
st.title("CV Training Pipeline Generator")
st.caption("Upload a dataset and generate Colab Ready Training Pipeline")

# -------------------- 
#  INITIALIZE SESSION STATE
# --------------------
if "dataset_processed" not in st.session_state:
    st.session_state.dataset_processed = False
if "dataset_path" not in st.session_state:
    st.session_state.dataset_path = None
if "analysis_root" not in st.session_state:
    st.session_state.analysis_root = None
if "dataset_structure" not in st.session_state:
    st.session_state.dataset_structure = None

# -------------------- 
#  DATASET INPUT FIRST
# --------------------

st.header("1 - Dataset Input")

uploaded_zip = st.file_uploader("Upload Dataset Zip File",type=["zip"])

st.markdown("**OR**")

dataset_url = st.text_input("Upload dataset by url",placeholder="https://example.com/dataset.zip")

download_clicked = st.button("Download dataset")


def create_dataset_dir(base_dir="datasets"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(base_dir, f"dataset_{timestamp}")
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def get_data_details(dataset_path):
    root, structure = find_dataset_root(dataset_path)
    
    if structure == "pre_split":
        analysis_root = os.path.join(root, "train")
    else:
        analysis_root = root

    if not root:
        st.error("No Valid Image Folder Root Found!")
        return None, None
    else:
        st.success(f"Image Folder Found at:: {root}")
        st.success(f"Dataset Structure:: {structure}")
        return analysis_root, structure


def dataset_report(dataset_path):
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = None

    st.header("2 - Dataset Analysis")
    st.write("Dataset path:", dataset_path)
    st.info("ImageFolder validation & class distribution")

    distribution = get_class_distribution(dataset_path)
    df = pd.DataFrame(list(distribution.items()),columns=["Class","Image Count"])
    df = df.sort_values("Image Count",ascending=False)
    
    st.dataframe(df)
    fig,ax = plt.subplots(figsize=(8,4))

    ax.bar(df["Class"],df["Image Count"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.set_title("Image Distribution Per Class")

    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    df,recommendation_table = get_recommended_details(df)

    st.header("Dataset Details")
    st.table(df)

    st.header("Recommendations:")
    st.table(recommendation_table)

    # -------------- recommending details below ----------
    recommended_model = recommendation_table["model"][0]
    recommended_training_approach = recommendation_table["training_approach"][0]
    recommended_batch_size = (recommendation_table["batch_size"][0])
    recommended_epochs = (recommendation_table["epochs"][0])

    # ----------------Initialising session ---------------
    # Reinitialize ONLY when dataset changes
    if st.session_state.active_dataset != dataset_path:
        st.session_state.active_dataset = dataset_path
        st.session_state.final_epochs = recommended_epochs
        st.session_state.final_batch_size = recommended_batch_size
        st.session_state.final_model_name = recommended_model
        st.session_state.final_training_approach = recommended_training_approach

    # Initialize defaults if not present
    if "final_model_name" not in st.session_state:
        st.session_state.final_model_name = recommended_model
    if "final_training_approach" not in st.session_state:
        st.session_state.final_training_approach = recommended_training_approach
    if "final_batch_size" not in st.session_state:
        st.session_state.final_batch_size = recommended_batch_size
    if "final_epochs" not in st.session_state:
        st.session_state.final_epochs = recommended_epochs

    st.header("Input Overriding:")

    # Use different keys for widgets, then update session state
    final_model_name = st.selectbox(
        "Select Final Model: ",
        ("resnet18", "mobilenetv2"),
        index=0 if st.session_state.final_model_name == "resnet18" else 1,
        key="model_select"
    )

    final_training_approach = st.selectbox(
        "Select training mode:",
        ("Final Head Only", "Layer4 Unfreeze + Final Head"),
        index=0 if st.session_state.final_training_approach == "Final Head Only" else 1,
        key="training_select"
    )

    final_batch_size = st.slider(
        "Select Batch Size: ",
        16, 64, 
        value=st.session_state.final_batch_size,
        step=8,
        key="batch_slider"
    )
    
    final_epochs = st.number_input(
        "Enter Epochs Rounds:",
        1, 50,
        value=st.session_state.final_epochs,
        step=1,
        key="epochs_input"
    )

    # Update session state with widget values
    st.session_state.final_model_name = final_model_name
    st.session_state.final_training_approach = final_training_approach
    st.session_state.final_batch_size = final_batch_size
    st.session_state.final_epochs = final_epochs

    st.header("Final Summary:")
    final_model_summary = {
        "Model": [final_model_name],
        "Training mode": [final_training_approach],
        "Batch size": [final_batch_size],
        "Epochs": [final_epochs]
    }

    final_model_summary = pd.DataFrame(final_model_summary)
    st.table(final_model_summary)


    # --------- Creating and sending config to generator-------

    config = {
        "dataset_path": st.session_state.dataset_path,
        "dataset_structure": st.session_state.dataset_structure,
        "model_name": st.session_state.final_model_name,
        "training_mode": st.session_state.final_training_approach,
        "batch_size": st.session_state.final_batch_size,
        "epochs": st.session_state.final_epochs
    }

    st.header("Final Config:")
    display_config = {k: str(v) for k, v in config.items()}
    st.table(display_config)

    training_script = generate_training_script(config)
    inference_script = generate_inference_script()


    if st.button("Generate Scripts"):
        

        st.success("Scripts generated successfully!")

    st.download_button(
        label="Download Training Script",
        data=training_script,
        file_name="train.py",
        mime="text/x-python"
    )

    st.download_button(
        label="Download Inference Script",
        data=inference_script,
        file_name="inference.py",
        mime="text/x-python"
    )






# -------------------- 
#  PROCESS UPLOADED FILE
# --------------------
if uploaded_zip and not st.session_state.dataset_processed:
    dataset_dir = create_dataset_dir()
    zip_path = os.path.join(dataset_dir,"dataset.zip")

    with open(zip_path,"wb") as f:
        f.write(uploaded_zip.getvalue())

    with zipfile.ZipFile(zip_path,"r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    st.success("Dataset uploaded and extracted successfully!")
    os.remove(zip_path)
    
    analysis_root, dataset_structure = get_data_details(dataset_dir)
    
    # Store in session state
    st.session_state.dataset_path = dataset_dir
    st.session_state.analysis_root = analysis_root
    st.session_state.dataset_structure = dataset_structure
    st.session_state.dataset_processed = True
    st.rerun()




# -------------------- 
#  PROCESS URL DOWNLOAD
# --------------------    
elif download_clicked:
    if not dataset_url:
        st.error("Please provide a url.")
    else:
        with st.spinner("Downloading dataset...."):
            try: 
                dataset_dir = create_dataset_dir()
                zip_path = os.path.join(dataset_dir,"dataset.zip")

                r = requests.get(dataset_url)
                with open(zip_path,"wb") as f:
                    f.write(r.content)

                with zipfile.ZipFile(zip_path,"r") as zip_ref:
                    zip_ref.extractall(dataset_dir)
        
                st.success("Dataset Downloaded and extracted.")
                os.remove(zip_path)
                
                analysis_root, dataset_structure = get_data_details(dataset_dir)
                
                # Store in session state
                st.session_state.dataset_path = dataset_dir
                st.session_state.analysis_root = analysis_root
                st.session_state.dataset_structure = dataset_structure
                st.session_state.dataset_processed = True
                st.rerun()

                

            except Exception as e:
                st.error(f"Failed to download dataset! {e}")

# -------------------- 
#  SHOW ANALYSIS IF DATASET IS LOADED
# --------------------
if st.session_state.dataset_processed and st.session_state.analysis_root:
    dataset_report(st.session_state.analysis_root)