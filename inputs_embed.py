############### Built-in Modules ###############
import argparse
import csv  # Still used for writing initially, but can be fully replaced by df.to_csv
import itertools
import os
import time
import traceback  # For more detailed error printing
import warnings
from collections import OrderedDict
from decimal import Decimal
from io import StringIO
from typing import Optional
import argparse
from pathlib import Path


############### Third-party Libraries ###############
import matplotlib.pyplot as plt  # For plotting
import numpy as np
import pandas as pd  # For groupby averaging and data handling
from openpyxl import load_workbook
from scipy.interpolate import UnivariateSpline  # For spline fitting
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # For a progress bar
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator

############### PyTorch ###############
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"You are using '{device}' device")

model_name = 'E:/research/models'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# List of all 118 chemical elements (Symbol, Name)
ELEMENTS_DATA = [
    {"symbol": "H", "name": "Hydrogen"}, {"symbol": "He", "name": "Helium"},
    {"symbol": "Li", "name": "Lithium"}, {"symbol": "Be", "name": "Beryllium"},
    {"symbol": "B", "name": "Boron"}, {"symbol": "C", "name": "Carbon"},
    {"symbol": "N", "name": "Nitrogen"}, {"symbol": "O", "name": "Oxygen"},
    {"symbol": "F", "name": "Fluorine"}, {"symbol": "Ne", "name": "Neon"},
    {"symbol": "Na", "name": "Sodium"}, {"symbol": "Mg", "name": "Magnesium"},
    {"symbol": "Al", "name": "Aluminum"}, {"symbol": "Si", "name": "Silicon"},
    {"symbol": "P", "name": "Phosphorus"}, {"symbol": "S", "name": "Sulfur"},
    {"symbol": "Cl", "name": "Chlorine"}, {"symbol": "Ar", "name": "Argon"},
    {"symbol": "K", "name": "Potassium"}, {"symbol": "Ca", "name": "Calcium"},
    {"symbol": "Sc", "name": "Scandium"}, {"symbol": "Ti", "name": "Titanium"},
    {"symbol": "V", "name": "Vanadium"}, {"symbol": "Cr", "name": "Chromium"},
    {"symbol": "Mn", "name": "Manganese"}, {"symbol": "Fe", "name": "Iron"},
    {"symbol": "Co", "name": "Cobalt"}, {"symbol": "Ni", "name": "Nickel"},
    {"symbol": "Cu", "name": "Copper"}, {"symbol": "Zn", "name": "Zinc"},
    {"symbol": "Ga", "name": "Gallium"}, {"symbol": "Ge", "name": "Germanium"},
    {"symbol": "As", "name": "Arsenic"}, {"symbol": "Se", "name": "Selenium"},
    {"symbol": "Br", "name": "Bromine"}, {"symbol": "Kr", "name": "Krypton"},
    {"symbol": "Rb", "name": "Rubidium"}, {"symbol": "Sr", "name": "Strontium"},
    {"symbol": "Y", "name": "Yttrium"}, {"symbol": "Zr", "name": "Zirconium"},
    {"symbol": "Nb", "name": "Niobium"}, {"symbol": "Mo", "name": "Molybdenum"},
    {"symbol": "Tc", "name": "Technetium"}, {"symbol": "Ru", "name": "Ruthenium"},
    {"symbol": "Rh", "name": "Rhodium"}, {"symbol": "Pd", "name": "Palladium"},
    {"symbol": "Ag", "name": "Silver"}, {"symbol": "Cd", "name": "Cadmium"},
    {"symbol": "In", "name": "Indium"}, {"symbol": "Sn", "name": "Tin"},
    {"symbol": "Sb", "name": "Antimony"}, {"symbol": "Te", "name": "Tellurium"},
    {"symbol": "I", "name": "Iodine"}, {"symbol": "Xe", "name": "Xenon"},
    {"symbol": "Cs", "name": "Caesium"}, {"symbol": "Ba", "name": "Barium"},
    {"symbol": "La", "name": "Lanthanum"}, {"symbol": "Ce", "name": "Cerium"},
    {"symbol": "Pr", "name": "Praseodymium"}, {"symbol": "Nd", "name": "Neodymium"},
    {"symbol": "Pm", "name": "Promethium"}, {"symbol": "Sm", "name": "Samarium"},
    {"symbol": "Eu", "name": "Europium"}, {"symbol": "Gd", "name": "Gadolinium"},
    {"symbol": "Tb", "name": "Terbium"}, {"symbol": "Dy", "name": "Dysprosium"},
    {"symbol": "Ho", "name": "Holmium"}, {"symbol": "Er", "name": "Erbium"},
    {"symbol": "Tm", "name": "Thulium"}, {"symbol": "Yb", "name": "Ytterbium"},
    {"symbol": "Lu", "name": "Lutetium"}, {"symbol": "Hf", "name": "Hafnium"},
    {"symbol": "Ta", "name": "Tantalum"}, {"symbol": "W", "name": "Tungsten"},
    {"symbol": "Re", "name": "Rhenium"}, {"symbol": "Os", "name": "Osmium"},
    {"symbol": "Ir", "name": "Iridium"}, {"symbol": "Pt", "name": "Platinum"},
    {"symbol": "Au", "name": "Gold"}, {"symbol": "Hg", "name": "Mercury"},
    {"symbol": "Tl", "name": "Thallium"}, {"symbol": "Pb", "name": "Lead"},
    {"symbol": "Bi", "name": "Bismuth"}, {"symbol": "Po", "name": "Polonium"},
    {"symbol": "At", "name": "Astatine"}, {"symbol": "Rn", "name": "Radon"},
    {"symbol": "Fr", "name": "Francium"}, {"symbol": "Ra", "name": "Radium"},
    {"symbol": "Ac", "name": "Actinium"}, {"symbol": "Th", "name": "Thorium"},
    {"symbol": "Pa", "name": "Protactinium"}, {"symbol": "U", "name": "Uranium"},
    {"symbol": "Np", "name": "Neptunium"}, {"symbol": "Pu", "name": "Plutonium"},
    {"symbol": "Am", "name": "Americium"}, {"symbol": "Cm", "name": "Curium"},
    {"symbol": "Bk", "name": "Berkelium"}, {"symbol": "Cf", "name": "Californium"},
    {"symbol": "Es", "name": "Einsteinium"}, {"symbol": "Fm", "name": "Fermium"},
    {"symbol": "Md", "name": "Mendelevium"}, {"symbol": "No", "name": "Nobelium"},
    {"symbol": "Lr", "name": "Lawrencium"}, {"symbol": "Rf", "name": "Rutherfordium"},
    {"symbol": "Db", "name": "Dubnium"}, {"symbol": "Sg", "name": "Seaborgium"},
    {"symbol": "Bh", "name": "Bohrium"}, {"symbol": "Hs", "name": "Hassium"},
    {"symbol": "Mt", "name": "Meitnerium"}, {"symbol": "Ds", "name": "Darmstadtium"},
    {"symbol": "Rg", "name": "Roentgenium"}, {"symbol": "Cn", "name": "Copernicium"},
    {"symbol": "Nh", "name": "Nihonium"}, {"symbol": "Fl", "name": "Flerovium"},
    {"symbol": "Mc", "name": "Moscovium"}, {"symbol": "Lv", "name": "Livermorium"},
    {"symbol": "Ts", "name": "Tennessine"}, {"symbol": "Og", "name": "Oganesson"}
]

PREDEFINED_QUERY_TEMPLATES = {
    1: {"template": "{symbol}", "description": "Chemical symbol only (e.g., 'Fe')"},
    2: {"template": "{name}", "description": "Element name only (e.g., 'Iron')"},
    3: {"template": "Influence of {name} on steel microstructure, phase stability, and resulting mechanical properties like strength and toughness.","description": "Comprehensive: Links element to microstructure and resulting mechanical properties (Original #9)."},
    4: {"template": "{name} as a strengthening agent in steel: mechanisms including solid solution, precipitation, and grain boundary strengthening.","description": "Focus on strengthening: Details how an element contributes to steel strength (Original #15)."},
    5: {"template": "{name} ({symbol}) in steel: primary alloying effects, typical concentration ranges, and common steel grades.","description": "Contextual overview: Alloying effects with concentration and grade context (Original #8)."},
    6: {"template": "Role of {name} in heat treatment of steels: influence on hardenability, tempering response, and phase transformations.","description": "Heat treatment impact: Element's effect on hardenability, tempering, and phase changes (Original #17)."},
    7: {"template": "Properties of {name} in steel","description": "Direct and general: Asks broadly for the properties imparted by the element in steel (Original #6)."}
}


def generate_element_embeddings(
    save_file_name: str,
    template_choice: int = None,
    custom_query_template: str = None,
    model_path: str = 'E:/research/models',
    elements_data: list = ELEMENTS_DATA
    ):
    """
    Generates [CLS] embeddings for chemical elements using SteelBERT,
    saves them to a CSV file, and returns them as a Pandas DataFrame.

    Args:
        save_file_name (str): Path to save the CSV file (e.g., "element_embeddings.csv").
        template_choice (int, optional): An integer key to select a predefined query template.
                                        See PREDEFINED_QUERY_TEMPLATES.
        custom_query_template (str, optional): A custom string template for the query.
                                            Can use {symbol} and {name} placeholders.
                                            Ignored if template_choice is valid.
        model_path (str, optional): Path or Hugging Face model name for SteelBERT.
                                    Defaults to "MGE-LLMs/SteelBERT".
        elements_data (list, optional): A list of dictionaries, where each dictionary
                                        has "symbol" and "name" keys for an element.
                                        Defaults to the internal comprehensive list.

    Returns:
        pd.DataFrame | None: A Pandas DataFrame containing the element symbols and their
                            embeddings, or None if an error occurs.
    """
    actual_query_template = None
    if template_choice is not None and template_choice in PREDEFINED_QUERY_TEMPLATES:
        actual_query_template = PREDEFINED_QUERY_TEMPLATES[template_choice]["template"]
        print(f"Using predefined template #{template_choice}: '{PREDEFINED_QUERY_TEMPLATES[template_choice]['description']}' -> '{actual_query_template}'")
    elif custom_query_template is not None:
        actual_query_template = custom_query_template
        print(f"Using custom query template: '{custom_query_template}'")
    else:
        print("Error: You must provide a valid 'template_choice' or a 'custom_query_template'.")
        print("Available predefined template choices:")
        for key, val in PREDEFINED_QUERY_TEMPLATES.items():
            print(f"  {key}: {val['description']} (Template: '{val['template']}')")
        return None

    print(f"\nInitializing tokenizer and model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None

    print(f"\nGenerating embeddings for {len(elements_data)} elements...")
    results_for_df = []

    for element in tqdm(elements_data, desc="Processing elements"):
        element_symbol = element["symbol"]
        element_name = element["name"]

        try:
            query_text = actual_query_template.format(symbol=element_symbol, name=element_name)
        except KeyError as e:
            print(f"\nError: The query_template '{actual_query_template}' uses an invalid placeholder: {e}")
            print("Ensure your custom template uses only {symbol} and/or {name} if provided.")
            return None

        inputs = tokenizer(query_text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding_tensor = outputs.last_hidden_state[:, 0, :].squeeze()
            embedding_list = cls_embedding_tensor.cpu().numpy().tolist()

        row_data = {'Element': element_symbol}
        for i, val in enumerate(embedding_list):
            row_data[f'Dim{i+1}'] = val
        results_for_df.append(row_data)

    if not results_for_df:
        print("No results generated.")
        return pd.DataFrame() # Return empty DataFrame

    df = pd.DataFrame(results_for_df)
    df.rename(columns={'Element': 'symbol'}, inplace=True)

    try:
        df.to_csv(save_file_name, index=False)
        print(f"\nSuccessfully saved embeddings to {save_file_name}")
        print(f"DataFrame shape: {df.shape}")
    except Exception as e:
        print(f"\nError saving DataFrame to CSV: {e}")
        # Still return the DataFrame even if saving fails
        return df

    return df


def normalize_array(input_array):
    """
    Normalize a 2D numpy array or PyTorch tensor to the range (0.0001, 0.9999) using sklearn MinMaxScaler.

    Parameters:
    input_array (numpy.ndarray or torch.Tensor): 2D array or tensor to be normalized.

    Returns:
    torch.Tensor: Normalized 2D tensor.
    """
    is_tensor = isinstance(input_array, torch.Tensor)

    if is_tensor:
        input_array = input_array.cpu().numpy()

    scaler = MinMaxScaler(feature_range=(0.0001, 0.9999))  # Set range strictly within (0, 1)
    normalized_array = scaler.fit_transform(input_array.T).T

    if is_tensor:
        normalized_array = torch.tensor(normalized_array)

    return normalized_array


def gen_target(df):
    tensors = []

    # Loop through each row in the DataFrame
    for index in range(len(df)):
        temp_df = pd.read_csv(StringIO(df.point[index]), header=None, names=['x', 'y'])
        temp_tensor = torch.tensor(temp_df.values).unsqueeze(0)
        tensors.append(temp_tensor)

    output = torch.cat(tensors, dim=0)

    return output

def encode_strings_optimized(model_name: str, strings: list, batch_size: int = 16,
                            save_output: str = None, cls_embed: bool= True):
    """
    Encodes a list of strings using a specified Hugging Face model in batches,
    optimized for cases with duplicate strings.

    Args:
    model_name (str): Name or path of the Hugging Face model.
    strings (list): List of input strings to encode.
    batch_size (int, optional): Batch size for encoding. Defaults to 16.
    save_output (str, optional): Path to save the output embeddings. Defaults to None.

    Returns:
    torch.Tensor: Tensor of encoded string embeddings in the original order.
    """

    # Initialize Accelerator for device management
    accelerator = Accelerator()
    device = accelerator.device
    print(f"You are using {device} device.")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Handle missing padding token in the tokenizer
    # Some tokenizers may not have a padding token, which is required for batch processing
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            # Use end-of-sequence token as padding if available
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a new special padding token if none exists
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize model embeddings to include the new padding token
            model.resize_token_embeddings(len(tokenizer))

    # Step 1: Extract unique strings to avoid redundant computations
    unique_strings = list(OrderedDict.fromkeys(strings))  # Maintain order and remove duplicates
    string_to_index = {s: i for i, s in enumerate(unique_strings)}  # Map each unique string to its index
    original_to_unique = [string_to_index[s] for s in strings]  # Map original strings to unique indices

    # Step 2: Tokenize unique strings
    inputs = tokenizer(
        unique_strings,  # Tokenize only the unique strings
        return_tensors='pt',  # Return PyTorch tensors
        padding='max_length',  # Pad to the maximum sequence length
        truncation=True,  # Truncate sequences longer than max_length
        max_length=512  # Maximum sequence length (adjust if necessary)
    )

    # Step 3: Create TensorDataset and DataLoader
    # TensorDataset groups tensors into a dataset that can be iterated in batches
    dataset = TensorDataset(
        inputs['input_ids'],  # Input IDs tensor
        inputs['attention_mask']  # Attention mask tensor
    )
    # DataLoader splits the dataset into batches for efficient processing
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Step 4: Encode unique strings
    unique_encodings = []  # List to store embeddings of unique strings
    for batch in tqdm(dataloader, desc="Processing Unique Strings"):  # Process each batch
        # Move batch tensors to the appropriate device (CPU or GPU)
        input_ids, attention_mask = [t.to(device) for t in batch]
        with torch.no_grad():  # Disable gradient calculation for inference
            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # Extract the hidden states

            # Select the [CLS] token embedding for each sequence
            if cls_embed:
                cls_embeddings = hidden_states[:, 0, :]  # [CLS] token is at position 0
            else:
                cls_embeddings = hidden_states[:, :, :]  # [CLS] token is at position 0
            unique_encodings.append(cls_embeddings.cpu())  # Move to CPU and store

    # Combine all unique embeddings into a single tensor
    unique_encodings = torch.cat(unique_encodings, dim=0)  # [num_unique, hidden_dim]

    # Step 5: Map unique encodings back to the original order
    # Use the mapping to reconstruct the original sequence of embeddings
    original_encodings = torch.stack([unique_encodings[idx] for idx in original_to_unique], dim=0)

    # Step 6: Optionally save the output encodings
    if save_output:
        torch.save(original_encodings, f"{save_output}")  # Save embeddings as a .pt file

    return original_encodings  # Return the embeddings in the original order


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_cls_embeddings(text_list, batch_size=200):
    """
    Get text [cls] embeddings.
    """
    all_embeddings = []
    embedding_cache = {}

    # Initialize tqdm for batch processing
    progress_bar = tqdm(total=len(text_list), desc='Getting Embeddings', position=0, leave=True)

    # Process texts in batches
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        batch_embeddings = []

        # Identify texts that are already in the cache
        uncached_texts = [text for text in batch_texts if text not in embedding_cache]

        if uncached_texts:
            # Tokenize and encode the uncached texts
            encoded_input = tokenizer(
                uncached_texts, padding='max_length', max_length=512, truncation=True, return_tensors="pt"
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

            # Inference with no gradient calculation
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Pooling and converting to numpy array
            embeddings = cls_pooling(model_output).detach().cpu().numpy()

            # Store the embeddings in the cache
            for text, embedding in zip(uncached_texts, embeddings):
                embedding_cache[text] = embedding

        # Retrieve embeddings from the cache
        for text in batch_texts:
            batch_embeddings.append(embedding_cache[text])

        all_embeddings.append(np.stack(batch_embeddings))

        # Update tqdm progress bar
        progress_bar.update(len(batch_texts))

    progress_bar.close()

    # Concatenate all the batch embeddings
    return np.concatenate(all_embeddings, axis=0)


def gen_ele_embed(com_fes, inference_batch=200, template_type=1):

    # Load company embeddings
    # com_saved_emd = pd.read_csv(r'E:\shaohantian\research\work1_research\datasets\input_encodings\ele_embed.csv')
    df_embeddings_symbol = generate_element_embeddings(
        save_file_name=f"./datasets/ele_embed_t{str(template_type)}.csv",
        template_choice=1 # Use "{symbol}"
    )
    com_saved_emd = pd.read_csv(f'./datasets/ele_embed_t{str(template_type)}.csv')
    # Convert embeddings to torch tensor and move to GPU
    com_saved_emd_tensor = torch.tensor(com_saved_emd.iloc[:, 1:].values, dtype=torch.float32).to('cuda')
    symbols = com_saved_emd['symbol'].values

    # Create a dictionary for quick lookup
    embedding_dict = {symbol: com_saved_emd_tensor[i] for i, symbol in enumerate(symbols)}

    ele_embed = []
    progress_bar = tqdm(total=len(com_fes), desc='Composition embeddings', position=0, leave=True)
    for cos in com_fes:
        sum_embd = []
        com_value = []
        for name, v in cos.items():
            if name in embedding_dict:
                embd = embedding_dict[name]
                sum_embd.append(embd)
                com_value.append(v)
        if sum_embd:
            sum_embd = torch.stack(sum_embd)
            com_value = torch.tensor(com_value, dtype=torch.float32).to('cuda')
            com_embeds = torch.mv(sum_embd.T, com_value)
            ele_embed.append(com_embeds.unsqueeze(0))

        # Update tqdm progress bar
        progress_bar.update(1)

    progress_bar.close()
    ele_embed = torch.cat(ele_embed, dim=0)
    # ele_embed = normalize_array(ele_embed)

    return ele_embed.detach().cpu()

# Convert specified columns to list of dictionaries
def convert_to_dict_list(df):
    columns = df.columns
    dict_list = [dict(zip(columns, row)) for row in df.itertuples(index=False, name=None)]
    return dict_list

def gen_text_embed(input_texts):

    output = None
    for text in input_texts:
        inputs = tokenizer(text,
                        return_tensors='pt',
                        padding='max_length',
                        truncation=True,
                        max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        seq_embed = outputs.last_hidden_state

        if output is None:
            output = seq_embed
        else:
            # output = torch.cat([output, seq_embed], dim=0)
            output += seq_embed

    return output.detach().cpu()

def convert_to_lists(data_string):
    """Convert a comma-separated string to two lists of high-precision floats."""
    list1, list2 = [], []
    for line in data_string.strip().split("\n"):
        if line:
            values = line.split(",")
            list1.append(float(Decimal(values[0].strip())))  # Use Decimal for precision
            list2.append(float(Decimal(values[1].strip())))
    return list1, list2


def find_non_increasing_rows(tensor):
    """
    Find and print the indices of rows in the tensor that do not strictly increase.

    Args:
    tensor (torch.Tensor): A 2D PyTorch tensor of shape [N, M].
    """
    # Compute the difference between consecutive elements in each row
    diffs = tensor[:, 1:] - tensor[:, :-1]

    # Find rows where any difference is non-positive (i.e., not strictly increasing)
    non_increasing_rows = torch.where((diffs <= 0).any(dim=1))[0]

    # Print the indices and corresponding rows
    if len(non_increasing_rows) > 0:
        print("Rows that do not strictly increase:")
        for idx in non_increasing_rows:
            # print(f"Row {idx.item()}: {tensor[idx].tolist()}")
            print(f"Row {idx.item()}")
    else:
        print("All rows strictly increase.")


def fix_non_increasing_rows(voltage_original, current_density_original, row_idx_list,
                            save_dir=None, epsilon=1e-6, max_row_iter=None):
    """
    Iteratively fix non-increasing voltage rows to make them strictly increasing,
    while preserving "bottom points". Visualizes voltage-current_density curves.

    Args:
        voltage_original (torch.Tensor): Shape [N, M], original voltage data.
        current_density_original (torch.Tensor): Shape [N, M], corresponding current density.
        row_idx_list (list of int): Indices of rows to visualize.
        save_dir (str, optional): Directory to save plots. If None, plots are shown.
        epsilon (float, optional): Small value to ensure strict increase (e.g., V_new = V_prev + epsilon).
        max_row_iter (int, optional): Max iterations for fixing each row. Defaults to 2 * M (number of columns).

    Returns:
        torch.Tensor: Fixed voltage tensor.
    """
    if not isinstance(voltage_original, torch.Tensor) or voltage_original.dim() != 2 or \
        not isinstance(current_density_original, torch.Tensor) or current_density_original.dim() != 2:
        print("Error: Voltage and Current Density inputs must be 2D PyTorch tensors.")
        return voltage_original.clone() if isinstance(voltage_original, torch.Tensor) else None

    if voltage_original.shape != current_density_original.shape:
        print(f"Error: Voltage (shape {voltage_original.shape}) and Current Density (shape {current_density_original.shape}) tensors must have the same shape.")
        return voltage_original.clone()

    voltage = voltage_original.clone()  # This is the tensor we will modify and return
    N, M = voltage.shape

    if M <= 1:
        print("Tensor has only one column or is empty. No fixing applied as sequences are too short.")
        # Minimal plotting for single-point data if requested.
        if M == 1:
            for plot_idx in row_idx_list:
                if 0 <= plot_idx < N:
                    plt.figure(figsize=(10, 6))
                    plt.plot(voltage[plot_idx].cpu().numpy(), current_density_original[plot_idx].cpu().numpy(),
                            label=f"Data for Row {plot_idx}", marker="o", markersize=4, color='blue')
                    plt.xlabel("Voltage (V)")
                    plt.ylabel("Current Density (A/cm²)")
                    plt.yscale("log")
                    plt.title(f"Row {plot_idx}: Voltage vs Current Density (Single Data Point)")
                    plt.legend()
                    plt.grid(True, which="both", ls="-")
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(os.path.join(save_dir, f"row_{plot_idx}_single_point.png"), dpi=300)
                        plt.close()
                    else:
                        plt.show()
        return voltage

    if max_row_iter is None:
        max_row_iter = M * 2  # Default max iterations, can be tuned

    # Store original voltages from the unmodified input for plotting "Before Fix"
    original_voltages_for_plot = {}
    for plot_idx in row_idx_list:
        if 0 <= plot_idx < N:
            original_voltages_for_plot[plot_idx] = voltage_original[plot_idx].clone()

    for i in range(N):  # Process each row
        row = voltage[i]  # Get a view of the row to be modified in `voltage`

        # Identify "bottom points" for this row based on its state *before* its iterative fixing.
        # These bottom_points will remain constant during the inner `_iter` loop for THIS row `i`.
        bottom_points_mask = torch.zeros_like(row, dtype=torch.bool)
        if M > 2:  # For points between first and last
            # Use the current state of the row (which might have been affected by fixing previous rows)
            # to determine bottom points for *this* row's fixing process.
            current_row_state_for_bp = row.clone()
            bottom_points_mask[1:-1] = (current_row_state_for_bp[1:-1] < current_row_state_for_bp[:-2]) & \
                                        (current_row_state_for_bp[1:-1] < current_row_state_for_bp[2:])

        for _iter in range(max_row_iter):
            changed_in_pass = False

            # --- Fix internal points (index 1 to M-2) ---
            if M > 2:  # Need at least 3 points for j in range(1, M-1)
                for j in range(1, M - 1):  # j is the index of the point to potentially fix
                    if bottom_points_mask[j]:  # Do not modify designated bottom points
                        continue

                    if row[j] <= row[j-1]:  # If non-increasing
                        avg_val = (row[j-1] + row[j+1]) / 2.0
                        target_val = max(avg_val, row[j-1] + epsilon)

                        if abs(row[j] - target_val) > (epsilon / 100.0) or row[j] < target_val : # Ensure change or if it's still too low
                            row[j] = target_val
                            changed_in_pass = True

            # --- Fix the last point (index M-1) if necessary ---
            if M > 1:  # Need at least two points (indices 0 and 1 for row[M-2] and row[M-1])
                # The last point row[M-1] cannot be a 'bottom_point' by the 3-point definition.
                if row[M-1] <= row[M-2]: # Compare last element with second to last
                    target_val = row[M-2] + epsilon
                    if abs(row[M-1] - target_val) > (epsilon / 100.0) or row[M-1] < target_val:
                        row[M-1] = target_val
                        changed_in_pass = True

            if not changed_in_pass:
                break  # Row has stabilized
        # After iterations, voltage[i] (which is `row`) is updated.

    # --- Plotting selected rows ---
    for plot_idx in row_idx_list:
        if not (0 <= plot_idx < N):
            print(f"Warning: row index {plot_idx} for plotting is out of bounds (0 to {N-1}). Skipping.")
            continue

        original_v_plot = original_voltages_for_plot.get(plot_idx)
        if original_v_plot is None: # Should ideally not happen with the storage logic
            print(f"Internal Warning: Original data for row {plot_idx} not found for plotting. Using current state from initially cloned tensor as 'original'.")
            original_v_plot = voltage_original[plot_idx].clone() # Fallback to originally passed tensor

        fixed_v_plot = voltage[plot_idx]  # This is from the modified `voltage` tensor
        j_density_plot = current_density_original[plot_idx]  # Use original current density

        plt.figure(figsize=(10, 6), dpi=300)  # High DPI for better quality
        plt.plot(original_v_plot.cpu().numpy(), j_density_plot.cpu().numpy(),
                label="Before Fix", marker="o", markersize=4,
                linestyle="--", linewidth=1.5, color='red')
        plt.plot(fixed_v_plot.cpu().numpy(), j_density_plot.cpu().numpy(),
                label="After Fix", marker="s", markersize=2, # User's markersize for fixed
                linestyle="-", linewidth=1.5, color='blue')

        plt.xlabel("Voltage (V)")
        plt.ylabel("Current Density (A/cm²)")
        plt.yscale("log")
        plt.title(f"Row {plot_idx}: Voltage vs Current Density (Log Scale Current)")
        plt.legend()
        plt.grid(True, which="both", ls="-") # Grid for major and minor ticks on log scale

        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                save_filename = os.path.join(save_dir, f"row_{plot_idx}_voltage_fix.png")
                plt.savefig(save_filename, dpi=300)
                print(f"Plot saved to {save_filename}")
                plt.close()  # Close figure after saving
            except Exception as e:
                print(f"Error saving plot for row {plot_idx}: {e}")
                plt.show() # Attempt to show if saving failed
        else:
            plt.show()

    return voltage


if __name__ == "__main__":
    test_dir = Path("./datasets")
    test_dir.mkdir(parents=True, exist_ok=True)
    ################## load data
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization study for curve generation.")
    parser.add_argument("--input_data_path", type=str, default="./../train_data/traindata_postprocess.xlsx",
            help="Path to the input datasets directory")
    parser.add_argument("--template_type", type=int, default=1, help="Template type for element embeddings")
    args = parser.parse_args()

    data_path = args.input_data_path
    template_type = args.template_type

    if "test" in data_path:
        save_pre_idx = "test_"
    elif "train" in data_path:
        save_pre_idx = "train_"
    else:
        save_pre_idx = ""
    print(f"the data_pre_idx is '{save_pre_idx}'")
    df = pd.read_excel(data_path, na_filter=False)
    print(df.shape)

    ################## ele embeddiings ##################
    ele_embed = gen_ele_embed(convert_to_dict_list(df.loc[:, 'H':'W']), inference_batch=1024, template_type=template_type)
    ele_embed = torch.tensor(ele_embed, dtype=torch.float32)
    torch.save(ele_embed.to(torch.float32), f"./datasets/{save_pre_idx}ele_embed.pt")
    print(f"ele_embed shape: {ele_embed.shape}")
    # exprot as .csv file
    pd.DataFrame(
        ele_embed.cpu().numpy(),
        columns=[f"ele_d{j}" for j in range(768)],
    ).to_csv(f"./datasets/{save_pre_idx}ele_embed.csv", index=False)

    ################## text embeddings ##################
    batch_size = 256
    columns_to_encode = ['mat', 'polar', 'character']
    # columns_to_encode = ['character']

    # Encode embeddings for each column (each should be [N, 768])
    embeddings = [
        encode_strings_optimized(
            model_name,
            list(df[col].fillna("").astype(str)),  # robust to NaN
            batch_size=batch_size,
            cls_embed=True
        )
        for col in columns_to_encode
    ]
    # ----------------------------
    # Method A (recommended): CONCAT (N, 768 * num_cols) e.g., (N, 2304)
    # ----------------------------
    text_embed_cat = torch.cat(embeddings, dim=-1)                 # [N, 768*C]
    torch.save(text_embed_cat.to(torch.float32), f"./datasets/{save_pre_idx}text_embed.pt")
    print(f"text_embed_cat shape: {tuple(text_embed_cat.shape)}")

    # exprot as .csv file
    pd.DataFrame(
        text_embed_cat.cpu().numpy(),
        columns=[f"process_d{j}" for j in range(768)]+[f"condition_d{j}" for j in range(768)]+[f"character_d{j}" for j in range(768)],
    ).to_csv(f"./datasets/{save_pre_idx}text_embed.csv", index=False)


    ################## voltage ##################
    df["x_point"], df["y_point"] = zip(*df["uniform_point"].apply(convert_to_lists))
    voltage_embed = torch.tensor(df["x_point"].tolist(), dtype=torch.float32)
    current_embed = torch.tensor(df["y_point"].tolist(), dtype=torch.float32)
    # current_embed = -1 * torch.log10(current_embed)  # Log-transform current density for better numerical stability and visualization

    # Check for non-increasing rows
    print("-------------- before fixing the voltage")
    find_non_increasing_rows(voltage_embed)
    voltage_increase_embed = fix_non_increasing_rows(
        voltage_embed,
        current_embed,
        # row_idx_list=list(range(200)),
        row_idx_list=list(range(16)),
        save_dir="./figs/fix_plots"
    )
    print("--------------- after fixing the voltage")
    find_non_increasing_rows(voltage_increase_embed)

    # export the tensors
    torch.save(voltage_increase_embed, f"./datasets/{save_pre_idx}voltage_embed.pt")
    torch.save(current_embed, f"./datasets/{save_pre_idx}current_embed.pt")
    print(f"Voltage Embed Shape: {voltage_increase_embed.shape}")
    print(f"Current Embed Shape: {current_embed.shape}")

    # exprot as .csv file
    pd.DataFrame(
        voltage_increase_embed.cpu().numpy(),
        columns=[f"voltage_d{j}" for j in range(256)]
    ).to_csv(f"./datasets/{save_pre_idx}voltage_embed.csv", index=False)

    # exprot as .csv file
    pd.DataFrame(
        current_embed.cpu().numpy(),
        columns=[f"current_d{j}" for j in range(256)],
    ).to_csv(f"./datasets/{save_pre_idx}current_embed.csv", index=False)