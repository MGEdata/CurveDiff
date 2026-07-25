from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F # For cosine_similarity

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Configuration ---
model_path = 'E:/.cache/huggingface/hub/models--mge-llms--steelbert'
target_word = "steel" # The word whose embeddings we want to compare

# --- Load Model and Tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model/tokenizer from {model_path}: {e}")
    print("Please ensure the model_path is correct and the model files are accessible.")
    exit()

# --- Classic Example Sentences ---
# This curated list provides clear examples for different similarity levels.
sentence_data = [
    # Category 1: Significant Difference (Literal Material vs. Figurative Quality)
    {"id": "S1_Literal", "text": "The construction crew welded the steel girders for the skyscraper's frame."},
    {"id": "S2_Figurative", "text": "Despite the pressure, the diplomat showed nerves of steel during the intense negotiations."},

    # Category 2: Very High Similarity (Almost Identical Technical Context)
    {"id": "S3_TechA", "text": "The engineer specified a high-grade steel for the load-bearing components."},
    {"id": "S4_TechB", "text": "For the critical load-bearing parts, a superior quality steel was mandated by the engineer."},

    # Category 3: Moderate Similarity (Different but Related Industrial Applications)
    {"id": "S5_AppA", "text": "The automotive sector requires vast quantities of sheet steel for car manufacturing."},
    {"id": "S6_AppB", "text": "Reinforced steel bars are crucial for the stability of concrete structures in construction."}
]

sentences = [s["text"] for s in sentence_data]

# --- Tokenization and Embedding Generation ---
print("\nTokenizing sentences...")
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

print("\nGenerating embeddings...")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

last_hidden_states = outputs.hidden_states[-1]

# --- Helper Function to Get Word Embedding ---
def get_word_embedding(input_ids_single_sentence, hidden_state_single_sentence, target_word_str, tokenizer_instance, sentence_text_for_debug):
    tokens = tokenizer_instance.convert_ids_to_tokens(input_ids_single_sentence, skip_special_tokens=False)
    target_word_subwords = tokenizer_instance.tokenize(target_word_str)

    if not target_word_subwords:
        print(f"Warning: Tokenizer produced no subwords for target word '{target_word_str}'.")
        return None

    found_indices = []
    for i in range(len(tokens) - len(target_word_subwords) + 1):
        if tokens[i:i+len(target_word_subwords)] == target_word_subwords:
            found_indices = list(range(i, i + len(target_word_subwords)))
            break

    if not found_indices:
        print(f"  Warning: Could not find unambiguous token sequence for '{target_word_str}' (expected subwords: {target_word_subwords}) in sentence: \"{sentence_text_for_debug}\".\n    Sentence tokens: {tokens}")
        return None

    word_embedding = hidden_state_single_sentence[found_indices, :].mean(dim=0)
    return word_embedding

# --- Extract Embeddings for the Target Word in Each Sentence ---
print(f"\nExtracting embeddings for the word: '{target_word}'")
word_embeddings_map = {}

target_word_tokenized_example = tokenizer.tokenize(target_word)
print(f"The target word '{target_word}' is tokenized by this model as: {target_word_tokenized_example}")
if not target_word_tokenized_example:
    print(f"CRITICAL WARNING: Tokenizer returned empty list for target word '{target_word}'. This will likely cause embedding extraction to fail.")

for i, data in enumerate(sentence_data):
    sentence_id = data["id"]
    sentence_text = data["text"]
    input_ids_sent = inputs['input_ids'][i]
    hidden_state_sent = last_hidden_states[i]

    print(f"\nProcessing Sentence {sentence_id}: \"{sentence_text}\"")
    embedding = get_word_embedding(input_ids_sent, hidden_state_sent, target_word, tokenizer, sentence_text)

    if embedding is not None:
        word_embeddings_map[sentence_id] = {"embedding": embedding, "text": sentence_text}
        print(f"  Successfully extracted embedding for '{target_word}' in sentence {sentence_id}.")
    else:
        print(f"  Failed to extract embedding for '{target_word}' in sentence {sentence_id}.")

# --- Compare Embeddings ---
print(f"\n\n--- Comparing Embeddings for '{target_word}' ---")
sentence_ids_with_embeddings = list(word_embeddings_map.keys())

if len(sentence_ids_with_embeddings) < 2:
    print("Need at least two sentences with successful embeddings to compare.")
else:
    # Compare specific "classic" pairs first for clarity
    classic_pairs_to_compare = [
        ("S1_Literal", "S2_Figurative"),  # Expect low similarity
        ("S3_TechA", "S4_TechB"),        # Expect very high similarity
        ("S5_AppA", "S6_AppB"),          # Expect moderate similarity
        ("S1_Literal", "S3_TechA"),      # Compare literal construction to technical spec
        ("S2_Figurative", "S5_AppA")     # Compare figurative to an application
    ]

    for id1, id2 in classic_pairs_to_compare:
        if id1 in word_embeddings_map and id2 in word_embeddings_map:
            emb1_data = word_embeddings_map[id1]
            emb2_data = word_embeddings_map[id2]

            emb1 = emb1_data["embedding"]
            text1 = emb1_data["text"]
            emb2 = emb2_data["embedding"]
            text2 = emb2_data["text"]

            emb1_unsqueezed = emb1.unsqueeze(0)
            emb2_unsqueezed = emb2.unsqueeze(0)

            cos_sim = F.cosine_similarity(emb1_unsqueezed, emb2_unsqueezed).item()
            euclidean_dist = torch.cdist(emb1_unsqueezed, emb2_unsqueezed, p=2).item()

            print(f"\nComparison between Classic Pair: Sentence {id1} and Sentence {id2}:")
            print(f"  Sentence {id1}: \"{text1}\"")
            print(f"  Sentence {id2}: \"{text2}\"")
            print(f"  Cosine Similarity for '{target_word}': {cos_sim:.4f}")
            print(f"  Euclidean Distance for '{target_word}': {euclidean_dist:.4f}")

            if cos_sim > 0.9:
                print("    Interpretation (Cosine): Very highly similar contextual meaning (almost interchangeable).")
            elif cos_sim > 0.75:
                print("    Interpretation (Cosine): Strongly similar contextual meaning.")
            elif cos_sim > 0.6:
                print("    Interpretation (Cosine): Moderately similar contextual meaning, sharing key aspects.")
            elif cos_sim > 0.4:
                print("    Interpretation (Cosine): Somewhat related, but contextually quite different.")
            else:
                print("    Interpretation (Cosine): Contextually very different or largely unrelated.")
        else:
            print(f"\nSkipping comparison for ({id1}, {id2}): one or both embeddings not found.")


    # Optionally, you can add a loop here to compare all other pairs if desired,
    # similar to the previous version's all-pairs comparison loop.
    # For this "classic examples" version, we'll stick to the defined pairs.

print("\n--- End of Analysis ---")

