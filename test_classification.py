import os
import matplotlib.pyplot as plt
import pandas as pd
from datasets import list_datasets, load_from_disk
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.xpu import device
from transformers import AutoTokenizer, AutoModel
import time
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def explore_huggingface_datasets():
    all_datasets = list_datasets()
    print(f"there are {len(all_datasets)} datasets currently in the HuggingFace Hub...")
    print(f"the first 10 are:\n{all_datasets[:10]}\n")

def get_emotions_dataset():
    emotions = load_dataset("emotion")
    print(f"loaded the emotion DataSet:\n{emotions}\n")
    return emotions

# global
emotions = get_emotions_dataset()
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, clean_up_tokenization_spaces=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads((os.cpu_count() - 2))
print(f"running torch on device: {device}")
print(f"torch.cpu.current_device: {torch.cpu.current_device()}")
print(f"torch.cpu.device_count: {torch.cpu.device_count()}")
print(f"torch num of threads: {torch.get_num_threads()}")
print(f"current number of cores: {os.cpu_count()}")
model = AutoModel.from_pretrained(model_ckpt).to(device)
print(f"using model: {model}")
labels = emotions["train"].features["label"].names

def explore_huggingface_emotion():
    train_ds = emotions["train"]
    validation_ds = emotions["validation"]
    print(f"the training emotion DataSet:\n{train_ds}\n")
    print(f"the validation emotion DataSet:\n{validation_ds}\n")
    print(f"the training emotion DataSet features:{train_ds.features}")
    print(f"the training emotion DataSet data:{train_ds.data}")
    print(f"the training emotion DataSet format:{train_ds.format}")
    print(f"the training emotion DataSet cache files:{train_ds.cache_files}")
    print(f"the training emotion DataSet first five entries:{train_ds[:5]}")

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

def convert_emotions_from_dataset_to_dataframe():
    emotions.set_format(type="pandas")
    df = emotions["train"][:]
    print(f"head of emotions DataFrame:\n{df.head()}\n")
    df["label_name"] = df["label"].apply(label_int2str)
    print(f"enriched head of emotions DataFrame:\n{df.head()}\n")
    """
    When working on text classification problems, it is a good idea to examine
    the distribution of examples across the classes. A dataset with a skewed
    class distribution might require a different treatment in terms of the training
    loss and evaluation metrics than a balanced one.
    """
    df["label_name"].value_counts(ascending=True).plot.barh()
    plt.title("Frequency of Classes")
    """
    Transformer models have a maximum input sequence length that is referred to
    as the maximum context size. For applications using DistilBERT, the maximum
    context size is 512 tokens, which amounts to a few paragraphs of text.
    """
    df["Words Per Tweet"] = df["text"].str.split().apply(len)
    df.boxplot("Words Per Tweet", by="label_name", grid=False,
               showfliers=False, color="black")
    plt.suptitle("")
    plt.xlabel("")
    plt.show()
    emotions.reset_format()
    input()

"""
Transformer models like DistilBERT cannot receive raw strings as input; instead,
they assume the text has been tokenized and encoded as numerical vectors.
Tokenization is the step of breaking down a string into the atomic units used
in the model. There are several tokenization strategies one can adopt, and the
optimal splitting of words into subunits is usually learned from the corpus.
Before looking at the tokenizer used for DistilBERT, let’s consider two extreme
cases: character and word tokenization.
"""
def tokenize_to_characters():
    text = "Tokenizing text is a core task of NLP."
    tokenized_text = list(text)
    print(f"text:{text}")
    print(f"tokenized_text:{tokenized_text}")
    token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
    print(f"tokenized_text -> integer:{token2idx}")
    #  use token2idx to transform the tokenized text to a list of integers
    input_ids = [token2idx[token] for token in tokenized_text]
    print(f"tokenized_text -> input_ids:{input_ids}")
    """
    The last step is to convert input_ids to a 2D tensor of one-hot vectors.
    One-hot vectors are frequently used in machine learning to encode 
    categorical data, which can be either ordinal or nominal.
    """
    input_ids = torch.tensor(input_ids) # Constructs tensor with no autograd history (also known as a "leaf tensor")
    one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
    print(f"constructed a one_hot_encodings vector: {one_hot_encodings.shape}")
    print("testing the one-hot vector:")
    print(f"tokenized_text[0]: {tokenized_text[0]}")
    print(f"tensor index[0]: {input_ids[0]}")
    print(f"One-hot vector[0]: {one_hot_encodings[0]}")
    print(f"tokenized_text[1]: {tokenized_text[1]}")
    print(f"tensor index[1]: {input_ids[1]}")
    print(f"One-hot vector[1]: {one_hot_encodings[1]}")

"""
Character-level tokenization ignores any structure in the text and treats the
whole string as a stream of characters. Although this helps deal with 
misspellings and rare words, the main drawback is that linguistic structures
such as words need to be learned from the data. This requires significant 
compute, memory, and data. For this reason, character tokenization is rarely
used in practice. Instead, some structure of the text is preserved during the
tokenization step. Word tokenization is a straightforward approach to achieve this.

The basic idea behind subword tokenization is to combine the best aspects of
character and word tokenization. On the one hand, we want to split rare words
into smaller units to allow the model to deal with complex words and
misspellings. On the other hand, we want to keep frequent words as unique
entities so that we can keep the length of our inputs to a manageable size.
The main distinguishing feature of subword tokenization (as well as word
tokenization) is that it is learned from the pretraining corpus using a mix of
statistical rules and algorithms.
"""
def tokenize_to_words():
    text = "Tokenizing text is a core task of NLP."
    tokenized_text = text.split()
    print(f"text:{text}")
    print(f"tokenized_text:{tokenized_text}")
    """
    There are several subword tokenization algorithms that are commonly used in
    NLP, but let’s start with WordPiece,5 which is used by the BERT and DistilBERT
    tokenizers.
    """
    print(f"created tokenizer {tokenizer} from model {model_ckpt}.")
    encoded_text = tokenizer(text)
    print(f"encoded_text:{encoded_text}")
    # converting input_ids back into tokens to test
    tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
    print(f"converting encoded_text (input_ids) back into tokens:{tokens}")
    str_tokens = tokenizer.convert_tokens_to_string(tokens)
    print(f"converting tokens to string :{str_tokens}")
    # information about this tokenizer, such as the vocabulary size:
    print(f"model {model_ckpt} tokenizer info:\ntokenizer vocab length: {tokenizer.model_max_length}"
          f"\ntokenizer fields that model expects in forward pass: {tokenizer.model_input_names}")

"""
To tokenize the whole corpus, we’ll use the map() method of our DatasetDict object.
"""
def tokenize(batch):
    # padding=True will pad the examples with zeros to the size of the longest one in a batch
    # truncation=True will truncate the examples to the model’s maximum context size
    return tokenizer(batch["text"], padding=True, truncation=True)

def tokenize_entire_dataset():
    print(f"testing tokenizer(batch) with just 2 tweets:\n{tokenize(emotions["train"][:2])}")
    print(f"before mapping to tokenizer, emotions column names: {emotions["train"].column_names}")
    """
    By default, the map() method operates individually on every example in the
    corpus, so setting batched=True will encode the tweets in batches. Because
    we’ve set batch_size=None, our tokenize() function will be applied on the
    full dataset as a single batch. This ensures that the input tensors and
    attention masks have the same shape globally, and we can see that this
    operation has added new input_ids and attention_mask columns to the dataset.
    """
    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    print(f"after mapping to tokenizer, emotions column names: {emotions_encoded["train"].column_names}")
    return emotions_encoded

def extract_hidden_states(batch):
    start_time = time.time()
    # place model inputs on GPU if possible
    inputs = {k:v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    # extract last hidden states
    with torch.no_grad():
        start_time = time.time()
        last_hidden_state = model(**inputs).last_hidden_state
    # return vector for [CLS] token
    cls_hidden_state = {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}
    print(f"--- generated cls_hidden_state in {time.time() - start_time}[s] ---")
    return cls_hidden_state

def visualize_hidden_states(X_train, y_train):
    # scale features to [0,1] range
    X_scaled = MinMaxScaler().fit_transform(X_train)
    # initialize and fit UMAP
    mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
    # create a DataFrame of 2D embeddings
    df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
    df_emb["label"] = y_train
    print(f"after scaling the features to [0, 1] and creating a UMAP object {mapper}...")
    print(f"got a DataFrame of 2D embeddings:\n{df_emb.head()}\n")
    """
    The result is an array with the same number of training samples, but with
    only 2 features instead of the 768 we started with! Let’s investigate the
    compressed data a little bit further and plot the density of points for
    each category separately.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    axes = axes.flatten()
    cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_emb.query(f"label == {i}")
        axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                       gridsize=20, linewidths=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("DistilBERT Normalized Confusion Matrix")
    plt.show()

"""
Using a transformer as a feature extractor is fairly simple. we freeze the
body’s weights during training and use the hidden states as features for the
classifier. The advantage of this approach is that we can quickly train a
small or shallow model. Such a model could be a neural classification layer
or a method that does not rely on gradients, such as a random forest. This
method is especially convenient if GPUs are unavailable, since the hidden
states only need to be precomputed once.
"""
def training_using_feature_extractor():
    """
    To warm up, let’s retrieve the last hidden states for a single string. The
    first thing we need to do is encode the string and convert the tokens to
    PyTorch tensors. This can be done by providing the return_tensors="pt"
    argument to the tokenizer.
    """
    text = "this is a test"
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Input tensor shape: {inputs['input_ids'].size()}")
    """
    ... the resulting tensor has the shape [batch_size, n_tokens]. Now that we
    have the encodings as a tensor, the final step is to place them on the
    same device as the model and pass the inputs.
    """
    inputs = {k:v.to(device) for k, v in inputs.items()}
    # torch.no_grad() ctx manager disables the automatic calculation of the gradient.
    # useful for inference since it reduces the memory footprint of the computations.
    with torch.no_grad():
        start_time = time.time()
        outputs = model(**inputs)
        print(f"--- generated tensor outputs in {time.time() - start_time}[s] ---")
    print(f"tensor outputs:\n{outputs}\n")
    print(f"original text:{text}")
    print(f"encoded text:{inputs}")
    for k, v in inputs.items():
        print(f"inputs.items() key: {k}, value: {v}")
        if k == "input_ids":
            tokens = tokenizer.convert_ids_to_tokens(v[0])
            print(f"converting encoded_text (input_ids) back into tokens:{tokens}")
            str_tokens = tokenizer.convert_tokens_to_string(tokens)
            print(f"converting tokens to string :{str_tokens}")
    """
    tensor output last_hidden_stage shape: [batch_size, n_tokens, hidden_dim].
    In other words, a 768-dimensional vector is returned for each of the 6
    input tokens.
    tensor output: BaseModelOutput with just one attribute: last_hidden_state
    """
    print(f"tensor outputs last_hidden_state size: {outputs.last_hidden_state.size()}")
    """
    For classification tasks, it is common practice to just use the hidden
    state associated with the [CLS] token as the input feature. Since this
    token appears at the start of each sequence, we can extract it by simply
    indexing into outputs.last_hidden_state
    """
    print(f"last_hidden_state [CLS] token:\'{outputs.last_hidden_state[:0]}\n")
    print(f"last_hidden_state [CLS] token shape:\'{outputs.last_hidden_state[:0].size()}\n")
    """
    Now we know how to get the last hidden state for a single string; let’s do
    the same for the whole dataset by creating a new hidden_state column that
    stores all these vectors. As we did with the tokenizer, we’ll use the map()
    method of DatasetDict to extract all the hidden states in one go.
    """
    emotions_encoded = tokenize_entire_dataset()
    emotions_encoded.set_format("torch",
                                columns=["input_ids", "attention_mask"])

    emotions_hidden_reloaded = load_from_disk("./resources/emotions_hidden")
    if emotions_hidden_reloaded.num_rows["train"] <= 0:
        # go ahead and extract hidden states in one go
        emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

        print(f"after applying extract_hidden_states(), emotions_hidden column names "
              f"are: {emotions_hidden["train"].column_names}")
        print(f"emotions_hidden:\b{emotions_hidden}\n")

        # now save to disk
        emotions_hidden.save_to_disk("./resources/emotions_hidden")
    else:
        print(f"reloaded from disk: {emotions_hidden_reloaded}")
        print(f"num_rows[\"train\"] reloaded from disk: {emotions_hidden_reloaded.num_rows['train']}")
        emotions_hidden = emotions_hidden_reloaded

    """
    Now that we have the hidden states associated with each tweet, the next
    step is to train a classifier on them, we’ll need a feature matrix.
    The preprocessed dataset now contains all the information we need to train
    a classifier on it. We will use the hidden states as input features and
    the labels as targets. We can easily create the corresponding arrays in
    the well-known Scikit-learn format. 
    """
    X_train = np.array(emotions_hidden["train"]["hidden_state"])
    X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
    y_train = np.array(emotions_hidden["train"]["label"])
    y_valid = np.array(emotions_hidden["validation"]["label"])
    print("training a classifier using emotions_hidden...")
    print(f"X_train.shape: {X_train.shape}, X_valid.shape: {X_valid.shape}")
    print(f"y_train.shape: {y_train.shape}, y_valid.shape: {y_valid.shape}")

    visualize_hidden_states(X_train, y_train)

    """
    Training a simple classifier:
    from visualize_hidden_states(), we can see the hidden states are somewhat
    different between the emotions, although for several of them there is no
    obvious boundary. Let’s use these hidden states to train a logistic
    regression model with Scikit-learn. Training such a simple model is fast
    and does not require a GPU.
    """
    lr_clf = LogisticRegression(max_iter=3000)
    lr_clf.fit(X_train, y_train)
    lr_clf_score = lr_clf.score(X_valid, y_valid) #mean accuracy on test data & labels
    print(f"Logistic Regression training of our DistilBERT embeddings on X_train, y_train, "
          f"the mean accuracy score is: {lr_clf_score}")

    """
    we can examine whether our model is any good by comparing it against a
    simple baseline. In Scikit-learn there is a DummyClassifier that can be
    used to build a classifier with simple heuristics such as always choosing
    the majority class or always drawing a random class. In this case the
    best-performing heuristic is to always choose the most frequent class,
    which yields an accuracy of about 35%.
    """
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    dummy_clf_score = dummy_clf.score(X_valid, y_valid) #mean accuracy on test data & labels
    print(f"DummyClassifier(most_frequent) for our baseline on X_train, y_train, "
          f"the mean accuracy score is: {dummy_clf_score}")
    """
    We can further investigate the performance of the model by looking at the
    confusion matrix of the classifier, which tells us the relationship between
    the true and predicted labels.
    """
    y_preds = lr_clf.predict(X_valid)
    plot_confusion_matrix(y_preds, y_valid, labels)

def main():
    print("starting text_classification...\n")
    # explore_huggingface_datasets()
    # explore_huggingface_emotion()
    # convert_emotions_from_dataset_to_dataframe()
    # tokenize_to_characters()
    # tokenize_to_words()
    # tokenize_entire_dataset()
    training_using_feature_extractor()

if __name__ == '__main__':
    main()
