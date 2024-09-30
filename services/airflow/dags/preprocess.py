import os
import re
from sklearn.model_selection import train_test_split
import yaml
import nltk
import pandas as pd

def sample_data(raw_data_path, 
                output_chunk_path, 
                chunk_id, 
                chunk_count):
    """
    Sample data from a CSV file into chunks and save them separately.

    Args:
        raw_data_path (str): Path to the input CSV file.
        output_chunk_path (str): Path where to save the sampled chunks.
        chunk_id (int): Current chunk ID (0-indexed).
        chunk_count (int): Total number of chunks to split the data into.

    Returns:
        None

    Notes:
        - The function splits the data into equal-sized chunks +- (chunk_count - 1).
        - Chunk is saved as a separate CSV file.
        - The original file is not modified.
    """
    print(raw_data_path, 
                output_chunk_path, 
                chunk_id, 
                chunk_count)
    try:
        df = pd.read_csv(raw_data_path, low_memory=False, encoding='latin-1')
        chunk_id = chunk_id % chunk_count
        start = chunk_id * int(len(df) / chunk_count)
        end = min((chunk_id + 1) * int(len(df) / chunk_count), len(df))
        chunk = df[start:end]
        chunk.to_csv(output_chunk_path, index=False)
        print(f"Sampled data part {chunk_id + 1} saved to {output_chunk_path}")
    except Exception as e:
        print(f"An error during saving sample: {e}")
        raise


def lower_text(text: str):
    return text.lower()

def remove_numbers(text: str):
    text_nonum = re.sub(r'\d+', ' ', text)
    return text_nonum

def remove_punctuation(text: str):
    pattern = r'[^\w\s]'
    text_nopunct = re.sub(pattern, ' ', text)
    return text_nopunct

def remove_multiple_spaces(text: str):
    pattern = r'\s{2,}'
    text_no_doublespace = re.sub(pattern, ' ', text)
    return text_no_doublespace

def tokenize_text(text: str) -> list[str]:
    return nltk.tokenize.word_tokenize(text)

def remove_stop_words(tokenized_text: list[str]) -> list[str]:
    stopwords = nltk.corpus.stopwords.words('english')
    filtered = [w for w in tokenized_text if not w in stopwords]
    return filtered

def stem_words(tokenized_text: list[str]) -> list[str]:
    stemmer = nltk.stem.PorterStemmer()
    stemmed = [stemmer.stem(w) for w in tokenized_text]
    return stemmed

def preprocessing_stage_text(text):
    _lowered = lower_text(text)
    _without_numbers = remove_numbers(_lowered)
    _without_punct = remove_punctuation(_without_numbers)
    _single_spaced = remove_multiple_spaces(_without_punct)
    _tokenized = tokenize_text(_single_spaced)
    _without_sw = remove_stop_words(_tokenized)
    _stemmed = stem_words(_without_sw)

    return _stemmed


def handle_initial_data(path_to_data):
    df = pd.read_csv(path_to_data)
    df.columns = ['text', 'target']
    df['text'] = df['text'].astype('str')
    df['text'] = df['text'].apply(preprocessing_stage_text)
    df['target'] += 1
    df.to_csv(path_to_data, index=False)


def split_data(
    path_in,
    path_out,
    test_size,
    random_state
):
    data_preprocessed = pd.read_csv(path_in)
    train, test = train_test_split(data_preprocessed, 
                                   test_size=test_size, 
                                   random_state=random_state)
    train_path = os.path.join(path_out, 'train.csv')
    test_path = os.path.join(path_out, 'test.csv')
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    


if __name__ == "__main__":
    # with open('./configs/main.yml', 'r') as f:
    #     config = yaml.safe_load(f)
    # path_to_raw_data = os.path.join(config['BASE_DIR'], config['RAW_DATA'])
    # preprocessed_df = handle_initial_data(path_to_raw_data)
    # print(preprocessed_df.head())
    print("Adam")