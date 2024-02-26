import numpy as np
import argparse
import json
import re
import csv
import os

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

def check_for_uppercase(comment):
    # Only use this function if the token is >= 3 letters long
    uppercase_count = 0
    word_dict = {}
    split_comment = comment.split()

    for token in split_comment:
        split_token = token.split('/')
        body = split_token[0]
        tag = split_token[1]
        if len(body) >= 3 and body.isupper():
            uppercase_count += 1
            word_dict[body] = body.lower()

    return uppercase_count, word_dict

def check_first_person(comment):
    tokens = comment.split()
    first_person_count = sum(token.split('/')[0] in FIRST_PERSON_PRONOUNS for token in tokens)

    return first_person_count

def check_second_person(comment):
    tokens = comment.split()
    second_person_count = sum(token.split('/')[0] in SECOND_PERSON_PRONOUNS for token in tokens)

    return second_person_count

def check_third_person(comment):
    tokens = comment.split()
    third_person_count = sum(token.split('/')[0] in THIRD_PERSON_PRONOUNS for token in tokens)

    return third_person_count

def check_coord_conjunc(comment):
    coord_conjunc_pattern = re.compile(r'\b(for|and|nor|but|or|yet|so)\b', flags=re.IGNORECASE)
    coord_conjunc_count = len(coord_conjunc_pattern.findall(comment))

    return coord_conjunc_count

def check_past_tense(comment):
    past_tense_pattern = re.compile(r'\/VBD( |$)')
    past_tense_count = len(past_tense_pattern.findall(comment))

    return past_tense_count

def check_future_tense(comment):
    # There are two regular expression patterns we need to make for this
    # The word can either be preceded by 'will' and 'gonna', followed by 'll, or is preceded by the phrase 'going to'
    # Source: https://www.yourdictionary.com/articles/future-tense-verbs 
    one_word_pattern = re.compile(r"(^| )(will|'ll|gonna)/") 
    phrase_pattern = re.compile(r'(^| )going\/\S+ to\/\S+ \S+\/VB( |$)')

    future_tense_count = len(one_word_pattern.findall(comment)) + len(phrase_pattern.findall(comment))
    return future_tense_count

def check_commas(comment):
    comma_pattern = re.compile(r'(^| ),\/,( |$)')
    comma_count = len(comma_pattern.findall(comment))

    return comma_count

def check_multi_punctuation(comment):
    multi_punc_pattern = re.compile(r"(^| )([^\s\w]{2,}(\"|[^\s\w])\/)")
    multi_punc_count = len(multi_punc_pattern.findall(comment))

    return multi_punc_count

def check_common_nouns(comment):
    common_noun_pattern = re.compile(r'\/(NN|NNS)( |$)')
    common_noun_count = len(common_noun_pattern.findall(comment))

    return common_noun_count

def check_proper_nouns(comment):
    proper_noun_pattern = re.compile(r'\/(NNP|NNPS)( |$)')
    proper_noun_count = len(proper_noun_pattern.findall(comment))

    return proper_noun_count

def check_adverbs(comment):
    adverb_pattern = re.compile(r'\/(RB|RBR|RBS|RP)( |$)')
    adverb_count = len(adverb_pattern.findall(comment))

    return adverb_count

def check_wh_words(comment):
    wh_pattern = re.compile(r'\/(WDT|WP|WP\$|WRB)( |$)')
    wh_count = len(wh_pattern.findall(comment))

    return wh_count

def check_slang(comment):
    tokens = comment.split()
    slang_count = sum(token.split('/')[0] in SLANG for token in tokens)

    return slang_count

def calc_avg_sent_len(comment):
    sentence_end_pattern = re.compile(r'\n( |$)')
    sentence_count = len(sentence_end_pattern.findall(comment))

    if sentence_count == 0:
        return 0
    else:
        word_pattern = re.compile(r'\b\w+/\w+\b')
        word_count = len(word_pattern.findall(comment))
    
    avg_sent_len = word_count / sentence_count
    return avg_sent_len

def calc_avg_token_len(comment, avg_sent_len, multi_punc_count):
    word_pattern = re.compile(r"[^\s\w]*\w\S*\/")
    words = word_pattern.findall(comment)
    correct_words = [word.rstrip('/') for word in words]

    total_letters = 0
    for word in correct_words:
        total_letters += len(word)
    
    # Add in the conditional for comparing the multi-punctuation stuff to the number of words
    if avg_sent_len - multi_punc_count == 0:
        return 0
    else:
        avg_token_len = total_letters / (avg_sent_len - multi_punc_count)

    return avg_token_len

def calc_num_sent(comment):
    sentence_end_pattern = re.compile(r'\n( |$)')
    sentence_count = len(sentence_end_pattern.findall(comment))

    return sentence_count

# Helper function to extract the AoA (100-700), IMG, and FAM features for the different words in the BGL csv
def extract_bgl_norm_features(csv_file_path):
    bgl_norm_features = {}

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        
        # Skip the header row 
        next(csv_reader, None)
        
        # Iterate over each row in the CSV
        for row in csv_reader:
            word = row[1]  # WORD column
            aoa_100_700 = float(row[3]) if row[3] != '' else 0  # AoA (100-700) column
            img = float(row[4]) if row[4] != '' else 0  # IMG column
            fam = float(row[5]) if row[5] != '' else 0  # FAM column

            if row[3] != '' or row[4] != '' or row[5] != '':
                bgl_norm_features[word] = [aoa_100_700, img, fam]
        
    return bgl_norm_features

def calculate_bgl_metrics(comment, bgl_norm_features):
    tokens = re.sub(r"\/\S+", "", comment).split()
    aoa_100_700_vals = []
    img_vals = []
    fam_vals = []
    for word in tokens:
        if word in bgl_norm_features.keys():
            aoa_100_700_vals.append(bgl_norm_features[word][0])
            img_vals.append(bgl_norm_features[word][1])
            fam_vals.append(bgl_norm_features[word][2])
    
    avg_aoa = np.mean(aoa_100_700_vals) if aoa_100_700_vals else 0
    avg_img = np.mean(img_vals) if img_vals else 0
    avg_fam = np.mean(fam_vals) if fam_vals else 0
    std_aoa = np.std(aoa_100_700_vals) if aoa_100_700_vals else 0
    std_img = np.std(img_vals) if img_vals else 0
    std_fam = np.std(fam_vals) if fam_vals else 0

    return avg_aoa, avg_img, avg_fam, std_aoa, std_img, std_fam

# Helper function to extract the V.Mean.Sum, A.Mean.Sum, and D.Mean.Sum features for the different words in the BGL csv
def extract_warr_norm_features(csv_file_path):
    warr_norm_features = {}

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        
        # Skip the header row 
        next(csv_reader, None)
        # Iterate over each row in the CSV
        for row in csv_reader:
            word = row[1]  # WORD column
            v_mean_sum = float(row[2]) if row[2] != '' else 0  # AoA (100-700) column
            a_mean_sum = float(row[5]) if row[5] != '' else 0  # IMG column
            d_mean_sum = float(row[8]) if row[8] != '' else 0  # FAM column

            if row[2] != '' or row[5] != '' or row[8] != '':
                warr_norm_features[word] = [v_mean_sum, a_mean_sum, d_mean_sum]

    return warr_norm_features

def calculate_warr_metrics(comment, warr_norm_features):
    tokens = re.sub(r"\/\S+", "", comment).split()
    vms_vals = []
    ams_vals = []
    dms_vals = []
    for word in tokens:
        if word in warr_norm_features.keys():
            vms_vals.append(warr_norm_features[word][0])
            ams_vals.append(warr_norm_features[word][1])
            dms_vals.append(warr_norm_features[word][2])

    avg_vms = np.mean(vms_vals) if vms_vals else 0
    avg_ams = np.mean(ams_vals) if ams_vals else 0
    avg_dms = np.mean(dms_vals) if dms_vals else 0
    std_vms = np.std(vms_vals) if vms_vals else 0
    std_ams = np.std(ams_vals) if ams_vals else 0
    std_dms = np.std(dms_vals) if dms_vals else 0

    return avg_vms, avg_ams, avg_dms, std_vms, std_ams, std_dms

# This function extracts features 1-29
def extract1(comment):
    """ 
    This function extracts features from a single comment.

    Parameters:
    - comment: string, the body of a comment (after preprocessing).

    Returns:
    - feats: NumPy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here).
    """    

    features = np.zeros(29)
    # TODO: Extract features that rely on capitalization.
    # 1. Number of tokens in uppercase
    features[0], word_dict = check_for_uppercase(comment)

    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    for uppercase, lowercase in word_dict.items():
        comment = comment.replace(uppercase, lowercase)

    # TODO: Extract features that do not rely on capitalization.
    # 2. Number of first-person pronouns 
    features[1] = check_first_person(comment)

    # 3. Number of second-person pronouns 
    features[2] = check_second_person(comment)

    # 4. Number of third-person pronouns 
    features[3] = check_third_person(comment)

    # 5. Number of coordinating conjunctions 
    features[4] = check_coord_conjunc(comment)

    # 6. Number of past-tense verbs 
    features[5] = check_past_tense(comment)

    # 7. Number of future-tense verbs 
    features[6] = check_future_tense(comment)

    # 8. Number of Commas 
    features[7] = check_commas(comment)

    # 9. Number of multi-character punctuation tokens 
    features[8] = check_multi_punctuation(comment)

    # 10. Number of common nouns 
    features[9] = check_common_nouns(comment)

    # 11. Number of proper nouns 
    features[10] = check_proper_nouns(comment)

    # 12. Number of adverbs 
    features[11] = check_adverbs(comment)

    # 13. Number of wh- words 
    features[12] = check_wh_words(comment)

    # 14. Number of slang acronyms 
    features[13] = check_slang(comment)

    # 15. Average length of sentences in tokens 
    features[14] = calc_avg_sent_len(comment)

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    features[15] = calc_avg_token_len(comment, features[14], features[8])

    # 17. Number of sentences 
    features[16] = calc_num_sent(comment)

    # Get the bgl_norm_features
    # CHANGE THE PATH DIRECTORY TO THE NORMS PATH
    bgl_norm_features = extract_bgl_norm_features('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv')

    # 18. Different metrics from Bristol, Gilhooly, and Logie norms (STD and AVG for AoA (100-700), IMG, FAM)
    features[17], features[18], features[19], features[20], features[21], features[22] =  calculate_bgl_metrics(comment, bgl_norm_features)

    # Get the warr_norm_features
    # CHANGE THE PATH DIRECTORY TO THE NORMS PATH
    warr_norm_features = extract_warr_norm_features('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv')

    # 24. Different metrics from Warringer norms (STD and AVG of V.Mean.Sum, A.Mean.Sum, D.Mean.Sum)
    features[23], features[24], features[25], features[26], features[27], features[28] = calculate_warr_metrics(comment, warr_norm_features)

    return features


# This function extracts features 30-173
def extract2(feats, comment_class, comment_id):
    # global global_poli_features
    """ This function adds features 30-173 for a single comment.

    Parameters:
    - feats: np.array of length 173.
    - comment_class: str in {"Alt", "Center", "Left", "Right"}.
    - comment_id: int indicating the id of a comment.

    Returns:
    - feats: NumPy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    """

    # Load the features for the specific class and id into the feats np.ndarray
    with open(f"/u/cs401/A1/feats/{comment_class}_IDs.txt", "r") as f:
        ids = [line.strip() for line in f.readlines()]
    feat = np.load(f"/u/cs401/A1/feats/{comment_class}_feats.dat.npy")
    idx = ids.index(str(comment_id))

    feat = feat[idx, :].reshape(-1)
    return feat

def main(args):
    # Declare necessary global variables here. 
    category_int = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    print(len(data))
    for i in range(len(data)):
        print(i)
        feats[i][0:29] = extract1(data[i]['body'])
        feats[i][29:173] = extract2(feats[i], data[i]['cat'], data[i]['id'])
        feats[i][173] = category_int[data[i]['cat']]

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Specify the output file.", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1.", required=True)
    parser.add_argument("-p", "--a1-dir", help="Path to csc401 A1 directory. By default it is set to the teach.cs directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

