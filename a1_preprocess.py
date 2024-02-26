import sys
import argparse
import os
import json
import re
import spacy
import html

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')

def preprocess(comment):
    """ 
    This function preprocesses a single comment.

    Parameters:                                                                      
    - comment: string, the body of a comment.

    Returns:
    - modified_comment: string, the modified comment.
    """
    modified_comment = comment

    # STEP 1
    # TODO: Replace newlines with spaces to handle other whitespace chars.
    modified_comment = re.sub(r"\n{1,}", " ", modified_comment)

    # STEP 2
    # TODO: Remove '[deleted]' or '[removed]' statements.
    if modified_comment == '[deleted]' or modified_comment == '[removed]':
        modified_comment = ''
    else:
        modified_comment = modified_comment.replace('[deleted]', '')
        modified_comment = modified_comment.replace('[removed]', '')
    
    # STEP 3
    # TODO: Unescape HTML.
    modified_comment = modified_comment.strip()
    modified_comment = html.unescape(modified_comment)
    # Sanity Check

    # STEP 4
    # TODO: Remove URLs.
    modified_comment = re.sub(r"(http|www)\S+", "", modified_comment)

    # STEP 5
    # TODO: Remove duplicate spaces.
    modified_comment = re.sub(' +', ' ', modified_comment)

    # STEP 6
    # TODO: Get Spacy document for modified_comment.
    # TODO: Use Spacy document for modified_comment to create a string.
    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" (/tag) after each token.

    utt = nlp(modified_comment)
    final_modified_comment = ''
    for sent in utt.sents:
        for token in sent:
            text_val = token.text
            if token.lemma_.startswith('-') and not token.text.startswith('-'):
                final_modified_comment += token.text + '/' + token.tag_ + ' '
            else:
                final_modified_comment += token.lemma_ + '/' + token.tag_ + ' '
        final_modified_comment += '\n'

    return final_modified_comment


def main(args):
    all_output = []
    student_id = args.ID[0]
    output_file = args.output
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: Select appropriate args.max lines.
            # TODO: Read those lines with something like `j = json.loads(line)`.
            # TODO: Choose to retain fields from those lines that are relevant to you.
            # TODO: Add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...).
            # TODO: Process the body field (j['body']) with preprocess(...) using default for `steps` argument.
            # TODO: Replace the 'body' field with the processed text.
            # TODO: Append the result to 'all_output'.

            start_index = student_id % len(data)
            end_index = start_index + args.max

            if (len(data) - (start_index + 1) < end_index - start_index):
                data = data[start_index:] + data[0:end_index - len(data)]
            else:
                data = data[start_index : end_index]  
            
            for i in range(len(data)):
                line = json.loads(data[i])

                preprocessed_comment = preprocess(line['body'])
                line['body'] = preprocessed_comment
                line['cat'] = file

                processed_line = {'id': line['id'], 'body': line['body'], 'cat': line['cat']}
                all_output.append(processed_line)
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(all_output))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID.')
    parser.add_argument("-o", "--output", help="Specify the output file.", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file.", default=10000)
    parser.add_argument("--a1-dir", help="The directory for A1. This directory should contain the subdir `data`.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
