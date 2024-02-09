import os, json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

ID2LABEL = {0: 'Alt', 1: 'Center', 2: 'Left', 3: 'Right'}
LABEL2ID = {'Alt': 0, 'Center': 1, 'Left': 2, 'Right': 3}

def load_data(data_dir, count_per_label, ID):
    """
    Loads text data from raw JSON files in the specified directory.

    Parameters:
    - data_dir (str): The directory containing raw JSON files.
    -count_per_label (int): number of data per label
    - ID (int): student ID    

    Returns:
    - data_dict (dict): A dictionary containing text and corresponding labels.
        - 'text': List of text bodies extracted from the JSON files.
        - 'labels': List of labels (converted to numerical ints) associated with each text entry.
    """

    data_dict = {'text': [], 'labels': []}
    student_id = ID[0]
    # FIGURE OUT WHAT NEEDS TO BE PUT INTO LABELS
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            full_file = os.path.join(subdir, file)
            print(f"Processing {full_file}")
            with open(full_file) as open_file:
                data = json.load(open_file)
                
            # TODO: select the appropriate start of data using ID, Hint: the same start part of data used in preprocess code.
            # if count_per_label > len(data) use (count_per_label - len(data)) number of data from the beginning of data 
            start_index = student_id % len(data)    
            if (count_per_label > len(data)):
                select_data = data[start_index:count_per_label - len(data)]
            else:
                select_data = data[start_index:start_index + count_per_label]
                           
            for i in range(len(select_data)):
                line = json.loads(select_data[i])
                text = line['body']
                label = LABEL2ID[file]
                data_dict['text'].append(text)
                data_dict['labels'].append(label)
                
    return data_dict

def tokenize_text(text, tokenizer):
    """
    Tokenizes a given text using the provided tokenizer.

    Parameters:
    - text (dict): The sample dict in which the text is to be tokenized.
    - tokenizer (object): The tokenizer object.
    
    Returns:
    - tokenized_text.
    """

    # TODO: Implement the function body
    # Hint: Remember to enable the `truncation` option and set the maximum sequence length (max_length) to 100.

    tokenized_text = tokenizer(text['text'], truncation=True, max_length=100, return_tensors='pt', padding=True)
    # text['attention_mask'] = tokenized_text['attention_mask']
    # text['input_ids'] = tokenized_text['input_ids']

    return tokenized_text

# DONE: I THINK
def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for the model predictions.

    Parameters:
    - eval_pred (tuple): Tuple containing model predictions and true labels.

    Returns:
    - Computed metrics based on model predictions and true labels.
    """
    # TODO: Implement the function body

    predictions, labels = eval_pred
    assert len(predictions) == len(labels), "Arrays must have the same length"
    
    calculated_accuracy = np.mean(np.argmax(predictions, axis=1) == labels.astype(int))

    return {
        "accuracy": calculated_accuracy
    }

def train_model(model, tokenizer, tokenized_dataset, data_collator, training_args):
    """
    Trains the model using the provided Trainer object.

    Parameters:
    - model (object): The model to be trained.
    - tokenizer (object): The tokenizer object.
    - tokenized_dataset (object): Tokenized dataset.
    - data_collator (object): Data collator with padding.
    - training_args (object): Training arguments for the Trainer.
    """

    # TODO: Implement the function body
    # Step 1. You need to first create a `Trainer`. You need to specify the
    #   following parameters:
    #   model, args, train_dataset, eval_dataset, tokenizer, data_collator,
    #   compute_metrics
    # Step 2: Call `trainer.train()` to initiate the training process.
    # Step 3: Evaluate your trained result, and return the evaluation results.
    tokenized_data_splits = tokenized_dataset.train_test_split(test_size=0.2, seed=1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data_splits["train"],
        eval_dataset=tokenized_data_splits["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    evaulation_results = trainer.evaluate()

    return evaulation_results

def main(args):
    """
    Main function that orchestrates the training process.
    """

    if args.toy:
        count_per_label = 10
    else:
        count_per_label = 10000
        assert torch.cuda.is_available(), "The actual training loop must be initiated with a GPU available. Read `Debugging and Using GPUs on teach.cs` on the handout for more information."

    # TODO: Load data and split into train and test.
    # Hint: Call `load_data` with `args.input_dir`, 'args.ID', and `count_per_label`.
    data_dict = load_data(args.input_dir, count_per_label, args.ID)

    # Hint: Call `Dataset.from_dict` first. Then, use the `.train_test_split`
    #   function to create a evaluation set with 20% of the data.
    dataset = Dataset.from_dict(data_dict)
    
    # Use AutoTokenizer to retrieve the tokenizer from POLITICS.
    # Hints:
    # 1. Don't forget to specify `cache_dir='/u/cs401/A1/models'`!
    # 2. Model name: `launch/POLITICS`.
    cache_dir = '/u/cs401/A1/models'
    model_name = 'launch/POLITICS'
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Use AutoModelForSequenceClassification to retireve the model from POLITICS.
    # Hints:
    # 1. Don't forget to specify `cache_dir='/u/cs401/A1/models'`!
    # 2. Model name: `launch/POLITICS`.
    # 3. Use ID2LABEL and LABEL2ID.
    # 4. Don't forget to specify `num_labels`.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir, num_labels=len(ID2LABEL))
    
    # TODO: Set the device to CUDA if available, otherwise to CPU, and move the model to the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # TODO: Use the tokenize_text function to tokenize the dataset. Consider using batched processing for efficiency.
    # Hint: You can call the `Dataset.map` function with option `batched`.
    tokenized_text = dataset.map(lambda x: tokenize_text(x, tokenizer), batched=True)

    # Set the TrainingArguments for training the model.
    training_args = TrainingArguments(
        report_to="none",
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # TODO: Create a data collator with padding using the specified tokenizer.
    # Hint: Use `DataCollatorWithPadding`.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO: Implement the code to train the model using the Trainer object.
    # Hint: Call `train_model` with all the parameters you have just setup.
    evaluation_result = train_model(model, tokenizer, tokenized_text, data_collator, training_args)
    # Save the your evaluation result.
    output_file = f"{args.output_dir}/evaluation_result.txt"

    with open(output_file, "w") as f:
        json.dump(evaluation_result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-i", "--input-dir",
        help="Specify the input directory of data. Defaults to the directory for A1 on cdf.",
        type=Path,
        default=Path('/u/cs401/A1/data'))
    parser.add_argument(
        "-o", "--output-dir",
        help="Specify the directory to write the training results.",
        type=Path,
        default=Path("finetuned_results"))
    parser.add_argument(
        "--cache-dir",
        help="Specify the directory to cache the POLITICS model.",
        type=Path,
        default=Path("/u/cs401/A1/models"))
    parser.add_argument('--toy', action="store_true",
        help="Train the model on the toy dataset.")
    args = parser.parse_args()

    main(args)
