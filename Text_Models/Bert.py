"""
Import necessary Python libraries:
- Import NumPy for numerical computations.
- Import the time library for measuring code execution time.
- Import the psutil library for monitoring system resources.
- Import the profile function from thop for model profiling.
- Import torch for PyTorch deep learning.
"""
import numpy as np
import time
import psutil
from thop import profile
import torch

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_PyTorch_MaskedLM_bert-base-uncased")

"""
Load the BERT model for masked language modeling (MLM):
- Import the AutoTokenizer and BertForMaskedLM classes from the transformers library.
- Initialize the BERT tokenizer and model:
  - Initialize the BERT tokenizer for the "bert-base-uncased" model.
  - Initialize the BERT model for masked language modeling.
"""
from transformers import AutoTokenizer, BertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

"""
Tokenize the input sentence with a masked word:
- Tokenize the input sentence "The capital of France is [MASK]." using the pre-trained BERT tokenizer.
- Return the tokenized input as PyTorch tensors.
"""
inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

"""
Generate logits for the masked word prediction:
- Use a with torch.no_grad() block to disable gradient calculations because we are not training the model.
- The BERT model takes the tokenized input (inputs) and computes the logits for the masked word prediction.
"""
with torch.no_grad():
    logits = model(**inputs).logits

"""
Retrieve the index of the predicted masked word:
- Find the index of the [MASK] token in the tokenized input.
"""
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

"""
Find the predicted token and decode it:
- Determine the predicted token by selecting the token with the highest probability at the [MASK] token's index.
- Decode the predicted token using the tokenizer and print it.
"""
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id.item())
print("Predicted Token:", predicted_token)

"""
Prepare labels for MLM by masking non-[MASK] tokens:
- Tokenize the correct sentence "The capital of France is Paris." using the same tokenizer.
- Mask all tokens that are not [MASK] tokens by replacing them with -100.
"""
labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

"""
Measure latency for MLM inference 100 times:
- Record the start time, perform MLM inference using the BERT model with the labels prepared earlier,
- record the end time, and calculate the time taken for each inference.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs, labels=labels)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate the loss for masked language modeling (MLM):
- The loss is extracted from the outputs object returned by the BERT model.
- It represents how well the model is performing on the MLM task.
- The loss is rounded to two decimal places for clarity.
"""
print("Loss:", round(outputs.loss.item(), 2))

"""
Monitor CPU and memory utilization of the system:
- psutil.cpu_percent() provides the current CPU utilization as a percentage.
- psutil.virtual_memory().percent provides the current memory utilization as a percentage.
- The obtained values are then printed to the console, giving insights into the system's resource usage during the script's execution.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Modify inputs for thop profiling:
- Before performing model profiling using the thop library, the inputs are modified to match the format expected by thop.
- The input tensor for thop profiling is created as a tuple containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of Giga-Operations (GOPS) required for the BERT model:
- Use the thop.profile function to profile the model, passing in the modified thop_inputs.
- The result includes the number of FLOPs (floating-point operations) and parameters in the model.
- The FLOPs are then converted to GOPs by dividing by 1 billion.
- The result is printed to the console, providing insights into the computational complexity of the model.
"""
flops, params = profile(model, inputs=thop_inputs, verbose=False)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_PyTorch_NextSentencePrediction_bert-base-uncased")

"""
Load the BERT model for next sentence prediction (NSP):
- Import the AutoTokenizer and BertForNextSentencePrediction classes from the transformers library.
- Initialize the BERT tokenizer and NSP model:
  - Initialize the BERT tokenizer for the "bert-base-uncased" model.
  - Initialize the BERT model for next sentence prediction (NSP).
"""
from transformers import AutoTokenizer, BertForNextSentencePrediction

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

"""
Define a prompt and the next sentence:
- Define a prompt sentence and the next sentence that you want to check for relevance.
"""
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."

"""
Tokenize and encode the prompt and next sentence:
- Use the BERT tokenizer to tokenize and encode the sentences.
- Return the results as PyTorch tensors with return_tensors="pt".
"""
encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

"""
Measure latency for NSP inference 100 times:
- Run the NSP inference 100 times in a loop to gather performance data.
- Record the start and end times for each inference, calculate the execution time in milliseconds,
  and append it to the total_time list.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    # Perform NSP inference using the BERT model.
    outputs = model(**encoding, labels=torch.LongTensor([1]))
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Extract NSP logits and check if the next sentence is predicted correctly:
- Extract the NSP logits from the model's output, representing confidence scores for related or unrelated sentences.
- Check if the predicted next sentence is related to the prompt (True if related, False if unrelated).
"""
logits = outputs.logits
predicted_next_sentence = logits[0, 0] < logits[0, 1]
print("Predicted Next Sentence:", predicted_next_sentence)

"""
Calculate the probability of being the next sentence (0 is False, 1 is True):
- Calculate the probability of the next sentence being true (1) based on the computed logits using softmax.
"""
probability_next_sentence = torch.softmax(logits, dim=1)[0, 1]
print("Probability of Being the Next Sentence:", probability_next_sentence.item())

"""
Monitor CPU and memory utilization of the system:
- Use the psutil library to monitor and print the current CPU and memory utilization percentages.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Modify inputs for THOP (The Hardware-Aware Library for PyTorch) profiling:
- Group the input tensors for THOP profiling into a tuple named thop_inputs.
"""
thop_inputs = (encoding["input_ids"], encoding["attention_mask"])

"""
Calculate the number of GOPs (Giga-Operations) required for the BERT model using THOP:
- Use the profile function from THOP to profile the model, passing in the modified thop_inputs.
- Calculate the number of FLOPs (floating-point operations) and convert them to GOPs by dividing by 1 billion.
- Print the result, providing insights into the computational complexity of the model.
"""
flops, params = profile(model, inputs=thop_inputs, verbose=False)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_PyTorch_SequenceClassification-single-label-classification_textattack/bert-base-uncased-yelp-polarity")

"""
Load a fine-tuned BERT model for single-label text classification:
- Import the AutoTokenizer and BertForSequenceClassification classes from the transformers library.
- Initialize the BERT tokenizer and sequence classification model:
  - Initialize the BERT tokenizer for the "textattack/bert-base-uncased-yelp-polarity" model.
  - Initialize the BERT model for single-label text classification.
"""
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

"""
Tokenize and encode an input sentence:
- Tokenize and encode the input sentence "Hello, my dog is cute" using the BERT tokenizer.
- Return the results as PyTorch tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

"""
Generate logits for the masked word prediction:
- Generate logits (raw prediction scores) for the input sentence by feeding it through the BERT model.
- Use a 'with torch.no_grad()' context to disable gradient computations during inference.
"""
with torch.no_grad():
    logits = model(**inputs).logits

"""
Predict the class for the input sentence:
- Predict the class for the input sentence based on the logits obtained.
- Select the class with the highest probability and print the predicted class.
"""
predicted_class_id = logits.argmax().item()
predicted_class = model.config.id2label[predicted_class_id]
print("Predicted Class:", predicted_class)

"""
Initialize a BERT model for single-label text classification:
- Initialize a new instance of the BERT model for single-label text classification.
- Specify the number of labels based on the model's configuration.
"""
num_labels = len(model.config.id2label)
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=num_labels)

"""
Define labels for the input sentence:
- Define ground truth labels for the input sentence (in this case, assigning the label "1").
"""
labels = torch.tensor([1])

"""
Measure latency for single-label classification inference 100 times:
- Measure the inference latency of the BERT model for single-label text classification.
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs, labels=labels)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate the loss for single-label classification:
- Calculate the loss for single-label text classification using the BERT model.
"""
loss = model(**inputs, labels=labels).loss
print("Loss:", round(loss.item(), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Modify inputs for thop profiling:
- Prepare the input data in a format suitable for profiling using the THOP library.
- Create a tuple named thop_inputs containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of GOPs using thop:
- Use the THOP library to profile the BERT model and calculate the number of GOPs required for inference.
"""
flops, params = profile(model, inputs=thop_inputs, verbose=False)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_PyTorch_SequenceClassification-multi-label-classification_textattack/bert-base-uncased-yelp-polarity")

"""
Load a fine-tuned BERT model for multi-label text classification:
- Import the AutoTokenizer and BertForSequenceClassification classes from the transformers library.
- Initialize the BERT tokenizer and sequence classification model:
  - Initialize the BERT tokenizer for the "textattack/bert-base-uncased-yelp-polarity" model.
  - Initialize the BERT model for multi-label text classification.
"""
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", problem_type="multi_label_classification")

"""
Tokenize and encode an input sentence:
- Tokenize and encode the input sentence "Hello, my dog is cute" using the BERT tokenizer.
- Return the results as PyTorch tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

"""
Generate logits for the masked word prediction:
- Generate logits (raw prediction scores) for the input sentence by feeding it through the BERT model.
- Use a 'with torch.no_grad()' context to disable gradient computations during inference.
"""
with torch.no_grad():
    logits = model(**inputs).logits

"""
Predict the class for the input sentence:
- Predict multiple classes by applying a threshold of 0.5 to the sigmoid activation of the logits.
- Print the predicted class IDs.
"""
predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
print("Predicted Class:", predicted_class_ids)

"""
Initialize a BERT model for multi-label text classification:
- Initialize a new instance of the BERT model for multi-label text classification.
- Specify the number of labels based on the model's configuration.
"""
num_labels = len(model.config.id2label)
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity", num_labels=num_labels, problem_type="multi_label_classification"
)

"""
Define labels for the input sentence:
- Define ground truth labels for the input sentence based on the predicted class IDs.
- Create a one-hot encoding for the predicted classes and sum them to form the multi-label labels.
"""
labels = torch.sum(
    torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
).to(torch.float)

"""
Measure latency for multi-label classification inference 100 times:
- Measure the inference latency of the BERT model for multi-label text classification.
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs, labels=labels)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate the loss for multi-label classification:
- Calculate the loss for multi-label text classification using the BERT model.
"""
loss = model(**inputs, labels=labels).loss
print("Loss:", round(loss.item(), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Modify inputs for thop profiling:
- Prepare the input data in a format suitable for profiling using the THOP library.
- Create a tuple named thop_inputs containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of GOPs using thop:
- Use the THOP library to profile the BERT model and calculate the number of GOPs required for inference.
"""
flops, params = profile(model, inputs=thop_inputs, verbose=False)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_PyTorch_TokenClassification_dbmdz/bert-large-cased-finetuned-conll03-english")

"""
Load a BERT model for token classification:
- Import the AutoTokenizer and BertForTokenClassification classes from the transformers library.
- Initialize the BERT tokenizer and token classification model:
  - Initialize the BERT tokenizer for the "dbmdz/bert-large-cased-finetuned-conll03-english" model.
  - Initialize the BERT model for token classification.
"""
from transformers import AutoTokenizer, BertForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

"""
Tokenize and encode an input sentence without special tokens:
- Tokenize and encode the input sentence "HuggingFace is a company based in Paris and New York"
  without adding special tokens (e.g., [CLS] or [SEP]).
- Return the results as PyTorch tensors.
"""
inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
)

"""
Measure latency for token classification inference 100 times:
- Measure the inference latency of the BERT model for token classification.
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    logits = model(**inputs).logits
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Predict token classes for the input sentence:
- Predict the token classes for the input sentence.
- Select the class with the highest probability (argmax) for each token.
- Note that tokens are classified rather than input words, which means that
  there might be more predicted token classes than words.
- Multiple token classes might account for the same word.
- Print the predicted token classes.
"""
predicted_token_class_ids = logits.argmax(-1)
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
print("Predicted Token Classes:", predicted_tokens_classes)

"""
Calculate the loss for token classification:
- Calculate the loss for token classification using the BERT model.
"""
labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss
print("Loss:", round(loss.item(), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Modify inputs for thop profiling:
- Prepare the input data in a format suitable for profiling using the THOP library.
- Create a tuple named thop_inputs containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of GOPs using thop:
- Use the THOP library to profile the BERT model and calculate the number of GOPs required for inference.
"""
flops, params = profile(model, inputs=thop_inputs, verbose=False)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_PyTorch_QuestionAnswering_deepset/bert-base-cased-squad2")

"""
Load a BERT model for question answering:
- Import the AutoTokenizer and BertForQuestionAnswering classes from the transformers library.
- Initialize the BERT tokenizer and question answering model:
  - Initialize the BERT tokenizer for the "deepset/bert-base-cased-squad2" model.
  - Initialize the BERT model for question answering.
"""
from transformers import AutoTokenizer, BertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

"""
Define a question and a text for question answering:
- Define the question and the text passage for question answering.
"""
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

"""
Tokenize and encode the question and text:
- Tokenize and encode the question and text using the BERT tokenizer.
- Return the results as PyTorch tensors.
"""
inputs = tokenizer(question, text, return_tensors="pt")

"""
Generate logits for the masked word prediction:
- Generate logits for predicting the answer span within the given text.
- The BERT model provides logits for both the start and end positions of the answer span.
"""
with torch.no_grad():
    outputs = model(**inputs)

"""
Extract the predicted answer span and decode it:
- Extract the predicted answer span based on positions with the highest logits.
- Decode the answer span from the tokenized format to a human-readable string.
"""
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
predicted_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
print("Predicted answer:", predicted_answer)

"""
Define the target answer span:
- Define the target answer span indices for evaluation (ground truth answer span).
"""
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])

"""
Measure latency for question answering inference 100 times:
- Measure the inference latency of the BERT model for question answering.
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate the loss for question answering:
- Calculate the loss for question answering using the BERT model.
"""
loss = outputs.loss
print("Loss:", round(loss.item(), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Modify inputs for thop profiling:
- Prepare the input data in a format suitable for profiling using the THOP library.
- Create a tuple named thop_inputs containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of GOPs using thop:
- Use the THOP library to profile the BERT model and calculate the number of GOPs required for inference.
"""
flops, params = profile(model, inputs=thop_inputs, verbose=False)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Import the TensorFlow library for deep learning tasks.
"""
import tensorflow as tf

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_TensorFlow_MaskedLM_bert-base-uncased")

"""
Load a BERT model for masked language modeling (MLM) in TensorFlow:
- Import the required classes for BERT MLM from Hugging Face Transformers.
- Initialize the BERT tokenizer and MLM model in TensorFlow.
"""
from transformers import AutoTokenizer, TFBertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForMaskedLM.from_pretrained("bert-base-uncased")

"""
Tokenize the input sentence with a masked word:
- Tokenize and encode the input sentence with a masked word using the BERT tokenizer.
- Return the results as TensorFlow tensors.
"""
inputs = tokenizer("The capital of France is [MASK].", return_tensors="tf")

"""
Generate logits for the masked word prediction:
- Generate logits for predicting the masked word in the input sentence.
- The BERT model provides logits for the masked token.
"""
logits = model(**inputs).logits

"""
Retrieve the index of the predicted masked word:
- Find the index of the masked token in the input.
- Select the logits corresponding to the masked token.
"""
mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

"""
Find the predicted token and decode it:
- Determine the predicted token ID with the highest probability (argmax) based on the logits.
- Decode the predicted token from its ID to a human-readable string.
"""
predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print("Predicted Token:", predicted_token)

"""
Prepare labels for MLM by masking non-[MASK] tokens:
- Prepare labels for masked language modeling (MLM) by masking non-[MASK] tokens.
"""
labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

"""
Measure latency for MLM inference 100 times in TensorFlow:
- Measure the inference latency of the BERT model for masked language modeling (MLM).
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs, labels=labels)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate the loss for MLM in TensorFlow:
- Calculate and print the loss for masked language modeling (MLM).
"""
print("Loss:", round(float(outputs.loss), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate FLOPs manually based on the model architecture:
- Calculate the number of Giga-Operations (GOPs) required for one forward pass of the BERT model.
- This estimation is based on the model's architecture and configuration.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_TensorFlow_NextSentencePrediction_bert-base-uncased")

"""
Load a BERT model for next sentence prediction (NSP) in TensorFlow:
- Import the required classes for BERT NSP from Hugging Face Transformers.
- Initialize the BERT tokenizer and NSP model in TensorFlow.
"""
from transformers import AutoTokenizer, TFBertForNextSentencePrediction

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

"""
Define a prompt and the next sentence:
- Define a prompt and the next sentence for NSP.
"""
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."

"""
Tokenize and encode the prompt and next sentence:
- Tokenize and encode the input prompt and next sentence using the BERT tokenizer.
- Return the results as TensorFlow tensors.
"""
encoding = tokenizer(prompt, next_sentence, return_tensors="tf")

"""
Measure latency for NSP inference 100 times in TensorFlow:
- Measure the inference latency of the BERT model for next sentence prediction (NSP).
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    logits = model(encoding["input_ids"], token_type_ids=encoding["token_type_ids"])[0]
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Extract NSP logits and check if the next sentence is predicted correctly:
- Extract the NSP logits from the model's outputs and check if the next sentence is predicted correctly.
"""
predicted_next_sentence = logits[0][0] < logits[0][1]  
print("Predicted Next Sentence:", predicted_next_sentence)

"""
Calculate the probability of being the next sentence (0 is False, 1 is True):
- Calculate the probability of the next sentence being true (1) based on the softmax probabilities of the model's logits.
"""
probability_next_sentence = tf.nn.softmax(logits, axis=1)[0, 1]
print("Probability of Being the Next Sentence:", probability_next_sentence.numpy())

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate an approximate number of GOPs based on the model architecture:
- Calculate an approximate number of Giga-Operations (GOPs) required for one forward pass of the BERT model.
- This estimation is based on the model's architecture and configuration.
"""
num_attention_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
sequence_length = encoding["input_ids"].shape[-1]
num_hidden_layers = model.config.num_hidden_layers

# Number of FLOPs for one forward pass (approximate)
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_hidden_layers
    * num_attention_heads
    * hidden_size
    * sequence_length
    * (hidden_size + sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs (Approximate): {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_TensorFlow_SequenceClassification_ydshieh/bert-base-uncased-yelp-polarity")

"""
Load a fine-tuned BERT model for single-label text classification in TensorFlow:
- Import the required classes for BERT sequence classification from Hugging Face Transformers.
- Initialize the BERT tokenizer and sequence classification model in TensorFlow.
"""
from transformers import AutoTokenizer, TFBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")
model = TFBertForSequenceClassification.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity")

"""
Tokenize and encode an input sentence in TensorFlow:
- Tokenize and encode the input sentence "Hello, my dog is cute" using the BERT tokenizer.
- Return the results as TensorFlow tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

"""
Generate logits for the text classification task:
- Generate logits for the input sentence using the BERT model.
"""
logits = model(**inputs).logits

"""
Predict the class for the input sentence:
- Find the class with the highest probability (argmax) based on the logits.
- Map the class ID to a human-readable label using the model's configuration.
"""
predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
predicted_class = model.config.id2label[predicted_class_id]
print("Predicted Class:", predicted_class)

"""
Initialize a BERT model for multi-label text classification in TensorFlow:
- To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`.
"""
num_labels = len(model.config.id2label)
model = TFBertForSequenceClassification.from_pretrained("ydshieh/bert-base-uncased-yelp-polarity", num_labels=num_labels)

"""
Define labels for the input sentence in TensorFlow:
- Define labels for the input sentence; in this case, it's a constant value of 1, representing a label.
"""
labels = tf.constant(1)

"""
Measure latency for sequence classification inference 100 times in TensorFlow:
- Measure the inference latency of the BERT model for sequence classification.
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs, labels=labels)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate the loss for multi-label text classification in TensorFlow:
- Calculate and print the loss for multi-label text classification from the model's outputs.
"""
loss = model(**inputs, labels=labels).loss
print("Loss:", round(float(loss), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate FLOPs manually based on the model architecture:
- Calculate an approximate number of Giga-Operations (GOPs) required for one forward pass of the BERT model.
- This estimation is based on the model's architecture and configuration.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_TensorFlow_TokenClassification_dbmdz/bert-large-cased-finetuned-conll03-english")

"""
Load a BERT model for token classification in TensorFlow:
- Import the required classes for BERT token classification from Hugging Face Transformers.
- Initialize the BERT tokenizer and token classification model in TensorFlow.
"""
from transformers import AutoTokenizer, TFBertForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = TFBertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

"""
Define an input sentence for token classification in TensorFlow:
- Tokenize and encode the input sentence for token classification.
- Special tokens are not added in this case.
"""
inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
)

"""
Measure latency for token classification inference 100 times in TensorFlow:
- Measure the inference latency of the BERT model for token classification.
- Run inference 100 times in a loop and record the time taken for each inference.
- Calculate the mean latency time and print it.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    logits = model(**inputs).logits
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Extract predicted token classes for each token in the input sentence:
- Find the class with the highest probability (argmax) for each token.
- Map the class IDs to human-readable labels using the model's configuration.
"""
predicted_token_class_ids = tf.math.argmax(logits, axis=-1)
predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
print("Predicted Token Classes:", predicted_tokens_classes)

"""
Calculate the loss for token classification in TensorFlow:
- Calculate and print the loss for token classification from the model's outputs.
"""
labels = predicted_token_class_ids
loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
print("Loss:", round(float(loss), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate FLOPs manually based on the model architecture:
- Calculate an approximate number of Giga-Operations (GOPs) required for one forward pass of the BERT model.
- This estimation is based on the model's architecture and configuration.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_TensorFlow_QuestionAnswering_ydshieh/bert-base-cased-squad2")

"""
Load a BERT model for question answering in TensorFlow:
- Import the required classes for BERT question answering from Hugging Face Transformers.
- Initialize the BERT tokenizer and question answering model in TensorFlow.
"""
from transformers import AutoTokenizer, TFBertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("ydshieh/bert-base-cased-squad2")
model = TFBertForQuestionAnswering.from_pretrained("ydshieh/bert-base-cased-squad2")

"""
Define a question and context for question answering in TensorFlow:
- Define the question and context that you want to use for question answering.
"""
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

"""
Tokenize and encode the question and context for question answering:
- Tokenize and encode the question and context using the BERT tokenizer.
- Use the BERT model to obtain outputs, including start and end logits for the answer span.
"""
inputs = tokenizer(question, text, return_tensors="tf")
outputs = model(**inputs)

"""
Extract the predicted answer span from the model's outputs:
- Extract the predicted answer span based on the start and end logits.
"""
answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
predicted_answer = tokenizer.decode(predict_answer_tokens)
print("Predicted answer:", predicted_answer)

"""
Define the target answer span for question answering:
- Define the target answer span indices for evaluation.
"""
target_start_index = tf.constant([14])
target_end_index = tf.constant([15])

"""
Measure latency for question answering inference 100 times in TensorFlow:
- Measure the inference latency of the BERT model for question answering.
- Run inference 100 times and calculate the mean time.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    # Include the target answer span for evaluation
    outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate and print the loss for question answering:
- Calculate the loss for question answering from the model's outputs.
"""
loss = tf.math.reduce_mean(outputs.loss)
print("Loss:", round(float(loss), 2))

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate FLOPs manually based on the model architecture:
- Calculate an approximate number of Giga-Operations (GOPs) required for one forward pass of the BERT model.
- This estimation is based on the model's architecture and configuration.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Import the JAX library, which is used for numerical computing, particularly suitable for hardware acceleration.
"""
import jax
import jax.numpy as jnp

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_Flax_CausalLM_bert-base-uncased")

"""
Load a BERT model for causal language modeling (CLM) in Flax:
- Import the required classes for BERT causal language modeling from Hugging Face Transformers.
- Initialize the BERT tokenizer and CLM model in Flax.
"""
from transformers import AutoTokenizer, FlaxBertForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForCausalLM.from_pretrained("bert-base-uncased")

"""
Tokenize and encode an input sentence for CLM in Flax:
- Tokenize and encode the input sentence using the BERT tokenizer.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="np")

"""
Measure latency for CLM inference 100 times in Flax:
- Measure the inference latency of the BERT model for causal language modeling (CLM).
- Run inference 100 times and calculate the mean time.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Retrieve logits for the next token prediction in CLM:
- Retrieve the logits for the next token prediction in the causal language modeling (CLM) model.
- Specifically, look at the logits for the last token.
"""
print("Logits for Next Token:", outputs.logits[:, -1])

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate FLOPs manually based on the model architecture:
- Calculate an approximate number of Giga-Operations (GOPs) required for one forward pass of the BERT model.
- This estimation is based on the model's architecture and configuration.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_Flax_MaskedLM_bert-base-uncased")

"""
Load a BERT model for masked language modeling (MLM) in Flax:
- Import the required classes for BERT masked language modeling from Hugging Face Transformers.
- Initialize the BERT tokenizer and MLM model in Flax.
"""
from transformers import AutoTokenizer, FlaxBertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForMaskedLM.from_pretrained("bert-base-uncased")

"""
Tokenize and encode a sentence with a masked word for MLM in Flax:
- Tokenize and encode the input sentence "The capital of France is [MASK]." with a masked word using the BERT tokenizer.
"""
inputs = tokenizer("The capital of France is [MASK].", return_tensors="jax")

"""
Measure latency for MLM inference 100 times in Flax:
- Measure the inference latency of the BERT model for masked language modeling (MLM).
- Run inference 100 times and calculate the mean time.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Retrieve logits for the masked word prediction in MLM:
- Retrieve the logits for the masked word prediction in the MLM model.
"""
print("Logits for Next Token:", outputs.logits)

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate FLOPs manually based on the model architecture:
- Calculate an approximate number of Giga-Operations (GOPs) required for one forward pass of the BERT model.
- This estimation is based on the model's architecture and configuration.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_Flax_NextSentencePrediction_bert-base-uncased")

"""
Load a BERT model for next sentence prediction (NSP) in Flax:
- Import the required classes for BERT next sentence prediction from Hugging Face Transformers.
- Initialize the BERT tokenizer and NSP model in Flax.
"""
from transformers import AutoTokenizer, FlaxBertForNextSentencePrediction

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

"""
Define a prompt and the next sentence for NSP in Flax:
- Define a prompt and the following sentence for the next sentence prediction task.
"""
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."

"""
Tokenize and encode the prompt and next sentence for NSP in Flax:
- Tokenize and encode the provided prompt and next sentence using the BERT tokenizer and store the encodings in the 'encoding' variable.
"""
encoding = tokenizer(prompt, next_sentence, return_tensors="jax")

"""
Measure latency for NSP inference 100 times in Flax:
- Measure the inference latency of the BERT model for next sentence prediction (NSP).
- Run inference 100 times and calculate the mean time.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**encoding)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Extract NSP logits and check if the next sentence is predicted correctly:
- Extract the NSP logits from the model's outputs and check if the model correctly predicts the next sentence.
"""
logits = outputs.logits
# Check if the model correctly predicts the next sentence (1 for True, 0 for False)
print("Predicted Next Sentence:", logits[0, 0] < logits[0, 1])
# Calculate the probability of being the next sentence (0 is False, 1 is True)
probability_next_sentence = jax.nn.softmax(logits, axis=1)[0, 1]
print("Probability of Being the Next Sentence:", probability_next_sentence)

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate an approximate number of Giga-Operations (GOPs) based on the model architecture:
- Note: This is an approximate value and may not be precise. Adjust based on the specific BERT variant being used.
"""
num_attention_heads = model.config.num_attention_heads
hidden_size = model.config.hidden_size
sequence_length = encoding["input_ids"].shape[-1]
num_hidden_layers = model.config.num_hidden_layers

# Number of FLOPs for one forward pass (approximate)
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_hidden_layers
    * num_attention_heads
    * hidden_size
    * sequence_length
    * (hidden_size + sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_Flax_SequenceClassification_bert-base-uncased")

"""
Load a BERT model for sequence classification in Flax:
- Import the required classes for BERT sequence classification from Hugging Face Transformers.
- Initialize both the tokenizer and the model using the Flax framework.
"""
from transformers import AutoTokenizer, FlaxBertForSequenceClassification

# Initialize the BERT tokenizer for the "bert-base-uncased" model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Initialize the BERT model for sequence classification using the Flax framework
model = FlaxBertForSequenceClassification.from_pretrained("bert-base-uncased")

"""
Tokenize and encode an input sentence for sequence classification in Flax:
- Tokenize and encode the input sentence using the tokenizer.
- The resulting encodings are stored in the 'inputs' variable.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

"""
Measure latency for sequence classification inference 100 times in Flax:
- Measure the inference latency of the BERT model for sequence classification.
- Run inference 100 times and calculate the mean time.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Retrieve logits for sequence classification:
- This line of code retrieves the logits (raw output scores) produced by the model for sequence classification.
"""
print("Sequence Classification Logits:", outputs.logits)

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate an approximate number of Giga-Operations (GOPs) based on the model architecture:
- Note: This is an approximate value and may not be precise. Adjust based on the specific BERT variant being used.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Print a descriptive message indicating the beginning of the script.
"""
print("Bert_Flax_TokenClassification_bert-base-uncased")

"""
Load a BERT model for token classification in Flax:
- Import the required classes for BERT token classification from Hugging Face Transformers.
- Initialize both the tokenizer and the model using the Flax framework.
"""
from transformers import AutoTokenizer, FlaxBertForTokenClassification

# Initialize the BERT tokenizer for the "bert-base-uncased" model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Initialize the BERT model for token classification using the Flax framework
model = FlaxBertForTokenClassification.from_pretrained("bert-base-uncased")

"""
Tokenize and encode an input sentence for token classification in Flax:
- Tokenize and encode the input sentence using the tokenizer.
- The resulting encodings are stored in the 'inputs' variable.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

"""
Measure latency for token classification inference 100 times in Flax:
- Measure the inference latency of the BERT model for token classification.
- Run inference 100 times and calculate the mean time.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Retrieve logits for token classification:
- This line of code retrieves the logits (raw output scores) produced by the model for token classification.
- These logits can be used to predict the token classes.
"""
print("Logits for Token Classification:", outputs.logits)

"""
Resource Utilization:
- Measure and print CPU and memory utilization using the `psutil` library.
"""
cpu_utilization = psutil.cpu_percent()
memory_utilization = psutil.virtual_memory().percent
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

"""
Calculate an approximate number of Giga-Operations (GOPs) based on the model architecture:
- Note: This is an approximate value and may not be precise. Adjust based on the specific BERT variant being used.
"""
num_hidden_units = model.config.hidden_size
num_attention_heads = model.config.num_attention_heads
num_sequence_length = inputs.input_ids.shape[-1]
num_layers = model.config.num_hidden_layers
flops_per_add_mult = 2  # FLOPs for one addition/multiplication
flops = (
    num_layers
    * num_attention_heads
    * num_hidden_units
    * num_sequence_length
    * (num_hidden_units + num_sequence_length * flops_per_add_mult)
)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")
