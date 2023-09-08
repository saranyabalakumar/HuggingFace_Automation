"""
Import necessary Python libraries:
- Import the NumPy library for numerical computations.
- Import the time library for measuring code execution time.
- Import the psutil library for monitoring system resources.
- Import the profile function from thop for model profiling.
- Import the PyTorch library for deep learning.
"""
import numpy as np
import time
import psutil
from thop import profile
import torch

"""
Print a descriptive message indicating the beginning of the script.
"""
print("ALBERT_PyTorch_MaskedLM_albert-base-v2")

"""
Load the ALBERT model for masked language modeling (MLM):
- Import the AutoTokenizer and AlbertForMaskedLM classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "albert-base-v2" model.
  - Initialize the ALBERT model itself for masked language modeling.
"""
from transformers import AutoTokenizer, AlbertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AlbertForMaskedLM.from_pretrained("albert-base-v2")

"""
Tokenize the input sentence with a masked word:
- Tokenize the input sentence "The capital of [MASK] is Paris." using the pre-trained ALBERT tokenizer.
- Return the tokenized input as PyTorch tensors.
"""
inputs = tokenizer("The capital of [MASK] is Paris.", return_tensors="pt")

"""
Generate logits for the masked word prediction:
- Use a with torch.no_grad() block to disable gradient calculations because we are not training the model.
- The ALBERT model takes the tokenized input (inputs) and computes the logits for the masked word prediction.
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
- Record the start time, perform MLM inference using the ALBERT model with the labels prepared earlier,
  record the end time, and calculate the time taken for each inference.
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
- The loss is extracted from the outputs object returned by the ALBERT model.
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
Calculate the number of Giga-Operations (GOPS) required for the ALBERT model:
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
print("ALBERT_PyTorch_SequenceClassification-single-label-classification_textattack/albert-base-v2-imdb")

"""
Load the ALBERT model for sequence classification using TextAttack:
- Import the AutoTokenizer and AlbertForSequenceClassification classes from the transformers library.
- Initialize the tokenizer and model:
  - Initialize the ALBERT tokenizer for the "textattack/albert-base-v2-imdb" model.
  - Initialize the ALBERT model for sequence classification.
"""
from transformers import AutoTokenizer, AlbertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")

"""
Tokenize an input text:
- Tokenize the input text "Hello, my dog is cute" using the ALBERT tokenizer.
- Return the tokenized input as PyTorch tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

"""
Generate logits for sequence classification:
- Use a with torch.no_grad() block to disable gradient calculations since we are not training the model.
- The ALBERT model takes the tokenized input (inputs) and computes the logits for sequence classification.
"""
with torch.no_grad():
    logits = model(**inputs).logits

"""
Determine the predicted class:
- Find the class with the highest logit value as the predicted class.
- Map the predicted class ID to the corresponding label using model.config.id2label.
"""
predicted_class_id = logits.argmax().item()
predicted_class = model.config.id2label[predicted_class_id]
print("Predicted Class:", predicted_class)

"""
Modify the model for single-label classification:
- Update the model's configuration to match the number of labels in the dataset.
"""
num_labels = len(model.config.id2label)
model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb", num_labels=num_labels)

"""
Prepare labels for sequence classification:
- Create a tensor with the label ID(s) for the input text (e.g., [1] for a positive sentiment label).
"""
labels = torch.tensor([1])

"""
Measure latency for sequence classification inference 100 times:
- Record the start time, perform inference using the ALBERT model with the labels prepared earlier,
  record the end time, and calculate the time taken for each inference.
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
Calculate the loss for sequence classification:
- The loss is extracted from the outputs object returned by the ALBERT model.
- The loss represents how well the model is performing on the sequence classification task.
- The loss is rounded to two decimal places for clarity.
"""
loss = model(**inputs, labels=labels).loss
print("Loss:", round(loss.item(), 2))

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
Modify inputs for THOP profiling:
- Before performing model profiling using the THOP library, the inputs are modified to match the format expected by THOP.
- The input tensor for THOP profiling is created as a tuple containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of Giga-Operations (GOPS) required for the ALBERT model:
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
print("ALBERT_PyTorch_SequenceClassification-multi-label-classification_textattack/albert-base-v2-imdb")

"""
Load the ALBERT model for sequence classification with multi-label classification problem type using TextAttack:
- Import the AutoTokenizer and AlbertForSequenceClassification classes from the transformers library.
- Initialize the tokenizer and model:
  - Initialize the ALBERT tokenizer for the "textattack/albert-base-v2-imdb" model.
  - Initialize the ALBERT model for sequence classification with multi-label classification problem type.
"""
from transformers import AutoTokenizer, AlbertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb", problem_type="multi_label_classification")

"""
Tokenize an input text:
- Tokenize the input text "Hello, my dog is cute" using the ALBERT tokenizer.
- Return the tokenized input as PyTorch tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

"""
Generate logits for multi-label sequence classification:
- Use a with torch.no_grad() block to disable gradient calculations since we are not training the model.
- The ALBERT model takes the tokenized input (inputs) and computes the logits for multi-label sequence classification.
"""
with torch.no_grad():
    logits = model(**inputs).logits

"""
Determine the predicted class IDs:
- Find class IDs with sigmoid(logits) > 0.5 as the predicted class IDs.
"""
predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]
print("Predicted Class:", predicted_class_ids)

"""
Get the number of labels from the model's configuration.
"""
num_labels = len(model.config.id2label)

"""
Modify the model for multi-label sequence classification:
- Update the model's configuration to match the number of labels in the dataset.
"""
model = AlbertForSequenceClassification.from_pretrained(
    "textattack/albert-base-v2-imdb", num_labels=num_labels, problem_type="multi_label_classification"
)

"""
Prepare labels for multi-label sequence classification:
- Create a tensor with one-hot encoded labels based on predicted class IDs.
"""
labels = torch.sum(
    torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
).to(torch.float)

"""
Measure latency for multi-label sequence classification inference 100 times:
- Record the start time, perform inference using the ALBERT model with the labels prepared earlier,
  record the end time, and calculate the time taken for each inference.
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
Calculate the loss for multi-label sequence classification:
- The loss is extracted from the outputs object returned by the ALBERT model.
- The loss represents how well the model is performing on the multi-label sequence classification task.
- The loss is rounded to two decimal places for clarity.
"""
loss = model(**inputs, labels=labels).loss
print("Loss:", round(loss.item(), 2))

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
Modify inputs for THOP profiling:
- Before performing model profiling using the THOP library, the inputs are modified to match the format expected by THOP.
- The input tensor for THOP profiling is created as a tuple containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of Giga-Operations (GOPS) required for the ALBERT model:
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
print("ALBERT_PyTorch_TokenClassification_albert-base-v2")

"""
Load the ALBERT model for token classification:
- Import the AutoTokenizer and AlbertForTokenClassification classes from the transformers library.
- Initialize the tokenizer and model:
  - Initialize the ALBERT tokenizer for the "albert-base-v2" model.
  - Initialize the ALBERT model for token classification.
"""
from transformers import AutoTokenizer, AlbertForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AlbertForTokenClassification.from_pretrained("albert-base-v2")

"""
Tokenize an input text:
- Tokenize the input text "HuggingFace is a company based in Paris and New York" without adding special tokens.
- Return the tokenized input as PyTorch tensors.
"""
inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
)

"""
Measure latency for token classification inference 100 times:
- Record the start time, perform inference using the ALBERT model,
  record the end time, and calculate the time taken for each inference.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
"""
with torch.no_grad():
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
Determine predicted token class IDs:
- Find class IDs with the highest logits for each token.
"""
predicted_token_class_ids = logits.argmax(-1)

"""
Map predicted token class IDs to label names using the model's configuration.
"""
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
print("Predicted Token Classes:", predicted_tokens_classes)

"""
Prepare labels for token classification:
- Use the predicted token class IDs as labels for the loss calculation.
"""
labels = predicted_token_class_ids

"""
Calculate the loss for token classification:
- The loss is extracted from the outputs object returned by the ALBERT model.
- The loss represents how well the model is performing on the token classification task.
- The loss is rounded to two decimal places for clarity.
"""
loss = model(**inputs, labels=labels).loss
print("Loss:", round(loss.item(), 2))

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
Modify inputs for THOP profiling:
- Before performing model profiling using the THOP library, the inputs are modified to match the format expected by THOP.
- The input tensor for THOP profiling is created as a tuple containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of Giga-Operations (GOPS) required for the ALBERT model:
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
print("ALBERT_PyTorch_QuestionAnswering_twmkn9/albert-base-v2-squad2")

"""
Load the ALBERT model for question answering:
- Import the AutoTokenizer and AlbertForQuestionAnswering classes from the transformers library.
- Initialize the tokenizer and model:
  - Initialize the ALBERT tokenizer for the "twmkn9/albert-base-v2-squad2" model.
  - Initialize the ALBERT model for question answering.
"""
from transformers import AutoTokenizer, AlbertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")
model = AlbertForQuestionAnswering.from_pretrained("twmkn9/albert-base-v2-squad2")

"""
Define a question and a text passage.
"""
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

"""
Tokenize the question and text:
- Return the tokenized inputs as PyTorch tensors.
"""
inputs = tokenizer(question, text, return_tensors="pt")

"""
Perform question answering inference:
- Pass the tokenized inputs to the ALBERT model and obtain the outputs.
"""
with torch.no_grad():
    outputs = model(**inputs)

"""
Determine the start and end indices of the answer span:
- The answer span is determined based on the start_logits and end_logits returned by the model.
"""
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

"""
Extract the predicted answer tokens:
- The answer tokens are extracted based on the start and end indices.
"""
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

"""
Decode the predicted answer tokens to obtain the answer string.
"""
predicted_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
print("Predicted answer:", predicted_answer)

"""
Define target start and end indices for the answer (ground truth):
- In this example, the target is "nice puppet," so the start index is 12 and the end index is 13.
"""
target_start_index = torch.tensor([12])
target_end_index = torch.tensor([13])

"""
Measure latency for question answering inference 100 times:
- Record the start time, perform inference using the ALBERT model,
  record the end time, and calculate the time taken for each inference.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
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
- The loss is extracted from the outputs object returned by the ALBERT model.
- The loss represents how well the model is performing on the question answering task.
- The loss is rounded to two decimal places for clarity.
"""
loss = outputs.loss
print("Loss:", round(loss.item(), 2))

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
Modify inputs for THOP profiling:
- Before performing model profiling using the THOP library, the inputs are modified to match the format expected by THOP.
- The input tensor for THOP profiling is created as a tuple containing input_ids and attention_mask.
"""
thop_inputs = (inputs["input_ids"], inputs["attention_mask"])

"""
Calculate the number of Giga-Operations (GOPS) required for the ALBERT model:
- Use the thop.profile function to profile the model, passing in the modified thop_inputs.
- The result includes the number of FLOPs (floating-point operations) and parameters in the model.
- The FLOPs are then converted to GOPs by dividing by 1 billion.
- The result is printed to the console, providing insights into the computational complexity of the model.
"""
flops, params = profile(model, inputs=thop_inputs, verbose=False)
gops = flops / 1e9  # Convert FLOPs to GOPs
print(f"Number of GOPs: {gops} GOPs")

"""
Import necessary libraries.
"""
import tensorflow as tf

"""
Print a descriptive message indicating the beginning of the script.
"""
print("ALBERT_TensorFlow_MaskedLM_albert-base-v2")

"""
Load the ALBERT model for masked language modeling (MLM):
- Import the AutoTokenizer and TFAlbertForMaskedLM classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "albert-base-v2" model.
  - Initialize the ALBERT model for masked language modeling.
"""
from transformers import AutoTokenizer, TFAlbertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = TFAlbertForMaskedLM.from_pretrained("albert-base-v2")

"""
Tokenize a sentence with a masked token:
- Return the tokenized inputs as TensorFlow tensors.
"""
inputs = tokenizer(f"The capital of [MASK] is Paris.", return_tensors="tf")

"""
Perform masked language modeling inference:
- Calculate the logits for the masked token in the input sentence.
- Logits represent the predicted token probabilities.
"""
logits = model(**inputs).logits

"""
Retrieve the index of the masked token ([MASK]):
- This is done to locate the predicted token later.
"""
mask_token_index = tf.where(inputs.input_ids == tokenizer.mask_token_id)[0][1]

"""
Find the predicted token ID:
- Find the token ID with the highest probability for the masked position.
- Decode the predicted token ID to get the predicted token.
"""
predicted_token_id = tf.math.argmax(logits[0, mask_token_index], axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print("Predicted Token:", predicted_token)

"""
Prepare labels for masked language modeling:
- Tokenize a reference sentence without the [MASK] token.
- Replace masked tokens in labels with -100 (to be ignored in loss calculation).
"""
labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

"""
Measure latency for masked language modeling inference 100 times:
- Record the start time, perform inference using the ALBERT model,
  record the end time, and calculate the time taken for each inference.
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
Calculate the loss for masked language modeling:
- The loss is extracted from the outputs object returned by the ALBERT model.
- The loss represents how well the model is performing on the masked language modeling task.
- The loss is rounded to two decimal places for clarity.
"""
print("Loss:", round(float(outputs.loss), 2))

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
Calculate the number of Giga-Operations (GOPS) required for the ALBERT model:
- The calculations are based on model configuration parameters such as hidden units, attention heads, sequence length, and layers.
- The result is printed to the console, providing insights into the computational complexity of the model.
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
print("ALBERT_TensorFlow_SequenceClassification_vumichien/albert-base-v2-imdb")

"""
Load the ALBERT model for SequenceClassification:
- Import the AutoTokenizer and TFAlbertForSequenceClassification classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "vumichien/albert-base-v2-imdb" model.
  - Initialize the ALBERT model for sequence classification.
"""
from transformers import AutoTokenizer, TFAlbertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("vumichien/albert-base-v2-imdb")
model = TFAlbertForSequenceClassification.from_pretrained("vumichien/albert-base-v2-imdb")

"""
Tokenize a text sequence:
- Return the tokenized inputs as TensorFlow tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

"""
Perform sequence classification inference:
- Calculate the logits for the input text sequence.
- Logits represent the predicted class probabilities.
"""
logits = model(**inputs).logits

"""
Find the predicted class ID:
- Find the class ID with the highest probability among the predicted logits.
- Convert the predicted class ID to its label using the model's configuration.
"""
predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
predicted_class = model.config.id2label[predicted_class_id]
print("Predicted Class:", predicted_class)

"""
To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`:
- In this case, the model's configuration includes information about the number of labels.
"""
num_labels = len(model.config.id2label)
model = TFAlbertForSequenceClassification.from_pretrained("vumichien/albert-base-v2-imdb", num_labels=num_labels)

"""
Define labels for sequence classification:
- In this example, a single label with the value 1 is used.
"""
labels = tf.constant(1)

"""
Measure latency for inference:
- This section measures the time it takes to perform inference 100 times.
- It records the start time, performs inference using the ALBERT model, records the end time,
  and calculates the time taken for each inference.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
"""
total_time = []
for i in range(100):
    start_time = time.time()
    loss = model(**inputs, labels=labels).loss
    end_time = time.time()
    total_time.append((end_time - start_time) * 1000)
total_time = np.array(total_time)
mean_time = np.mean(total_time)
print("Latency in ms:", mean_time)

"""
Calculate the loss for sequence classification:
- The loss is extracted from the outputs object returned by the ALBERT model.
- The loss represents how well the model is performing on the sequence classification task.
- The loss is rounded to two decimal places for clarity.
"""
loss = model(**inputs, labels=labels).loss
print("Loss:", round(float(loss), 2))

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
Calculate the number of GOPs (Giga-Operations) for the model:
- This section estimates the computational complexity of the model in terms of GOPs.
- It considers model configuration parameters such as hidden units, attention heads, sequence length, and layers.
- The result is printed to the console, providing insights into the computational complexity of the model.
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
print("ALBERT_TensorFlow_TokenClassification_albert-base-v2")

"""
Load the ALBERT model for TokenClassification:
- Import the AutoTokenizer and TFAlbertForTokenClassification classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "albert-base-v2" model.
  - Initialize the ALBERT model for token classification.
"""
from transformers import AutoTokenizer, TFAlbertForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = TFAlbertForTokenClassification.from_pretrained("albert-base-v2")

"""
Tokenize a text sequence without adding special tokens:
- Tokenize the input text "HuggingFace is a company based in Paris and New York" without adding special tokens.
- Return the tokenized inputs as TensorFlow tensors.
"""
inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
)

"""
Measure latency for inference:
- This section measures the time it takes to perform inference 100 times.
- It records the start time, performs inference using the ALBERT model, records the end time,
  and calculates the time taken for each inference.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
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
Find the predicted token class IDs:
- Find the class IDs with the highest logits for each token.
- Note that tokens are classified rather than input words, which means that
  there might be more predicted token classes than words.
- Multiple token classes might account for the same word.
"""
predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

"""
Map predicted token class IDs to label names using the model's configuration:
- Retrieve the label names corresponding to the predicted token class IDs.
- These label names represent the predicted classes for each token.
"""
predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
print("Predicted Token Classes:", predicted_tokens_classes)

"""
Define labels for token classification (using the predicted token class IDs):
- Use the predicted token class IDs as labels for the loss calculation.
"""
labels = predicted_token_class_ids

"""
Calculate the mean loss for token classification:
- The loss is extracted from the outputs object returned by the ALBERT model.
- The mean loss provides an indication of how well the model is performing on the token classification task.
- The loss is rounded to two decimal places for clarity.
"""
loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
print("Loss:", round(float(loss), 2))

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
Calculate the number of GOPs (Giga-Operations) for the model:
- This section estimates the computational complexity of the model in terms of GOPs.
- It considers model configuration parameters such as hidden units, attention heads, sequence length, and layers.
- The result is printed to the console, providing insights into the computational complexity of the model.
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
print("ALBERT_TensorFlow_QuestionAnswering_vumichien/albert-base-v2-squad2")

"""
Load the ALBERT model for QuestionAnswering:
- Import the AutoTokenizer and TFAlbertForQuestionAnswering classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "vumichien/albert-base-v2-squad2" model.
  - Initialize the ALBERT model for question-answering.
"""
from transformers import AutoTokenizer, TFAlbertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("vumichien/albert-base-v2-squad2")
model = TFAlbertForQuestionAnswering.from_pretrained("vumichien/albert-base-v2-squad2")

"""
Define a question and a text passage:
- Specify a question and a text passage for which you want to find an answer.
"""
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

"""
Tokenize the question and text, and return the tokenized inputs as TensorFlow tensors:
- Tokenize the question and text using the ALBERT tokenizer.
- Return the tokenized inputs as TensorFlow tensors.
"""
inputs = tokenizer(question, text, return_tensors="tf")

"""
Perform question-answering inference:
- Use the tokenized inputs to perform question-answering inference with the ALBERT model.
- The outputs contain logits for start and end positions of the answer span.
"""
outputs = model(**inputs)

"""
Find the answer span:
- Find the start and end indices of the answer span by identifying the positions with the highest logits.
- Extract the predicted answer tokens based on these indices and decode them.
"""
answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
predicted_answer = tokenizer.decode(predict_answer_tokens)
print("Predicted answer:", predicted_answer)

"""
Define target start and end indices for the answer span (ground truth):
- Specify the ground truth start and end indices for the answer span.
"""
target_start_index = tf.constant([12])
target_end_index = tf.constant([13])

"""
Measure latency for question-answering:
- This section measures the time it takes to perform question-answering 100 times.
- It records the start time, performs question-answering using the ALBERT model,
  records the end time, and calculates the time taken for each question-answering operation.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
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
Calculate the mean loss for question-answering:
- The loss is extracted from the outputs object returned by the ALBERT model.
- The mean loss provides an indication of how well the model is performing on the question-answering task.
- The loss is rounded to two decimal places for clarity.
"""
loss = tf.math.reduce_mean(outputs.loss)
print("Loss:", round(float(loss), 2))

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
Calculate the number of GOPs (Giga-Operations) for the model:
- This section estimates the computational complexity of the model in terms of GOPs.
- It considers model configuration parameters such as hidden units, attention heads, sequence length, and layers.
- The result is printed to the console, providing insights into the computational complexity of the model.
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
Import necessary libraries:
- Import JAX and JAX NumPy libraries for numerical computation with JAX arrays.
"""
import jax
import jax.numpy as jnp

"""
Print a descriptive message indicating the beginning of the script.
"""
print("ALBERT_Flax_MaskedLM_albert-base-v2")

"""
Load the ALBERT model for masked language modeling (MLM):
- Import the AutoTokenizer and FlaxAlbertForMaskedLM classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "albert-base-v2" model.
  - Initialize the ALBERT model for masked language modeling (MLM).
"""
from transformers import AutoTokenizer, FlaxAlbertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = FlaxAlbertForMaskedLM.from_pretrained("albert-base-v2")

"""
Tokenize a sentence with a masked token and return the tokenized inputs as JAX tensors:
- Tokenize a sentence containing a masked token using the ALBERT tokenizer.
- Return the tokenized inputs as JAX tensors.
"""
inputs = tokenizer("The capital of France is [MASK].", return_tensors="jax")

"""
Measure latency for MLM:
- This section measures the time it takes to perform masked language modeling (MLM) 100 times.
- It records the start time, performs MLM using the ALBERT model, records the end time, and calculates the time taken for each MLM operation.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
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
Retrieve logits for the next token (the masked token):
- The model returns logits for the next token, which corresponds to the masked token in the input sequence.
"""
print("Logits for Next Token:", outputs.logits)

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
Calculate the number of GOPs (Giga-Operations) for the model:
- This section estimates the computational complexity of the model in terms of GOPs.
- It considers model configuration parameters such as hidden units, attention heads, sequence length, and layers.
- The result is printed to the console, providing insights into the computational complexity of the model.
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
print("ALBERT_Flax_SequenceClassification_albert-base-v2")

"""
Load the ALBERT model for sequence classification:
- Import the AutoTokenizer and FlaxAlbertForSequenceClassification classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "albert-base-v2" model.
  - Initialize the ALBERT model for sequence classification.
"""
from transformers import AutoTokenizer, FlaxAlbertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = FlaxAlbertForSequenceClassification.from_pretrained("albert-base-v2")

"""
Tokenize a sentence and return the tokenized inputs as JAX tensors:
- Tokenize a sentence using the ALBERT tokenizer.
- Return the tokenized inputs as JAX tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

"""
Measure latency for inference:
- This section measures the time it takes to perform inference 100 times.
- It records the start time, performs inference using the ALBERT model, records the end time, and calculates the time taken for each inference operation.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
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
- The model returns logits for sequence classification.
"""
print("Sequence Classification Logits:", outputs.logits)

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
Calculate the number of GOPs (Giga-Operations) for the model:
- This section estimates the computational complexity of the model in terms of GOPs.
- It considers model configuration parameters such as hidden units, attention heads, sequence length, and layers.
- The result is printed to the console, providing insights into the computational complexity of the model.
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
print("ALBERT_Flax_TokenClassification_albert-base-v2")

"""
Load the ALBERT model for TokenClassification:
- Import the AutoTokenizer and FlaxAlbertForTokenClassification classes from the transformers library.
- Initialize the ALBERT tokenizer and model:
  - Initialize the ALBERT tokenizer for the "albert-base-v2" model.
  - Initialize the ALBERT model for token classification.
"""
from transformers import AutoTokenizer, FlaxAlbertForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = FlaxAlbertForTokenClassification.from_pretrained("albert-base-v2")

"""
Tokenize a sentence and return the tokenized inputs as JAX tensors:
- Tokenize a sentence using the ALBERT tokenizer.
- Return the tokenized inputs as JAX tensors.
"""
inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

"""
Measure latency for inference:
- This section measures the time it takes to perform inference 100 times.
- It records the start time, performs inference using the ALBERT model, records the end time, and calculates the time taken for each inference operation.
- The results are stored in the total_time list, which is converted to a NumPy array for analysis.
- The mean latency time is then calculated and printed.
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
- The model returns logits for token classification.
"""
print("Logits for Token Classification:", outputs.logits)

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
Calculate the number of GOPs (Giga-Operations) for the model:
- This section estimates the computational complexity of the model in terms of GOPs.
- It considers model configuration parameters such as hidden units, attention heads, sequence length, and layers.
- The result is printed to the console, providing insights into the computational complexity of the model.
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
