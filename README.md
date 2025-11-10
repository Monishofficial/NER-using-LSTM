# EXP-06 Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

Named Entity Recognition (NER) is a Natural Language Processing (NLP) task that involves identifying and classifying entities such as names of people, organizations, locations, dates, and more within a given text.  
The main goal is to label each word in a sentence with its corresponding entity type, such as PERSON, ORGANIZATION, LOCATION, or O (for non-entity words).

In this experiment, an LSTM-based neural network model is used to perform NER.  
The dataset consists of sentences where each word is tagged with an entity label.  
Before training, the text is tokenized, encoded into numerical form, and padded to ensure uniform sequence length.  
The model learns the relationships between word sequences and their corresponding entity tags to accurately predict entities in new, unseen text.


## DESIGN STEPS

### Step 1:
Import the required libraries such as PyTorch, NumPy, and Pandas.

### Step 2:
Load the text dataset containing words and their corresponding entity tags.

### Step 3:
Preprocess the text data by tokenizing words and converting them into integer indices.

### Step 4:
Encode the entity labels and prepare input-output pairs for training.

### Step 5:
Define the LSTM-based model architecture with an embedding layer, LSTM layer, and a fully connected output layer.

### Step 6:
Specify the loss function (CrossEntropyLoss) and optimizer (Adam).

### Step 7:
Train the model by feeding the input sequences, calculating loss, and updating weights.

### Step 8:
Evaluate the trained model on test data to measure prediction accuracy.

### Step 9:
Predict named entities for new unseen text sequences.

### Step 10:
Display the input text along with predicted named entities for visualization.


## PROGRAM
### Developed By: MONISH N
### Register Number: 212223240097

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.dropout=nn.Dropout(0.1)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_dim*2,tagset_size)

    def forward(self,x):
        x=self.embedding(x)
        x=self.dropout(x)
        x,_=self.lstm(x)
        return self.fc(x)  


model=BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


# Training and Evaluation Functions
def train_model(model,train_loader,test_loader,loss_fn,optimixer,epochs=3):
  train_losses,val_losses=[],[]
  for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch in train_loader:
      input_ids=batch["input_ids"].to(device)
      labels=batch["labels"].to(device)
      optimizer.zero_grad()
      outputs=model(input_ids)
      loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss)

    model.eval()
    val_loss=0
    with torch.no_grad():
      for batch in test_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        val_loss+=loss.item()
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

  return train_losses,val_losses
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
<img width="707" height="655" alt="image" src="https://github.com/user-attachments/assets/17de9d20-d6b8-44e0-ba6b-877927abe104" />


### Sample Text Prediction
<img width="404" height="437" alt="image" src="https://github.com/user-attachments/assets/ffcce468-2f26-42c1-a415-c2a18784a162" />


## RESULT
The LSTM-based Named Entity Recognition model was successfully implemented and accurately identified entities from the given text dataset.
