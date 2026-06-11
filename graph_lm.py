import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# -------------------
# DATASET
# -------------------

# -------------------
# TRAINING DATA
# -------------------

subjects = [
    "compressor",
    "turbine",
    "combustor",
    "rocket",
    "nozzle",
    "pump",
    "shaft",
    "bearing",
    "airfoil",
    "engine"
]

verbs = [
    "increases",
    "produces",
    "drives",
    "accelerates",
    "creates",
    "transfers",
    "supports",
    "compresses"
]

objects = [
    "pressure",
    "power",
    "thrust",
    "temperature",
    "torque",
    "flow",
    "energy",
    "air"
]

sentences = []

for s in subjects:
    for v in verbs:
        for o in objects:
            sentences.append(
                f"{s} {v} {o}"
            )

# Add some fixed engineering facts

sentences.extend([
    "compressor increases pressure",
    "air enters compressor",
    "fuel enters combustor",
    "combustor raises temperature",
    "hot gas enters turbine",
    "turbine produces power",
    "turbine drives compressor",
    "rocket produces thrust",
    "nozzle accelerates gas",
    "stress causes strain",
    "hooke law relates stress strain",
    "young modulus relates stress strain",
    "shaft transmits torque",
    "torque causes shear",
    "bearing supports shaft",
    "airfoil produces lift",
    "lift supports aircraft",
    "drag opposes motion",
    "reynolds number predicts flow",
    "boundary layer affects drag"
])

print("Training Sentences:", len(sentences))

# -------------------
# VOCAB
# -------------------

words = set()

for s in sentences:
    words.update(s.split())

vocab = sorted(list(words))

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(vocab)

print("Vocabulary")
print(word2idx)

# -------------------
# GRAPH BUILDER
# -------------------

def sentence_to_graph(sentence):

    tokens = sentence.split()

    node_ids = [word2idx[t] for t in tokens]

    x = torch.tensor(
        node_ids,
        dtype=torch.long
    )

    edges = []

    for i in range(len(tokens)-1):

        edges.append([i, i+1])
        edges.append([i+1, i])

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t()

    return Data(
        x=x,
        edge_index=edge_index
    )

# -------------------
# MODEL
# -------------------

class GraphLM(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            32
        )

        self.gcn1 = GCNConv(
            32,
            64
        )

        self.gcn2 = GCNConv(
            64,
            64
        )

        self.fc = nn.Linear(
            64,
            vocab_size
        )

    def forward(self, graph):

        x = self.embedding(
            graph.x
        )

        x = self.gcn1(
            x,
            graph.edge_index
        )

        x = torch.relu(x)

        x = self.gcn2(
            x,
            graph.edge_index
        )

        x = torch.relu(x)

        sentence_embedding = x.mean(
            dim=0
        )

        logits = self.fc(
            sentence_embedding
        )

        return logits

# -------------------
# TRAIN
# -------------------

model = GraphLM(vocab_size)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

loss_fn = nn.CrossEntropyLoss()

for epoch in range(2000):

    total_loss = 0

    for sentence in sentences:

        tokens = sentence.split()

        input_text = " ".join(
            tokens[:-1]
        )

        target_word = tokens[-1]

        graph = sentence_to_graph(
            input_text
        )

        target = torch.tensor([
            word2idx[target_word]
        ])

        prediction = model(graph)

        loss = loss_fn(
            prediction.unsqueeze(0),
            target
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:

        print(
            f"Epoch {epoch} Loss {total_loss:.4f}"
        )

# -------------------
# TEST
# -------------------

while True:

    text = input(
        "\nInput: "
    )

    graph = sentence_to_graph(
        text
    )

    with torch.no_grad():

        out = model(graph)

        pred = out.argmax().item()

    print(
        "Prediction:",
        idx2word[pred]
    )