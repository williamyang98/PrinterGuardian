import onnxruntime as ort
import numpy as np
from tabulate import tabulate

class Model:
    def __init__(self, session: ort.InferenceSession, metadata: dict, prototypes: dict):
        self.session = session
        self.metadata = metadata
        self.prototypes = prototypes
    
        self.normalise_mean = metadata["preprocessing"]["normalise_mean"]
        self.normalise_std = metadata["preprocessing"]["normalise_std"]

        self.class_names = list(prototypes["prototypes"].keys())
        self.class_embeddings = np.array(list(prototypes["prototypes"].values()))

        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()
        assert len(model_inputs) == 1
        assert len(model_outputs) == 1
        model_input = model_inputs[0]
        model_output = model_outputs[0]
        assert len(model_input.shape) == 4
        assert model_input.shape[1] == 3
        input_width = model_input.shape[2]
        input_height = model_input.shape[3]
        self.input_name = model_input.name
        self.output_name = model_output.name

        assert len(model_inputs) == 1
        model_input = model_inputs[0]
        assert len(model_input.shape) == 4 # B,C,H,W
        self.input_channels = 3
        assert model_input.shape[0] == 1
        assert model_input.shape[1] == self.input_channels
        self.input_width = model_input.shape[2]
        self.input_height = model_input.shape[3]

        assert len(model_outputs) == 1
        model_output = model_outputs[0]
        assert len(model_output.shape) == 2 # B,L
        self.embedding_dims = self.class_embeddings.shape[1]
        assert model_output.shape[0] == 1
        assert model_output.shape[1] == self.embedding_dims

        self.success_class_index = self.class_names.index("success")

    def debug_print(self):
        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()

        tablefmt = "grid"
        output = ""
        output += "Model inputs:\n"
        output += tabulate(
            [(i.name, i.shape, i.type) for i in model_inputs],
            headers=["Name", "Shape", "Type"],
            showindex="always", tablefmt=tablefmt,
        )
        output += "\n\nModel outputs:\n"
        output += tabulate(
            [(o.name, o.shape, o.type) for o in model_outputs],
            headers=["Name", "Shape", "Type"],
            showindex="always", tablefmt=tablefmt,
        )
        return output

    # expect batch of grayscale image
    def preprocess_batch(self, batch):
        assert len(batch.shape) == 3 # B,H,W
        assert batch.shape[1] == self.input_height
        assert batch.shape[2] == self.input_width
        assert batch.dtype == np.float32

        batch = batch[:, np.newaxis, ...]
        batch = np.repeat(batch, self.input_channels, axis=1) # B,C,H,W
        for c in range(self.input_channels):
            batch[:,c,...] = (batch[:,c,...] - self.normalise_mean[c]) / self.normalise_std[c]
        return batch

    def infer_batch_probabilities(self, batch, temperature=1.0):
        assert len(batch.shape) == 4 # B,C,H,W
        assert batch.shape[1] == self.input_channels
        assert batch.shape[2] == self.input_height
        assert batch.shape[3] == self.input_width
        assert batch.dtype == np.float32
        batch_count = batch.shape[0]

        outputs = np.zeros((batch_count, self.embedding_dims), dtype=np.float32)
        # model was exported with batch_size=1
        for i in range(batch_count):
            output, = self.session.run([self.output_name], { self.input_name: batch[i,...][np.newaxis,...] })
            outputs[i,...] = output[0,...]

        L2_distances = np.zeros([batch_count, len(self.class_names)], dtype=np.float32)
        for i, name in enumerate(self.class_names):
            class_embedding = self.class_embeddings[i,:]
            L2_distance = np.sum((outputs - class_embedding)**2, axis=1)
            L2_distances[:,i] = L2_distance

        probs = L2_distances_to_probabilities(L2_distances, temperature=temperature)
        probs = probs[:, self.success_class_index]
        return probs

def L2_distances_to_probabilities(L2_distances, temperature=1.0):
    assert len(L2_distances.shape) == 2
    logits = -L2_distances / temperature
    logits -= np.max(logits, axis=1)[:,np.newaxis]
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1)[:,np.newaxis]
    return probs