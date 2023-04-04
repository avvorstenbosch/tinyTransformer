import torch

dtype_dict = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class Config:
    def __init__(
        self,
        data_dir="./data/corpus/",  # Where is the data stored?
        batch_size=128,  # How many independent sequences will we process in parallel?
        block_size=256,  # What is the maximum context length for predictions?
        n_head=6,  # How many self-attention head does a multiheaded self attention block get?
        n_embed=None,  # How many dimensions do our embeddings have?
        n_blocks=4,  # How many sequential self-attention blocks does our model get?
        n_layers=2,  # How many FF-compute layers should the end of a block have?
        epochs=30,  # For how many epochs will we train the model?
        steps_per_epoch=10000,  # How many training steps will we take per Epoch?
        eval_interval=1000,  # How often will we print results for training?
        learning_rate=1e-4,  # What is our learning rate?
        eval_iters=200,  # How many samples to use to estimate loss?
        model_precision="bfloat16",  # Do you want to set the model_precision to float16 to be faster and reduce the memory?
        compile=True,  # Do you want to compile the model in Pytorch 2.0 to be faster?
        testrun=False,  # Do you want to test the code on a small dataset?
        out_dir="./models/",  # Where do you want the output to go?
        device="cuda" if torch.cuda.is_available() else "cpu",  # Where will we train?
        dropout_percentage=0.2,  # How do you want to set the dropout rate?
        compute_layer_scaling=2,  # With what factor do you want the FF-compute layer to scale?
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_head = n_head
        self.n_embed = n_embed or 128 * n_head
        self.head_size = self.n_embed // n_head
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.eval_iters = eval_iters
        self.model_precision = dtype_dict[model_precision]
        self.compile = compile
        self.testrun = testrun
        self.out_dir = out_dir
        self.device = device
        self.dropout_percentage = dropout_percentage
        self.compute_layer_scaling = compute_layer_scaling
