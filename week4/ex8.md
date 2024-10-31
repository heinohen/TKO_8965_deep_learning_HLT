# EX 8

1) Introduction

    Summary: Describes the current progress of architectures and proposes new better model.

    What I learned or found interesting: They use term "as little as twelve hours on _eight_ P100 GPUs", as that would be considered as a small amount of computational time.

2) Background

    Summary: Presents the problem of reducing sequential computation by giving solution as transformers model with relying solely on self-attention for presentations and not seq RNNs.

    What I learned or found interesting: The timeframe is not very long from RNNs to transformers to today.

3) Model Architecture

    Summary: The transformer model is pieced and presented in slices and the reasoning including just that part.

    What I learned or found interesting: I found interesting the different complexities per layer type.

4) Why Self-Attention

    Summary: Writers argued the use of self-attention using three parameters and showed per parameter gains of this proposed model.

    What I learned or found interesting: How general the writers of the paper thought that the BERT model could be.

5) Training

    Summary: The section goes throught he rigorious training process and the parameters used.

    What I learned or found interesting: That the training of the big models were trained for 3.5 days (300,000 steps).

6) Results

    Summary: Section presents the results of the new model compared to previous work.

    What I learned or found interesting: That the gain of 2 BLEU is considered significant gains.

7) Why Self-Attention

    Summary: The writers are convinced that this is the new future of LLMs.

    What I learned or found interesting: That they give out all of the code in respect to open source.
