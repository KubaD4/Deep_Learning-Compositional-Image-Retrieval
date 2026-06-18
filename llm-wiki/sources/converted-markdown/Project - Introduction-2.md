Project
Assignment

Description & deliverables

Marco Garosi

University of Trento

Agenda

Project assignment

• Background

• Datasets

• Task

• Evaluation & deliverables

• Group registration

• Polices

Background

CLAY

Recent paper (accepted at CVPR

2026, so still to be published!)

Retrieval

User query

Matched entries

Database, i.e., a collection of images

“Indexing”

Indexed database

Retrieval

User query

Matched entries
(highest cosine
similarity)

Database, i.e., a collection of images

Embedding

Tensor with shape
(num_samples, emb_dim)

Retrieval

If you change the
embedding model, you
need to re-embed the
whole database, which can
be very expensive!

User query

Matched entries
(highest cosine
similarity)

Database, i.e., a collection of images

Embedding

Tensor with shape
(num_samples, emb_dim)

Retrieval

Idea: instead of changing
the entire model, we tweak
the query encoder, keeping
it compatible with the other
encoder, so that we can
keep the original database!

User query

Matched entries
(highest cosine
similarity)

Database, i.e., a collection of images

Embedding

Tensor with shape
(num_samples, emb_dim)

CLAY

CLAY conditions the similarity

between images by modulating

the original similarity

CLAY

To keep the database intact, we need an asymmetric modulation:

CLAY

Datasets

Datasets

You will work on CELEB-A

40 binary attributes for each

image, allowing you to

experiment easily

Task

Task description

CLAY has a rigid pre-SVD embedding

stacking as a fusion mechanism.

You should develop a new

methodological approach.

No restriction: you can

do whatever you want!

Task description

Training-free approaches

Training-based approaches

Prompt engineering, zero-

Lightweight adapters, cross-

shot attention scaling,

latent space arithmetic

attention networks, interpretability-

based sparse autoencoders (SAEs)

Task description

Main challenges

Overcoming the SVD bottleneck

• CLAY uses SVD, but it limits the

expressiveness of the model

• Explore diverse mechanisms!

Expressive multimodal conditioning

• Accept one or multiple conditions as input,

together with the given image

• Accept both positive and negative conditions

Evaluation protocol

We will provide you with queries

Simple queries, such as

Compound (composed) queries,

“+glasses” or “-red hair”

such as “+glasses, -smile” or

“+sad, -blue eyes”

You will find them in the document. We will update it shortly!

Evaluation protocol

To make things simple and fair, we have defined a set of queries.

For each query, we checked that there existed images that would

satisfy it.

We will provide you with a JSON file that contains all the queries,

the images that you should use as “source” for each query, and

the candidate targets (i.e., the ground truth) for each source.

Evaluation protocol

To make things simple and fair, we have defined a set of queries.

We will also provide the

For each query, we checked that there existed images that would

evaluation code, so that it’s

satisfy it.

We will provide you with a JSON file that contains all the queries,

the images that you should use as “source” for each query, and

the candidate targets (i.e., the ground truth) for each source.

easy to compare results

across groups – both for you

and for us!

Metrics

The code we will provide evaluates Recall@K and Precision@K:

Roadmap

We suggest you follow this roadmap:

1. Data exploration and preprocessing

2. Offline feature extraction

3. Vanilla zero-shot baseline

4. Method development

Evaluation &
deliverables

What we expect

A single Jupyter Notebook, hosted on Google Colab

It should be self-contained. It must have:

• Your code, so that we can run it

– Clean and readable! Leave comments

• Leave output cells, so that we can check it more easily

• A comprehensive report, in the Markdown cells between code cells

– Methodological description: a detailed description of the solution you are proposing

– Experimental setup: describe your training / evaluation strategy, and motivate all your choices!

– Results and discussion: make sure we understand why you got your results

Evaluation axes

We will evaluate your work based on:

1. Originality and creativity

2. Methodological thoroughness

– Don’t just say “we did this and it worked”

3. Report clarity

4. Empirical performance

– We’d like to see some decent performance, but we’re not interested

in you becoming SOTA. Just show you can improve upon a baseline!

5. Code quality

– Don’t submit bad code

Group registration

Group registration

Do you have
a group?

No

Yes

Do you want
to join one?

Yes

No

We will try to match you

with other students.

You will work alone.

Only one person

should register the

whole group!

Policies

Policies

• Do not copy-paste code from other repositories

– We want you to understand what you are doing and how PyTorch works. Don’t

use code as a black-box!

• You can borrow code from the Internet, such as “standard” functions,

but cite the source!

• You are encouraged to discuss your approach with other groups

• You should not share your code with other groups!

Project
Assignment

Description & deliverables

Marco Garosi

University of Trento

