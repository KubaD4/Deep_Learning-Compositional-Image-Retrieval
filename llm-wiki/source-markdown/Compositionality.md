Compositionality
&
the Geometry of CLIP
Davide Berasi
Deep Learning Course, 13/04/2026

Outline
● Introduction to compositionality
● Compositionality in deep learning
● Compositionality in CLIP
● Geometrical properties of CLIP embeddings

Understanding complex concepts
● Humans can understand new complex concepts never seen before.
● How can we do it?
“A purple elephant fixing
a refrigerator.”

Understanding complex concepts
● Humans can understand new complex concepts never seen before.
● How can we do it? → Compositional Understanding
“A purple elephant fixing
a refrigerator.”

Compositionality
The meaning of an expression is a function of the meanings of
its parts and of the way they are syntactically combined.[1]
Subject: Dog Mood: Peacefull
Age: Young Fur color: Beige
Action: Sitting Location: Grass Field
[1] Partee, Barbara, "Compositionality" (1984)

Compositional Understanding
Compositional Understanding is the ability to exploit compositionality.
Humans do Compositional Understanding:
● We decompose complex concepts into their primitive parts.
● We compose primitive meanings into complex concepts.

Compositional Understanding
We use this mechanism in:
● Language: a sentence combines words via grammatical rules,
● Visual perception: in a scene, we identify objects, their properties,
relations, …
It is at the core of intelligence: it is efficient and enables generalization!

Compositional Understanding
● Effiency: a finite set of primitives and rules gives an exponential
number of combinations.
(E.g., in english, we can express any thought with ~50k words.)
● Generalization: we can understand/imagine new concepts.
Memorized concepts
Compositional
Understanding

Compositionality in Deep Learning
Classical symbolic AI (1950-1980):
● Explicit symbols + rules (e.g., rule-based systems, logic programs).
● Compositionality was explicitly imposed.
Modern deep learning:
● Directly learns from data. Compositionality is not imposed a priory.
● Does any compositional behaviour emerge?

Let’s look at CLIP…
CLIP represents complex concepts (text & images) as vectors.
● Image encoder gives u ∈ ℝ512
I
● Text encoder gives u ∈ ℝ512
T
● Cosine similarity measures semantic similarity.

Are CLIP embeddings compositional?
Two possibilities:
1. CLIP is not compositional:
● It simply memorizes all the concepts in the training set.
2. CLIP is compositional:
● It learns primitive concepts and how to combine them.
● Embeddings are organized in compositional structures.

CLIP compositionality
Let’s consider a simple setting:
| ● Set of complex concepts Z = {a    | ,...,a }x{o | , …,o | }    |     |
| ----------------------------------- | ----------- | ----- | ---- | --- |
|                                     | 1 N         | 1     | M    |     |
| ● Embeddings are decomposable if  u |             | = u   |  + u |     |
|                                     | (attr, obj) |       | attr | obj |
Trager et al., “Linear spaces of meanings compositional structures in vision-language models”, ICCV23

CLIP compositionality
CLIP embeddings are actually on a sphere…let’s account for that.
Berasi et al., “Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models”, CVPR25

CLIP compositionality
1. Embeddings are not perfectly decomposable.
2. We search the decomposable set that best
approximates them.
3. Primitive directions of the ‘best decomposable
approximation’ are vector means.
4. We show this is a good approximation of the
original embeddings.

CLIP compositionality
Experiments show that the approximation is good, meaning CLIP
embeddings are close to be decomposable.
Projection of 2x3 concepts set. Inverting decomposable approximations.

Shortcomings in CLIP compositionality
● CLIP is bad at more complex compositions.
● It struggles with words order (treats text as a ‘bag-of-words’).
● Actually, CLIP may encode order information, but fail in cross-modal
alignment[Koishigarina et al., 2026].
Yuksekgonul et al.,”When and why vision-language models behave like bags-of-words, and what to do about it?”, ICLR23
Kamath et al., “What’s “up” with vision-language models? investigating their struggle with spatial reasoning”, EMNLP23
Lewis et al., “Does CLIP Bind Concepts? Probing Compositionality in Large Image Models“, EACL24
Kang et al., “Is CLIP ideal? No. Can we fix it? Yes!”, ICCV25
Koishigarina et al., “CLIP Behaves like a Bag-of-Words Model Cross-modally but not Uni-modally”, ICLR26

Latent space geometry
Besides compositional structures, the CLIP latent space has other
interesting properties:
● Narrow Cone Effect
● Modality Gap
● Orthogonality and Superposition
● Spherical geometry: Log and Exp maps

The narrow cone effect
● Embeddings from deep networks cluster in a narrow cone.
● E.g., for CLIP image encoder, avg. cosine is ~0.5 (60o).
RK: In ℝ512 a 60o cone occupies ~10-153% of the hypersphere surface.
avg cosine = 0 avg cosine > 0
Ethayarajh, “How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings”, EMNLP19

Modality Gap
● Text and image embeddings are in two narrow separate cones.
● Remember this if you combine embedding across modalities…
Liang et al., “Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning”, NeurIPS22

Orthogonality and Superposition
● Concepts can be grouped in attributes (e.g., subject, shape, color,
size, …)
● Varying a concept within an attribute should not change concepts
within the other attributes. Indeed:
○ Concepts from different attributes are roughly orthogonal.
○ Concepts within the same attribute are not orthogonal.
Oldfield et al., “Parts of Speech-Grounded Subspaces in Vision-Language Models”, NeurIPS23
Stein et al., “Towards Compositionality in Concept Learning”, ICML24

Orthogonality and Superposition
How can we have thousands of orthogonal concepts in the ‘small’ CLIP
latent space (ℝ512)?
Superposition [from Johnson-Lindenstrauss lemma]:
In ℝN, you can fit ~exp( εN ) vectors with angles in (90o- ε, 90o+ ε ).
Ex: in ℝ100, we can fit 10k vectors with pairwise angles in (89o, 91o).

Spherical geometry
● When playing with normalized embeddings (arithmetic,
projections), the sphere geometry can lead to small distortions.
● A solution is using tangent plane approximation (log and exp maps).

Thank you for your
attention!