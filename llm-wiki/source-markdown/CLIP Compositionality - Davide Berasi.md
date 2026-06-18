ThisCVPRpaperistheOpenAccessversion,providedbytheComputerVisionFoundation.
Exceptforthiswatermark,itisidenticaltotheacceptedversion;
thefinalpublishedversionoftheproceedingsisavailableonIEEEXplore.
Not Only Text: Exploring Compositionality of Visual Representations
|     |     |     |                         |     | in            | Vision-Language |                     |                      | Models              |     |     |     |     |     |
| --- | --- | --- | ----------------------- | --- | ------------- | --------------- | ------------------- | -------------------- | ------------------- | --- | --- | --- | --- | --- |
|     |     |     | DavideBerasi1           |     |               | MatteoFarina2   |                     | MassimilianoMancini2 |                     |     |     |     |     |     |
|     |     |     |                         |     | ElisaRicci1,2 |                 | NicolaStrisciuglio3 |                      |                     |     |     |     |     |     |
|     |     |     | 1FondazioneBrunoKessler |     |               |                 | 2UniversityofTrento |                      | 3UniversityofTwente |     |     |     |     |     |
Abstract
| Vision-Language |           | Models      | (VLMs)   | learn   | a              | shared      | feature |     |     |     |     |     |     |     |
| --------------- | --------- | ----------- | -------- | ------- | -------------- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- |
| space for       | text and  | images,     | enabling |         | the comparison |             | of in-  |     |     |     |     |     |     |     |
| puts of         | different | modalities. |          | While   | prior          | works       | demon-  |     |     |     |     |     |     |     |
| strated         | that VLMs | organize    |          | natural | language       | representa- |         |     |     |     |     |     |     |     |
tionsintoregularstructuresencodingcompositemeanings,
| it remains    | unclear   | if  | compositional |     | patterns   | also | emerge   |     |     |     |     |     |     |     |
| ------------- | --------- | --- | ------------- | --- | ---------- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- |
| in the visual | embedding |     | space.        | In  | this work, | we   | investi- |     |     |     |     |     |     |     |
gatecompositionalityintheimagedomain,wheretheanal-
ysisofcompositionalpropertiesischallengedbynoiseand
| sparsity   | of visual    | data. | We                | address      | these | problems     | and |     |     |     |     |     |     |     |
| ---------- | ------------ | ----- | ----------------- | ------------ | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
| propose    | a framework, |       | called            | Geodesically |       | Decomposable |     |     |     |     |     |     |     |     |
| Embeddings | (GDE),       |       | that approximates |              | image | representa-  |     |     |     |     |     |     |     |     |
tions with geometry-aware compositional structures in the Figure1.Compositionalstructuresinvisualembeddingspace.
|     |     |     |     |     |     |     |     | (left) Pre-trained |     | VLM represents |     | visual inputs | of  | composite |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | -------------- | --- | ------------- | --- | --------- |
latentspace.Wedemonstratethatvisualembeddingsofpre-
|         |      |         |                 |     |              |     |     | meanings | in regular | geometric | shapes. | The modularity |     | of these |
| ------- | ---- | ------- | --------------- | --- | ------------ | --- | --- | -------- | ---------- | --------- | ------- | -------------- | --- | -------- |
| trained | VLMs | exhibit | a compositional |     | arrangement, |     | and |          |            |           |         |                |     |          |
evaluate the effectiveness of this property in the tasks of structuresenablestheseparationoftheprimitivecomponentsand
|               |     |                |     |           |             |     |     | the composition |     | of unseen | combinations. | (right) | We  | evaluate |
| ------------- | --- | -------------- | --- | --------- | ----------- | --- | --- | --------------- | --- | --------- | ------------- | ------- | --- | -------- |
| compositional |     | classification |     | and group | robustness. |     | GDE |                 |     |           |               |         |     |          |
theusefulnessofthesepropertiesincompositionalclassification,
achievesstrongerperformanceincompositionalclassifica-
grouprobustness,andimagegeneration.
| tion compared                |     | to its counterpart |     | method                   | that | assumes | lin- |     |     |     |     |     |     |     |
| ---------------------------- | --- | ------------------ | --- | ------------------------ | ---- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- |
| eargeometryofthelatentspace. |     |                    |     | Notably,itisparticularly |      |         |      |     |     |     |     |     |     |     |
effectiveforgrouprobustness,whereweachievehigherre-
|     |     |     |     |     |     |     |     | capability | by  | developing | models | that imitate |     | composi- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ---------- | ------ | ------------ | --- | -------- |
sults than task-specific solutions. Our results indicate that tional processes, e.g., solving complex tasks via sub-goals
VLMscanautomaticallydevelopahuman-likeformofcom-
|     |     |     |     |     |     |     |     | [9, 31, | 49, 61], | modeling | objects | as compositions |     | of their |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | -------- | ------- | --------------- | --- | -------- |
positionalreasoninginthevisualdomain,makingtheirun- parts [13, 14, 47, 60, 62], encoding concept hierachies
derlying processes more interpretable. Code is available [11,17,33,55],explicitlylearningcompositionalrepresen-
athttps://github.com/BerasiDavide/vlm_image_
|     |     |     |     |     |     |     |     | tations[1,18,38,44],orarchitectures |     |     |     | [20,24,36,63,68]. |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | ----------------- | --- | --- |
compositionality. With the rise of modern Vision-Language Models
|     |     |     |     |     |     |     |     | (VLMs)      | [26, 54, | 74] jointly | trained | on large-scale |                  | image- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | ----------- | ------- | -------------- | ---------------- | ------ |
|     |     |     |     |     |     |     |     | text pairs, | there    | has been    | growing | interest       | in investigating |        |
1.Introduction
|                  |     |        |           |     |       |           |     | whether            | these | models exhibit                     |     | instrinsic compositional |     | be- |
| ---------------- | --- | ------ | --------- | --- | ----- | --------- | --- | ------------------ | ----- | ---------------------------------- | --- | ------------------------ | --- | --- |
|                  |     |        |           |     |       |           |     | haviors[45,52,71]. |       | Inparticular,Trageretal.[67]inves- |     |                          |     |     |
| Compositionality |     | is the | principle | by  | which | cognitive | and |                    |       |                                    |     |                          |     |     |
tigatedlatentcompositionalstructureswithintheCLIP[54]
computationalsystemscreatemeaningofacomplexexpres-
|         |           |     |         |     |               |     |            | text embedding |     | space, | demonstrating | that | composite | con- |
| ------- | --------- | --- | ------- | --- | ------------- | --- | ---------- | -------------- | --- | ------ | ------------- | ---- | --------- | ---- |
| sion by | combining | the | meaning | of  | its (simpler) |     | parts [50, |                |     |        |               |      |           |      |
ceptscanberepresentedaslinearcombinationsofembed-
51]. Humansleveragecompositionalityinstinctively,com-
|              |               |          |         |           |       |              |      | ding vectors              | corresponding |     | to                        | various factors.        | These | vec- |
| ------------ | ------------- | -------- | ------- | --------- | ----- | ------------ | ---- | ------------------------- | ------------- | --- | ------------------------- | ----------------------- | ----- | ---- |
| bining known |               | elements | to      | interpret | novel | situations.  | In   |                           |               |     |                           |                         |       |      |
|              |               |          |         |           |       |              |      | tors, calledidealwords,   |               |     | canbeusedtocomposenewcon- |                         |       |      |
| machine      | intelligence, |          | efforts | were      | made  | to replicate | this |                           |               |     |                           |                         |       |      |
|              |               |          |         |           |       |              |      | ceptsintheembeddingspace. |               |     |                           | Theirworkfocusesonfind- |       |      |
Correspondingauthor:dberasi@fbk.eu. ingcompositionalstructuresinthetextembeddingspaceof
24917

CLIP,motivatedbythefactthatthestructuredandsymbolic tionofsimplerconceptstounderstandandreasononcom-
natureoflanguagemayfacilitatethestudyofcomputational plexones, allowingforgeneralizationtonewunseencom-
approaches to capture compositional meaning. However, binations of concepts [31, 44, 60]. In computer vision,
cognitive studies show that language itself is used to de- early efforts focused on recognizing objects as composi-
scribeandinterpretthevisualworldanddirectlyaffectsvi- tion of parts [13, 14, 47, 48] and evolved into architec-
sual perception [6]. Hence, similar to text, human visual tures that can recognize and model objects in a composi-
representationsexhibitacompositionalstructure[19],made tional fashion [60], compositional generation [49, 64, 76],
of simpler components systematically combined. Despite andinterpretablerepresentations[4,63]. Compositionality
thisconnection,compositionalpropertiesofvisualembed- has also lead to progress in various tasks, such as human-
dingsofVLMshaveremainedsofarmostlyunexplored. object interaction detection, model spatial/semantic rela-
To fill this gap, in this paper, we introduce GEODESI- tionships [21, 22, 28], and compositional zero-shot learn-
CALLY DECOMPOSABLE EMBEDDINGS (GDE), a frame- ing,wherethegoalistorecognizeunseencompositionsof
work grounded in differential geometry and designed to trainingprimitives[37,41–44].Whiletheseworksfocuson
investigate compositional structures of pre-trained embed- specificapplications,inthispaperweaimtostudywhether
dings within Riemannian manifolds (Fig.1). Visual em- thereexistsanunderlyingcompositionalstructureinthevi-
beddings exhibit unique challenges not present in compo- sualembeddingsofVLMs.
sitional analysis of text embeddings, namely data sparsity Compositionality in VLMs. Modern Vision-Language
inthecompositionalspaceandnoiseandambiguityinim- Models(VLMs)likeCLIP[54]aretrainedtoextractmean-
ages. Specifically, we deal with the sparsity of composite ingful representations from complex visual scenes guided
concepts,ascertaincombinationsofelementaryprimitives by textual inputs without a priori imposing any form of
maynotappearinrealimagecollections(e.g.,focusingon compositionality.Inthiscontext,anaturalquestionis:Does
objectsandattributes,“bluedog”imagesareunlikelytoex- compositionalbehavioremergeautomaticallyinVLMs?
ist). Noise and ambiguity concern additional visual cues Previous works already showed how VLMs are more
and information present in images, e.g. background, con- suitable for tasks such as compositional zero-shot learn-
text,etc.,thatdonotcorrespondtothecompositeconcepts. ing [40, 45, 52], and how their representations allow for
We evaluated the compositional representations computed cross-modalcompositions,suchasvisualediting[5,29,75]
with the proposed approach in two relevant applications, and compositional retrieval [3, 23, 27, 59]. At the same
namely compositional classification and group robustness time,worksstudiedthechallengesofVLMsinmodelcom-
(Fig. 1), considering publicly available datasets, showing positional inputs, e.g., at the level of word order, object-
that it better captures visual compositional structures than attributebindings, spatialrelationshipsandothercomposi-
the alternatives (e.g., [10]). GDE is particularly effective tionalchallenges[23,65,66,70].
forgrouprobustness,whereweachievebetterdebiasingre- Inthispaper,westudythecompositionalstructureinthe
sultsthantask-specificmethods.Furthermore,weshowthat visualembeddingsextractedfromVLMs. Closetoourgoal
GDEcanbesuccessfullyusedincombinationwithstate-of- is [35],studyingthecompositionalpropertiesoftheCLIP
the-artgenerativemodelstosynthesizeimagesofcomposi- text encoder through compositional distributional seman-
tionalconcepts. Ourcontributionsare: ticsmodelsinsynthetictestscenarios. Similarly,[67]show
i) We study compositional structures within visual em- that the textual embeddings of VLMs can be well approx-
beddings for VLMs and demonstrate that the latent imatedbylinearcompositionsofsmallersetsofidealvec-
representationsofvisualsignalsalsoexhibitadegree tors. Motivatedbythecross-modalalignmentofVLMs,we
ofcompositionality. investigatewhethertheembeddingsofvisualinputsexhibit
ii) Weshowthat,unlikefortextembeddings,linearstruc- an analogous compositional property. We achieve this by
turesareinsufficientto(de)composevisualconcepts; constructing a geometry-aware decomposition framework,
thus,themanifoldgeometrymustbeconsidered. following ideas similar to [46], where Principal Geodesic
iii) We propose a framework that deals with the sparsity Analysis(PGA)[15]isappliedtolearnlower-dimensional
and noise of composite concepts in images, enabling submanifoldsoftheCLIPspherethatareassociatedtodis-
thecompositionalanalysisofvisualembeddings. tinctparts-of-speech. Tothebestofourknowledge,thisis
the first work that investigates the emergence of composi-
2.RelatedWork tionalstructuresinthevisualembeddingsofVLMs.
Compositionality in Vision. Compositionality is consid-
3.Method
ered a cornerstone of perception [34], and compositional
representations offer an effective tool to represent real- Weproposeaframeworktoanalyzethecompositionalprop-
world phenomena [12]. The primary benefit of compo- erties of image embeddings of neural encoders. We start
sitionality is the possibility of combining the representa- by reviewing the fundamentals of the CLIP model along
24918

with key concepts from differential geometry (Sec. 3.1). 3.2.GeodesicallyDecomposableEmbeddings
We then formalize the concept of geodesic decomposabil-
We now formalize our proposed notion of compositional
ity (Sec. 3.2) and we discuss our methodology for dealing
embeddings. We consider a set of composite meanings
withvisualinputs(Sec.3.3).
Z = Z × ··· × Z , defined as the Cartesian product
1 s
3.1.Preliminaries between finite lists of primitive concepts, and refer to the
Z (i = 1,...,s) as the dimensions of Z. For example,
i
Contrastive Language-Image Pretraining (CLIP) con-
Z ={red,blue}×{car,dress,flower}combinesprimitives
sistsofapre-trainedimageencoderϕ : X → Rd anda
im fromanattributedimensionandanobjectdimension.
textencoderϕ : Y → Rd thatrepresentmulti-modaltext-
t Wethenconsideranembeddingmapϕ : Z → Mrep-
visualinputsinasharedvision-languagespace. Thelatent
resenting the composite concepts as points on a manifold
representationsofanimagex∈X andtexty ∈Y arecom- M⊂Rd. Intuitively,thesetϕ(Z)={u |z ∈Z}iscom-
paredbycosinesimilarity,whichisthescalarproductu⊤u z
x y positionalifithasaregularstructurereflectingthecompos-
of their normalized versions u = ϕ (x)/||ϕ (x)||,
x im im ite nature of the inputs, i.e., if one can compose primitive
u = ϕ (y)/||ϕ (y)||. The weights of the encoders are
y t t conceptswithinthegeometricspacetoobtainembeddings
trained to optimize a contrastive objective on a huge col-
of complex meanings. In this paper, we associate compo-
lection of paired image-text samples. Since the norm of
sitionalitytothenotionofgeodesicdecomposabilitywhich
CLIPembeddingsdoesnotcarryanymeaningfulinforma-
accountsfortheintrinsicgeometryofthemanifold.
tion,sphericalgeometryappliestotheirpost-hocanalysis.
RiemannianManifoldsaregeometricspaceswhereintrin- Definition1(Geodesicallydecomposableembeddings). A
sic distances can be measured. For a generic manifold set of embeddings ϕ(Z) = {u |z ∈ Z} ⊂ M with in-
z
M ⊂ Rd withintrinsicdistanced : M×M → [0,∞), trinsic mean µ is geodesically decomposable if there exist
M
we now recall the notions of exponential map and intrin- v ∈T Mforallz ∈Z (i=1,...,s)suchthat
zi µ i i
sic mean. These tools permit operating with non-linear
u =Exp (v +···+v ) ∀z =(z ,...,z ) (3)
data,likethesphericalnormalizedCLIPembeddings,while z µ z1 zs 1 s
respecting their intrinsic shape. Let µ be a point on M
Note that in a decomposable set ϕ(Z) a new valid de-
and let T M be the tangent space in µ. The exponential
µ compositionisobtainedbyaddingthesametangentvector
map projects a tangent vector v ∈ T M onto the mani-
µ to all v and subtracting it from all v , for any i ̸= j.
foldbymovingalongthegeodesicsegmentitdefines. For-
zi zj
However,wecanguaranteetheuniquenessofthefactoriza-
mally, if γ : [0,1] → M is the unique geodesic path
v tionbyimposingacenteringconstraint.
starting from γ (0) = µ with initial velocity γ˙ (0) = v,
v v
then Exp (v) := γ (1). This function is locally invert- Lemma 1. Let ϕ(Z) be a geodesically decomposable set.
µ v
ibleanditsinverseisthelogarithmicmapLog
µ
= Exp−
µ
1. Thenthere
(cid:80)
existuniquevectorsv zi ∈T µ Mforallz i ∈Z i
The exponential and logarithmic maps send straight lines suchthat zi∈Zi v zi = 0foralli = 1,...,sandEq.(3)
of the tangent plane into geodesic curves of the manifold, holds.
andvice-versa. Moreover,theyapproximatelypreservedis-
Foranintuitiveinterpretation,theintrinsicmeanµofade-
tancesbetweenelementsclosetothepointoftangencyµ:
composablesetcanbeseenasthecontextofthedecomposi-
d M (u,u′)≈||Log µ (u)−Log µ (u′)||, u,u′ ∈M (1) t o i f on th , e an p d rim ea i c ti h ve un c i o q n u c e e d p i t re z ct r io el n at v iv z e i r to ep µ re . se T n h t e s s t e he “u m n e iv a e n r i s n a g l
i
Notethatin(1)theequalityholdsifu=µoru′ =µ.When directions” are combined by addition on the tangent space
applyingthelogarithmicmaptoasetofpoints{u
i
}N
i=1
⊂ T
µ
M. Theexponentialmapoftheresultingtangentvector
M, the natural choice for the point of tangency µ is the definesthegeodesicsegmentonthemanifoldMfromµto
intrinsicmean,i.e.,theelementofMminimizingtheaver- thecorrespondingcompositemeaning(seeFig.2).
agesquareddistancetothegivenpoints. Inamoregeneral Our notion of geodesic decomposability is general and
definition, each point u (i = 1,...,N) is associated to a applicable to manifolds of any shape. It generalizes that
i
scalarweightw belongingtoaprobability-simplexvector of [67], which is equivalent to ours in the special case
i
∆ andthe(weighted)intrinsicmeanis: M = Rn,wheretheintrinsicmeanisthearithmeticmean,
N
and the exponential and logarithmic maps behave like the
N
(cid:88) identity function. Our manifold formalization agrees with
µ=argmin w d (u,u )2 (2)
i M i thefactthatlower-dimensionalsemanticsubspacesinCLIP
u∈M
i=1
latentspacearecapturedbysubmanifoldsbetterthanlinear
Thisdistance-minimizingelementµguaranteesthattheim- subspaces[46].
agesofthepointsthroughthelogarithmicmaparecentered Best decomposable approximation. Decomposable sets
(cid:80)
intheoriginofthetangentspace: w Log (u )=0. live in a lower dimension subspace of their manifold M.
i i µ i
24919

The dimension of Span({v } ) is indeed at most
zi zi∈Zi
|Z |−1 for all i = 1,...,s, implying the additive com-
i
binations of the primitive directions belong to a subspace
(cid:80)
of dimension at most (|Z |−1). This suggests that a
i i
genericsetofembeddings{u }isunlikelytobeperfectly
z
decomposable. We thus search for its best decomposable
approximation,thatistheset{u˜ }thatminimizestheerror
z
(cid:88)
d (u ,u˜ )2 (4)
M z z
z∈Z
Ingeneral,thisisahardproblemtosolve. Similarlytothe
standard solution to Principal Geodesic Analysis [15], we
use Eq. (1) to approximate the objective in the “simpler”
EuclideanspaceT M,andrewriteEq.(4)as:
µ
(cid:88)
||Log (u )−Log (u˜ )||2, (5)
µ z µ z
z∈Z
The solution to the approximate problem is obtained by
computingvectormeansinT M, asdescribedinthenext
µ Figure2. Sketchofourdecompositionmethod. (top-left)Each
proposition. For a fixed primitive concept z
i
∈ Z
i
, let conceptinZ={red,blue}×{□,△}isrepresentedbyk=5em-
Z(z i )={(z 1 ′,...,z r ′)∈Z|z i ′ =z i }denotethesliceofZ beddingsonamanifold.(bottom)Thesearemappedinthetangent
containingalltupleswiththei-thcomponentequaltoz . spacewhereoptimalprimitivedirectionsarecomputedasvector
i
meansandcombinedbyaddition. (top-right)Theobtainedcom-
Proposition 1. Given a set ϕ(Z) = {u |z ∈ Z} ⊂ M
z positionsaremappedbacktothemanifoldtoobtainadecompos-
withintrinsicmeanµ,theminimizationproblem ableapproximationoftheinputembeddings.
(cid:88)
argmin ||Log (u )−Log (u˜ )||2,
µ z µ z
{u˜z} z∈Z (6) animagefromthetuplez = (red,car)likelycontainsnon-
negligibleextrainformation,e.g. adriver,aroad,orablue
s.t.{u˜ }isgeodesicallydecomposable
z
sky in the background. This stems from the inherent am-
issolvedbyu˜ =Exp (v +···+v ),where biguityandnon-uniquenessofvisualsignals. Mostimpor-
z µ z1 zr
tantly,itisabsentintext,forwhichitiseasiertomanually
1 (cid:88) craftthestring“aredcar”ensuringnoextrainformation.
v = Log (u ) (7)
zi |Z(z i )| µ z Problem formulation. Since images contain noise in ad-
z∈Z(zi)
dition to represented concepts, we consider an input set
Moreover, (cid:80) v =0foralli=1,...,s. ϕ(Z × E) = {u |(z,e) ∈ Z × E} where each z is
zi∈Zi zi (z,e)
represented by k = |E| different image embeddings vary-
This result tells us that each vector v in the optimal
zi ing along the unknown noise dimension E. Also, differ-
decompositionisthetangentmeanofalltheinputcompo-
ent images may contain different amounts of noise. For
sitions including the primitive z . Moreover, the choice of
i each fixed z, we model this aspect with a probability dis-
the intrinsic mean as the point of tangency guarantees the
tribution {p } describing how well the elements in
(z,e) e∈E
uniquenessconstraintissatisfied(seeAppendixfordetails).
{u } representtheirlabelz. Inthissetting,wewant
(z,e) e∈E
3.3.DecomposableEmbeddingsofVisualInputs thedecomposableset{u˜ z } z∈Z minimizingtheobjective
(cid:88)
Our framework holds for arbitrary manifolds and for any p (z,e) d M (u (z,e) ,u˜ z )2, (8)
embedding map, hence being independent of the input (z,e)∈Z×E
modality. However, collections of natural visual data con-
where the importance given to the approximation error for
tain noise and are sparse. We account for these properties
each input embedding is weighted according to the noise
inourframeworkaspresentedinthefollowing.
distribution. The next result generalizes Proposition 1,
which addresses the special case k=1, and provides an
3.3.1 Removingnoisefromfiniteimagesets easy-to-computeapproximatesolutiontotheproblem.
We refer to noise as information carried by images in ad- Proposition2. Letp ,(z,e)∈Z ×E,benon-negative
(z,e)
(cid:80)
dition to the composite concept of interest. For example, scalarssuchthat p = 1foreachz ∈ Z,andlet
e∈E (z,e)
24920

ϕ(Z×E)={u |(z,e)∈Z×E}⊂Mbeasetofem- Note that the obtained decomposable set contains vector
(z,e)
beddingswithweightedintrinsicmeanµw.r.t. theweights representations of all the concepts in Z, including the un-
w =p / (cid:80) p . Theminimizationproblem: seenelementsofZ\Z′. Theformulationin(11)dealswith
(z,e) (z,e) (z,e) (z,e)
all aspects mentioned so far: the manifold M, noise, and
(cid:88)
argmin p ||Log (u )−Log (u˜ )||2, sparsity. Inthenextsection,weuseittoevaluatethecom-
(z,e) µ (z,e) µ z
{u˜z}
(z,e)∈Z×E
positionalstructureofrealvisualembeddings.
Noisedistribution. Thedescribedsetuprequiresthenoise
s.t.{u˜ }isgeodesicallydecomposable
z scores p . Given a collection of visual inputs T repre-
(9) (z,e)
senting each label z ∈ Z′ with k > 0 elements, a sim-
issolvedbyu˜ =Exp (v +···+v ),where z
z µ z1 zs plechoiceisusinguniformscoresp = 1/k . Alterna-
(z,e) z
1 (cid:88) (cid:88) tively,weproposeusingtheCLIPimage-to-textdistribution
v zi = |Z(z )| v z , v z = p (z,e) Log µ (u (z,e) ) p (z,e) = P((z,e)|y(z)),wherey(z)isatextpromptforla-
i
z∈Z(zi) e∈E belz ∈Z′. Thisisthesoftmaxofthescaledsimilarities
(10)
Moreover, (cid:80) zi∈Zi v zi =0foralli=1,...,s. P((z,e)|y(z))=
(cid:80)
ex
e
p
x
(
p
u
(
⊤ (
u
z,
⊤
e) u y
u
(z) /t
/
)
t)
(12)
Fig. 2 visualizes the decomposition procedure. Notice e (z,e) y(z)
that, using the same notation of the proposition, the vec-
Thetemperatureparametertislearnedduringtraining,but
torsv canbeseenasadenoisedtangentrepresentationof
z itcanbetweakedtosmoothorsharpenthedistribution.
the tuples in Z and the solution {u˜ } to the weighted
z z∈Z
optimizationproblemcorrespondstothedecomposableap- 4.ExperimentalValidation
proximationgivenbyProposition1appliedtothedenoised
embeddings {u := Exp (v )} . Indeed, these have We carry out experiments to analyze the decomposable
z µ z z∈Z
intrinsicmeanequaltotheweightedintrinsicmeanµ. propertiesofvisualembeddingsofVLMs. Whennotspeci-
fieddifferently,weusethepre-trainedCLIPViT-L/14[54].
Lemma 2. Using the notation of Proposition 2, the set WealsoconsiderCLIPResNet50[54]andSigLIP[72]. All
{u z :=Exp µ (v z )} z∈Z hasintrinsicmeanµ. considered models are from the OpenCLIP repository [8].
Weuseimageswithattribute-objectlabelstorepresentsets
3.3.2 Dealingwithsparsityinfiniteimagesets ofcompositeconceptsoftheformZ =Z ×Z .
attr obj
In this setup, we first assess the decomposable nature
Thepreviouslydescribedsetupassumesthateveryz ∈Z is
of small sets of embeddings inspecting their geometric ar-
representedbyk > 0images. Thisrequirementcanbetoo
rangementaccordingtoProposition2(Sec.4.1). Then,we
restrictiveinpractice,becausesomecombinationsofprim-
leverage the structured nature of the decomposed embed-
itives may not occur in real image collections. For exam-
dings and experiment on the tasks of compositional clas-
ple,ifZ = {red,blue}×{car,apple},therewillprobably
sification (Sec. 4.2) and group robustness (Sec. 4.3). Fi-
be no pictures of a (blue, apple). We refer to the absence
nally, we visualize the approximate decomposable embed-
ofcompositeconceptsassparsity. Oncemore,pleasenote
dings using a diffusion model (StableDiffusion v2.1 [57])
that sparsity is not an issue with text, since strings can be
withtheunCLIPtechnique[56](Sec.4.4).
manuallycraftedforanyz ∈Z.
Attribute-object decomposition. We usually deal with
ProblemFormulation. Ingeneral,inalabeledimagecol-
sparsecollectionsofvisualinputsT whereonlyasubsetZ′
lection, only a subset T ⊂ Z × E is available, and only
oflabelspresentatleastoneimage. Thus,wecomputethe
a subgroup Z′ ⊂ Z of composite concepts is represented
embedding decomposition according to Eq. (11): the opti-
by at least one element in T. In this scenario, we obtain
malvectorsu˜ =Exp (v +v )arethecombinations
a decomposable approximation of ϕ(T) by approximating oftheattribute (a d ,o ir ) ectionsv µ = a 1 o (cid:80) v andtheob-
the vector means in Eq. (10) with the mean of the avail- a Z′(a) o (a,o)
ableelements. Theonlyrequirementisthateveryprimitive ject directions v o = Z′ 1 (o) (cid:80) a v (a,o) , where the denoised
z i ∈ Z i (i = 1,...,s)appearsinatleastonetupleofZ′. representations v (a,o) , (a,o) ∈ Z′, are the mean tangent
Precisely, we first compute the weighted intrinsic mean µ vectors within pairs. For compositional classification and
ofϕ(T)withweightsw =p / (cid:80) p ,and grouprobustness,weusetheCLIPimage-to-textprobabil-
(z,e) (z,e) (z,e)∈T (z,e)
thenconsideru˜ =Exp (v +···+v ),where: ities as the noise distribution discussed in Sec. 3.3.2. We
z µ z1 zs
finetune the temperature parameter (see Appendix for de-
1 (cid:88) (cid:88) tails). Intheotherexperiments,weutilizeuniformscores.
v = v , v = p Log (u )
zi |Z′(z )| z z (z,e) µ (z,e) Datasets. We represent composite concepts with images
i
z∈Z′(zi) e∈Es.t.
fromthetrainingsetsofdiversecompositionaldatasets.We
(z,e)∈T
(11) test compositional classification on the typical benchmark
24921

datasetsUT-Zappos[69]andMIT-states[25]withthesplits k=1 k=5 k=30
from[53].UT-Zapposcontainsimagesofshoescenteredon
|     |     |     |     |     |     |     |     | 0.3 |     | 0.2 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.6
| awhitebackgroundallsharingthesameorientation. |     |     |     |     | There |     |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
0.0
are 12 object classes referring to the footwear type and 16 0.0 0.0
| attributecategoriesreferringtothematerial. |     |     |     | MIT-statesisa |     |     |     |      |     |      |     |
| ------------------------------------------ | --- | --- | --- | ------------- | --- | --- | --- | ---- | --- | ---- | --- |
|                                            |     |     |     |               |     |     |     | -0.3 |     | -0.2 |     |
collectionofnaturalobjectsindifferentstates. Thedataset 0.0 0.6 -0.4 0.0 0.4 -0.3 0.0 0.3
contains115attributecategoriesand245objectcategories, k=1 k=5 k=30
generatingalargenumberofpossiblecombinations.
0.2
| WetestgrouprobustnessontheWaterbirdsandCelebA |     |     |     |     |     |     |     | 0.2 |     |     | 0.2 |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
datasetswiththesplitsin[58]. Thesecontainobjectswith 0.0 0.0 0.0
|            |            |             |        |               |     |     |     | -0.2 | -0.2 |     | -0.2 |
| ---------- | ---------- | ----------- | ------ | ------------- | --- | --- | --- | ---- | ---- | --- | ---- |
| spuriously | correlated | attributes, | making | them suitable | for |     |     |      |      |     |      |
|            |            |             |        |               |     |     |     | 0.3  | 0.2  |     | 0.2  |
debiasing tasks. Waterbirds contains images of two bird -0.2 0.0 -0.3 0.0 -0.2 0.0
|          |                                            |     |     |     |     | 0.0 0.2 | -0.3 | 0.0 |          | 0.0 | -0.2 |
| -------- | ------------------------------------------ | --- | --- | --- | --- | ------- | ---- | --- | -------- | --- | ---- |
|          |                                            |     |     |     |     |         |      |     | 0.3 -0.2 | 0.2 |      |
| speciesZ | obj ={waterbird,landbird}ontwotypesofback- |     |     |     |     |         |      |     |          |     |      |
Figure3.(top)2-Dprojectionsofimageembeddingsrepresenting
| groundZ | ={land,water}.WeusetheversionofCelebA |                    |        |                |     |                                              |     |     |     |            |     |
| ------- | ------------------------------------- | ------------------ | ------ | -------------- | --- | -------------------------------------------- | --- | --- | --- | ---------- | --- |
|         | attr                                  |                    |        |                |     | the2×2compositelabelsoftheWaterbirdsdataset. |     |     |     | (bottom)3- |     |
| from    | [58] that                             | contains close-ups | photos | of celebrities | la- |                                              |     |     |     |            |     |
Dprojectionsofimageembeddingsrepresenting2×3composite
| beled | with hair-color | Z obj =                        | {blonde,dark} | and | gender |                                                         |        |          |                    |          |      |
| ----- | --------------- | ------------------------------ | ------------- | --- | ------ | ------------------------------------------------------- | ------ | -------- | ------------------ | -------- | ---- |
|       |                 |                                |               |     |        | labels with                                             | images | from the | UT-Zappos dataset. | Denoised | pair |
| Z     | ={male,female}. | Thedatadistributionoverthefour |               |     |        |                                                         |        |          |                    |          |      |
| attr  |                 |                                |               |     |        | representations(markedwithablackcontour)arecomputedwith |        |          |                    |          |      |
different groups is highly unbalanced in the train sets of k=1,5,30randomlyselectedimages.
thesetwodatasets,implyingspuriouscorrelations.
4.1.VisualizingCompositionalEmbeddings
|     |     |     |     |     |     | labelsinthetestsetareinZ′. |     |     | Intheclosed-worldsetting, |     |     |
| --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | ------------------------- | --- | --- |
thesetoftargetlabelsZtest
Weevaluatedthedecomposabilityoftheembeddingsfrom ⊂Zcontainsonlythepairsap-
a geometric perspective. We visualize lower-dimensional pearinginthedataset,whileintheopen-worldframework,
|     |             |                |         | {v    | },   | no prior knowledge |     | is assumed | and all | the attribute-object |     |
| --- | ----------- | -------------- | ------- | ----- | ---- | ------------------ | --- | ---------- | ------- | -------------------- | --- |
| PCA | projections | of the tangent | vectors | (a,o) | con- |                    |     |            |         |                      |     |
Ztest
sidering that the denoised representations u := combinations in = Z are considered. Both settings
(a,o)
Exp (v )aregeodesicallydecomposableifandonlyif require generalizing the prior knowledge about the primi-
µ (a,o)
|     |     |     |     |     |     | tivestounderstandtheunseencompositionsinZtest |     |     |     |     | \Z′. |
| --- | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | ---- |
theirtangentdirectionsaretheverticesofageometricshape
withparallelfaces. Forexample,decomposablesetsofsize Thisoperationisparticularlychallengingintheopen-world
|Z|=2×2and|Z|=2×3correspondtoaparallelogram scenario, where the more numerous novel compositions in
andatriangularprism,respectively. thetestsetareadistractionforthepredictor.
In Fig. 3 we show the (first row) 2-D projection of Our framework provides a straightforward solution to
image embeddings from the Waterbirds dataset, rep- the complex problem of compositional classification. The
resenting the four compositions of two attributes and geodesically decomposable set {u˜ } computed with
(a,o)
two objects, and the (second row) 3-D projection of the full train data T represents all the pairs, includ-
the two-by-three concepts in the set {leather,suede} × ing the unseen ones. Thus we classify an image x
{bootsankle,bootskneehigh,shoesflats} of the UT- as argmax u˜⊤ u . We evaluate the predic-
|     |     |     |     |     |     |     | (a,o)∈Ztest | (a,o) | x   |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ----- | --- | --- | --- |
Zapposdataset. Byincreasingthenumberk ofimagesper tion with the standard metrics [7, 53]: attribute accu-
pair, the noise is successfully removed and the resulting racy (ATTR), object accuracy (OBJ), best seen accuracy
representationsdefineshapeswithparallelfaces,indicating
|     |     |     |     |     |     | (SEEN), | best unseen | accuracy | (UNSEEN), | best | harmonic |
| --- | --- | --- | --- | --- | --- | ------- | ----------- | -------- | --------- | ---- | -------- |
approximate geodesical decomposability. This highlights mean(HM)betweentheseenandunseenaccuracyandarea
the importance of the denoising step and demonstrates undertheseen-unseencurve(AUC).
compositionalregularitiesofvisualembeddings. Baselines. OurprimarygoalistoexamineiftheGeodesi-
|     |     |     |     |     |     | cally Decomposable |     | Embeddings | (GDE) | approximating |     |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | ---------- | ----- | ------------- | --- |
4.2.CompositionalClassification
thetraindatacontainsemanticallymeaningfulinformation
WeperformcompositionalclassificationontheUT-Zappos about the composite concepts they represent. We evaluate
and MIT-states datasets using the decomposable approxi- the relative performance ρ (AUC ratio) obtained with de-
mation of the train data as classifiers. This task serves to composed embeddings w.r.t. the results achieved with the
standardzero-shotbaseline(CLIP)usingthefull-stateem-
evaluatethegeneralizationcapabilitiestowardsnovelcom-
(a,o)∈Ztest
positions of objects and states. Specifically, we follow the beddings (attribute-object labels are repre-
standard generalized zero-shot evaluation protocol in both sentedbythetextembeddingof“Animageofa{a}{o}”).
closed-worldandopen-worldscenarios[41]. We investigate the importance of complying with data
We compute decomposable embeddings on a subset geometry and compare with the Linearly Decomposable
Z′
⊂ Z of seen pairs from the training set, while not all Embeddings (LDE) proposed in [67], which we compute
24922

|     |     |     |     |     | CLOSED-WORLD |     |     |     |     |     | OPEN-WORLD |     |     |     |
| --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | ---------- | --- | --- | --- |
DATASET METHOD ATTR OBJ SEEN UNSEEN HM AUC ρ ATTR OBJ SEEN UNSEEN HM AUC ρ
CLIP[54] 24.1 58.3 11.9 45.7 15.3 4.4 - 18.8 57.4 11.9 23.8 12.0 2.3 -
LDE(TEXT)*[67] 24.1 58.8 11.9 45.7 14.1 4.0 92.4% 19.2 57.2 11.9 20.0 11.1 1.9 83.2%
UT-ZAPPOS GDE(TEXT) 25.3 60.0 17.0 48.2 18.9 6.4 146.6% 18.7 59.9 17.0 21.4 12.2 2.5 111.1%
LDE(IMAGE) 13.9 52.6 5.6 32.1 6.6 0.9 21.1% 9.8 48.0 5.6 14.9 2.3 0.2 8.9%
GDE(IMAGE) 36.3 64.1 31.4 55.9 29.3 13.9 317.9% 28.6 61.7 31.3 33.3 19.0 6.7 293.5%
CLIP[54] 33.0 52.1 30.6 45.3 26.3 11.1 - 15.6 47.7 30.6 8.3 8.4 1.7 -
LDE(TEXT)*[67] 30.6 51.2 24.7 43.0 21.9 8.2 73.4% 21.1 50.7 24.7 13.8 11.9 2.5 148.1%
MIT-STATES GDE(TEXT) 32.6 51.7 27.8 45.2 24.5 10.0 89.7% 21.3 49.9 27.8 13.0 12.1 2.6 158.5%
LDE(IMAGE) 15.3 30.5 15.0 20.9 11.1 2.0 18.4% 11.0 34.8 15.0 5.6 4.6 0.4 27.1%
GDE(IMAGE) 28.1 45.3 30.7 36.1 23.4 8.6 77.7% 18.5 43.6 29.7 8.5 9.3 1.8 106.6%
Table1. CompositionalclassificationresultsontheUT-ZapposandMIT-statesdatasets. Highestvalueswithinmodalityareinboldand
“*”indicatesthattheresultsofourimplementationareshown.
|     |     |     |     | CLIP,RN50 |     |     | CLIP,VIT-L/14 |     |     |     | SIGLIP,VIT-SO400M/14 |     |     |     |
| --- | --- | --- | --- | --------- | --- | --- | ------------- | --- | --- | --- | -------------------- | --- | --- | --- |
DATASET METHOD ATTR OBJ SEEN UNSEEN HM AUC ρ ATTR OBJ SEEN UNSEEN HM AUC ρ ATTR OBJ SEEN UNSEEN HM AUC ρ
CLIP[54] 24.4 40.5 4.8 41.9 6.7 1.5 - 24.1 58.3 11.9 45.7 15.3 4.4 - 52.5 74.4 44.9 68.1 39.2 24.6 -
UT-ZAPPOS LDE(IMAGE) 15.1 44.6 3.2 21.7 4.7 0.5 33.5% 13.9 52.6 5.6 32.1 6.6 0.9 21.1% 21.4 50.3 7.0 42.1 8.2 1.6 6.5%
GDE(IMAGE) 28.2 56.1 24.0 43.7 23.7 8.6 578.5% 36.3 64.1 31.4 55.9 29.3 13.9 317.9% 48.1 72.4 42.5 68.7 41.3 24.7 100.4%
CLIP[54] 26.6 42.3 23.5 35.2 19.4 6.2 - 33.0 52.1 30.6 45.3 26.3 11.1 - 45.9 61.2 43.8 58.1 39.7 22.2 -
MIT-STATES LDE(IMAGE) 13.7 25.1 10.6 16.2 8.0 1.1 17.8% 15.3 30.5 15.0 20.9 11.1 2.0 18.4% 18.7 34.6 18.6 27.2 14.6 3.6 16.1%
GDE(IMAGE) 20.8 34.4 18.9 25.1 14.2 3.4 54.9% 28.1 45.3 30.7 36.1 23.4 8.6 77.7% 32.3 50.3 36.8 40.8 27.3 11.9 53.6%
Table2.Ablationonbackbonearchitectureincompositionalclassification,closed-worldscenario.
bysettingM=Rn inourmethod,forbothtextandimage ased embeddings. We evaluate it on the group robustness
modalities. We indicate the modality by adding “(TEXT)” benchmarkpresentedin[58],whichrequiresclassifyingan
or “(IMAGE)” next to method names. Decomposed text- imagewithoutleveragingspuriouscorrelations. Inthisset-
embeddings are given by Proposition 1, as noise and spar- ting, a set of target classes Z has spurious correlations
obj
sitybelongonlytovisualdata. with a set of attributes Z attr due to the highly unbalanced
|          |                                            |     |     |     |     |     | datadistributionoverthegroupsinG |     |     |     |     | =Z   | ×Z  | . The |
| -------- | ------------------------------------------ | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | --- | ---- | --- | ----- |
| Results. | Table1reportstheresultsintheclosed-wordand |     |     |     |     |     |                                  |     |     |     |     | attr | obj |       |
open-world settings. In general, GDEs of visual data per- goal is to obtain an object classifier that does not exploit
formcloselytothezero-shotfull-statebaseline,demonstrat- spuriouscorrelations,improvingtheaverageaccuracyover
allthegroups(AVG)whilekeepingthe(GAP)ontheworst
ingtheyencodesemanticallymeaningfulinformationabout
the labels. Interestingly, on the UT-Zappos dataset, they groupaccuracy(WG)small. Weusetheobjectembeddings
improve the standard zero-shot approach by a large mar- u˜ :=Exp (v ), o∈Z computed with our method to
|         |           |      |        |          |         |           | o        | µ o       |            | obj |              |     |              |     |
| ------- | --------- | ---- | ------ | -------- | ------- | --------- | -------- | --------- | ---------- | --- | ------------ | --- | ------------ | --- |
|         |           |      |        |          |         |           | evaluate | the group | robustness |     | performance. |     | Intuitively, |     |
| gin. We | attribute | this | gap to | the fact | that in | UT-Zappos |          |           |            |     |              |     |              |     |
numerous representations are used for the computation of theseembedonlyobjectrepresentationsthatarenotcorre-
|                |     |           |            |        |     |            | latedwithattribute-relatedspuriousfeatures. |     |     |     |     |     | Wethuspre- |     |
| -------------- | --- | --------- | ---------- | ------ | --- | ---------- | ------------------------------------------- | --- | --- | --- | --- | --- | ---------- | --- |
| each primitive |     | direction | on average | (∼1400 | per | attribute, |                                             |     |     |     |     |     |            |     |
u˜⊤
∼1900perobject). Incontrast,theMIT-statesdatasetcon- dicttheobjectclassofanimagexasargmax u x .
|     |     |     |     |     |     |     |     |     |     |     |     |     | o∈Zobj | o   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
tainsnoisyannotations[2]andonaveragefewerrepresen-
|     |     |     |     |     |     |     | Baselines. | In addition |     | to the | zero-shot | CLIP | and | LDE |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | --- | ------ | --------- | ---- | --- | --- |
tationstocomputetheprimitives(∼260perattribute,∼120
method,weincludetwostandardbaselinesthatuselabeled
| per object). | The | decomposition |     | shows | robustness | to spar- |     |     |     |     |     |     |     |     |
| ------------ | --- | ------------- | --- | ----- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
data,namelyEmpiricalRiskMinimization(ERM)withlin-
sity, as indicated by the good open-world unseen accuracy earprobing[32]andERMwithfeatureadapters[16]. Fur-
ontheMIT-statesdatasets,forwhichseenpairsarelessthan
|               |      |                                   |                |      |            |         | thermore,  | we compare  |             | with     | two recent | methods |             | improv- |
| ------------- | ---- | --------------------------------- | -------------- | ---- | ---------- | ------- | ---------- | ----------- | ----------- | -------- | ---------- | ------- | ----------- | ------- |
| 5%ofthetotal. |      | LDEforvisualdataperformsmuchworse |                |      |            |         |            |             |             |          |            |         |             |         |
|               |      |                                   |                |      |            |         | ing the    | performance | of          | VLMs,    | Deep       | Feature | Reweighting |         |
| than GDE      | on   | both datasets                     | and            | when | ablating   | the VLM |            |             |             |          |            |         |             |         |
|               |      |                                   |                |      |            |         | (DFR) [30] | and         | Contrastive | Adapters |            | (CA)    | [73], and   | with    |
| backbone      | (see | Tab. 2).                          | This indicates |      | that image | embed-  |            |             |             |          |            |         |             |         |
FairerCLIP[10]thatperformsdebiasingofthefrozenCLIP
dingsarenotcloselylinearlydecomposable,andhighlights representationsinthetraining-freesettinglikeourmethod.
theimportanceofrespectingthedatageometrywhendeal-
ing with the extra complexity given by noise and sparsity. Results. Table3reportstheresultsonthegrouprobustness
|     |     |     |     |     |     |     | benchmarks. | GDE | considerably |     | outperforms |     | CLIP | and |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------------ | --- | ----------- | --- | ---- | --- |
Thisverifiesalsoonothergeometries(seeAppendix).
LDE,withanincreaseofWGaccuracyontheWaterbirds
|     |     |     |     |     |     |     | and CelebA | datasets | of  | about | 42 and | 21.8, | respectively. |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | --- | ----- | ------ | ----- | ------------- | --- |
4.3.GroupRobustness
|     |     |     |     |     |     |     | This indicates | that | our | method | effectively | decomposes |     | the |
| --- | --- | --- | --- | --- | --- | --- | -------------- | ---- | --- | ------ | ----------- | ---------- | --- | --- |
Pre-trained VLMs produce biased representations, leading embeddings of object and attribute primitives, producing
tozero-shotclassifiersnotrobusttogroupshifts[73]. Our robust classifiers. Notably, it achieves state-of-the-art WG
framework offers a training-free method to compute unbi- accuracy and smaller Gap compared to all other methods
24923

|     |     |     |     |     |     |     | a:rubber | a:leather |     | a:fauxfur |     | a:patentleather |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------- | --- | --------- | --- | --------------- |
WATERBIRDS CELEBA o:bootsmid-calf o:shoesoxfords o:bootsmid-calf o:sandals
| METHOD             |     | WG   | AVG  | GAP WG    | AVG  | GAP  |     |     |     |     |     |     |
| ------------------ | --- | ---- | ---- | --------- | ---- | ---- | --- | --- | --- | --- | --- | --- |
| CLIP[54]           |     | 44.4 | 84.3 | 40.0 74.4 | 86.9 | 12.4 |     |     |     |     |     |     |
| LDE(TEXT)[67]      |     | 64.6 | 88.0 | 23.3 83.9 | 85.5 | 1.6  |     |     |     |     |     |     |
| ERMLINEARPROBE[32] |     | 65.4 | 97.7 | 32.3 30.4 | 94.6 | 64.2 |     |     |     |     |     |     |
| ERMADAPTER[16]     |     | 76.1 | 97.8 | 21.7 40.0 | 94.3 | 54.3 |     |     |     |     |     |     |
DFR(SUBSAMPLE)[30] 58.8 95.9 37.1 78.7 91.8 13.1 a:diced a:tiny a:old a:inflated
|                   |     |      |      |           |      |     | o:fruit | o:tiger |     | o:clock |     | o:boat |
| ----------------- | --- | ---- | ---- | --------- | ---- | --- | ------- | ------- | --- | ------- | --- | ------ |
| DFR(UPSAMPLE)[30] |     | 66.5 | 96.4 | 29.8 83.9 | 91.2 | 7.2 |         |         |     |         |     |        |
| CA[73]            |     | 85.3 | 94.5 | 9.3 83.9  | 90.4 | 6.4 |         |         |     |         |     |        |
| FAIRERCLIP[10]    |     | 86.0 | 92.2 | 6.1 85.2  | 87.8 | 2.5 |         |         |     |         |     |        |
| GDE(IMAGE)        |     | 86.4 | 91.5 | 5.0 87.5  | 87.9 | 0.4 |         |         |     |         |     |        |
Table3.Comparisonofresultsongrouprobustness.
|     |     |     |     |     |     |     | o 1 : c a t | o 1 | : e l e p h ant | o 1 : t i g e | r   | o 1 : c a t |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --------------- | ------------- | --- | ----------- |
Waterbirds CelebA o : ti g er o : h o r s e o : b e a r o : b e ar
|     |     |     |     |     |     |     | 2   | 2   |     | 2   |     | 2   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 90  |     |     | 88  |     |     |     |     |     |     |     |     |     |
80
| 70  |     |     | 87  |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | WG  |     |     |     | WG  |     |     |     |     |     |     |
| 60  |     | Avg |     |     |     | Avg |     |     |     |     |     |     |
86
5 25 50 75 100 5 25 50 75 100 Figure5. Attribute-objectpairsgeneratedusingdecomposedem-
%supportsamples %supportsamples beddingswithStableDiffusionfortheUT-Zappos(firstrow)and
Figure4. DataefficiencyofGDEonthegrouprobustnessbench- MIT-states(secondrow)datasets.Thetwoleftmostlabelsareseen
| mark. Subsets | of the | full support | set | are sampled | keeping | the |     |     |     |     |     |     |
| ------------- | ------ | ------------ | --- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- |
pairs,whilethetworight-mostareunseenpairs.Wealsogenerate
| group ratios | fixed. The | shaded | confidence | band | shows | the stan- |     |     |     |     |     |     |
| ------------ | ---------- | ------ | ---------- | ---- | ----- | --------- | --- | --- | --- | --- | --- | --- |
object-objectpairs(thirdrow)blendinganimalspecies.
darddeviationoverfiveexperiments.
thatuselabeleddata,includingthetask-specificFairerCLIP. The modularity of the decomposable structures allows
|                                                     |     |     |     |     |     |     | representing | the composition |     | of two | objects | o ,o ∈ Z |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------ | --------------- | --- | ------ | ------- | -------- |
| GDE isthusaneffectivetraining-freesolutiontocompute |     |     |     |     |     |     |              |                 |     |        |         | 1 2 obj  |
unbiased embeddings. Furthermore, GDE demonstrates asExp (v +v ). Inspiredby[39],weexperimentwith
|     |     |     |     |     |     |     | µ        | o1 o2     |        |               |          |           |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------- | ------ | ------------- | -------- | --------- |
|     |     |     |     |     |     |     | blending | different | animal | species (Fig. | 5, third | row). The |
remarkabledata-efficiencyperformance,achievinghighre-
sults using limited amount of data (see Fig. 4). For exam- generatedimagesportrayphotorealisticcreatureswithfea-
ple, when using 25% of the full train samples (randomly tures of the two input species. This further highlights the
|     |     |     |     |     |     |     | power and | versatility | of  | the proposed | framework. | More |
| --- | --- | --- | --- | --- | --- | --- | --------- | ----------- | --- | ------------ | ---------- | ---- |
selectedkeepinggroupratiosfixed)theWGdecreasesless
than1%onboththeWaterbirdsandCelebAdatasets. generatedimagesareintheAppendix.
4.4.VisualizeDecomposableApproximations
5.Conclusion
| We visualize | the decomposed |     | visual | embeddings |     | using a |     |     |     |     |     |     |
| ------------ | -------------- | --- | ------ | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- |
diffusionmodelimplementingtheunCLIPmechanism(Sta- Weinvestigatedtheemergenceofcompositionalstructures
bleDiffusionv2.1)[56,57],trainedtoinverttheCLIPimage within the image latent space of vision-language models
encoderbyconditioningthegenerativeprocesswiththeim- anddemonstratedthatvisualembeddingsalsoexhibitade-
ageembeddings.Weinvertthedecomposablevectorsu˜ greeofcompositionalitysimilartothatoftextualrepresen-
(a,o)
obtainedinpreviousexperiments. Inthisway,wecanqual- tations. We proposed a training-free framework, Geodesi-
itativelyexaminetheinformationtheycontain. cally Decomposable Embeddings (GDE), designed to ad-
In Fig. 5 we show some generated images for object- dress the noisy and sparse nature of image data. GDE de-
attribute pairs where the attribute is not the most common composesvisualrepresentationsasageometry-awarecom-
stateoftheobject(i.e. weavoidcommonpairslike“green bination of optimal directions representing primitive con-
broccoli”or“bigelephant”), observingwhetherthegener- cepts. We demonstrated that these composed representa-
atedimagecorrectlyrepresentsthefulllabelandnotjustthe tionsencodecomplexconceptsandareeffectiveinseveral
attribute/object. The generated images well represent both tasks, including compositional classification and group ro-
theobjectandtheattributeofthelabel,withnodifferencein bustness. Notably, GDE presents more robust abilities to
thequalityoftheoutputsfromseen(twoleftmostcolumns) perform compositionality than existing approaches based
and unseen pairs (two rightmost columns). This empha- on linear decomposition of latent spaces, contributing to
sizesthegeneralizationpropertiesofourdecomposableim- higherresultsingrouprobustnessthanexistingtask-specific
age embeddings, with potential to be applied in practical methods.Webelievethisworkcontributestoachievingbet-
taskslikeaugmentingcompositionalsparsedatasets. terinterpretabilityandcontrollabilityofmodernVLMs.
24924

Acknowledgements. This work was sponsored by [14] MartinAFischlerandRobertAElschlager. Therepresenta-
the ERJU project, the EU Horizon project ELIAS tionandmatchingofpictorialstructures. IEEETransactions
(No. 101120237), Ministero delle Imprese e del oncomputers,100(1):67–92,1973. 1,2
| Made in | Italy | (IPCEI | Cloud DM | 27  | giugno | 2022 |               |           |     |             |         |          |     |
| ------- | ----- | ------ | -------- | --- | ------ | ---- | ------------- | --------- | --- | ----------- | ------- | -------- | --- |
|         |       |        |          |     |        |      | [15] P Thomas | Fletcher, |     | Conglin Lu, | Stephen | M Pizer, | and |
– IPCEI-CL-0000007), and the FAIR - Future AI Sarang Joshi. Principal geodesic analysis for the study of
Research (PE00000013), funded by NextGeneration nonlinear statistics of shape. IEEE TMI, 23(8):995–1005,
| EU. The          | authors | acknowledge |           | the CINECA       |              | award | 2004.         | 2,4                       |        |                 |         |             |         |
| ---------------- | ------- | ----------- | --------- | ---------------- | ------------ | ----- | ------------- | ------------------------- | ------ | --------------- | ------- | ----------- | ------- |
| under the        | ISCRA   | initiative  | for       | the availability |              | of    |               |                           |        |                 |         |             |         |
|                  |         |             |           |                  |              |       | [16] Peng     | Gao, Shijie               | Geng,  | Renrui Zhang,   | Teli    | Ma, Rongyao |         |
| high-performance |         | computing   | resources |                  | and support. |       |               |                           |        |                 |         |             |         |
|                  |         |             |           |                  |              |       | Fang,         | Yongfeng                  | Zhang, | Hongsheng       | Li, and | Yu          | Qiao.   |
|                  |         |             |           |                  |              |       | Clip-adapter: |                           | Better | vision-language | models  | with        | feature |
|                  |         |             |           |                  |              |       | adapters.     | IJCV,132(2):581–595,2024. |        |                 | 7,8     |             |         |
References [17] Songwei Ge, Shlok Mishra, Simon Kornblith, Chun-Liang
Li,andDavidJacobs.Hyperboliccontrastivelearningforvi-
[1] Jacob Andreas. Measuring compositionality in representa- InCVPR,pages6840–
sualrepresentationsbeyondobjects.
| tionlearning. |     | InICLR,2019. | 1   |     |     |     |            |     |     |     |     |     |     |
| ------------- | --- | ------------ | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
|               |     |              |     |     |     |     | 6849,2023. | 1   |     |     |     |     |     |
[2] Yuval Atzmon, Felix Kreuk, Uri Shalit, and Gal Chechik. [18] YunyeGong, SrikrishnaKaranam, ZiyanWu, Kuan-Chuan
A causal view of compositional zero-shot recognition. Peng,JanErnst,andPeterCDoerschuk. Learningcompo-
| NeurIPS,33:1462–1473,2020.                               |     |     | 7   |     |     |     |                                              |     |     |     |     |         |     |
| -------------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | --- | --- | ------- | --- |
|                                                          |     |     |     |     |     |     | sitionalvisualconceptswithmutualconsistency. |     |     |     |     | InCVPR, |     |
| [3] AlbertoBaldrati,LorenzoAgnolucci,MarcoBertini,andAl- |     |     |     |     |     |     | 2018.                                        | 1   |     |     |     |         |     |
bertoDelBimbo. Zero-shotcomposedimageretrievalwith [19] AlonHafri,E.J.Green,andChazFirestone.Compositional-
| textualinversion. |         | InICCV,2023. |           | 2        |       |      |                        |     |     |                                |     |     |     |
| ----------------- | ------- | ------------ | --------- | -------- | ----- | ---- | ---------------------- | --- | --- | ------------------------------ | --- | --- | --- |
|                   |         |              |           |          |       |      | ityinvisualperception. |     |     | BehavioralandBrainSciences,46: |     |     |     |
| [4] Moritz        | Bo¨hle, | Mario Fritz, | and Bernt | Schiele. | B-cos | net- | e277,2023.             | 2   |     |                                |     |     |     |
works: Alignment is all we need for interpretability. In [20] Irina Higgins, Nicolas Sonnerat, Loic Matthey, Arka Pal,
| CVPR,2022. |     | 2   |     |     |     |     |             |     |          | Bosˇnjak, |        |           |     |
| ---------- | --- | --- | --- | --- | --- | --- | ----------- | --- | -------- | --------- | ------ | --------- | --- |
|            |     |     |     |     |     |     | Christopher | P   | Burgess, | Matko     | Murray | Shanahan, |     |
[5] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. MatthewBotvinick,DemisHassabis,andAlexanderLerch-
Pix2video: Video editing using image diffusion. In ICCV, ner. Scan: Learninghierarchicalcompositionalvisualcon-
| 2023. | 2   |     |     |     |     |     | cepts. | InICLR,2018. |     | 1   |     |     |     |
| ----- | --- | --- | --- | --- | --- | --- | ------ | ------------ | --- | --- | --- | --- | --- |
[6] SarahChabalandVioricaMarian. Speakersofdifferentlan- [21] ZhiHou, XiaojiangPeng, YuQiao, andDachengTao. Vi-
guagesprocessthevisualworlddifferently. JournalofEx- sualcompositionallearningforhuman-objectinteractionde-
perimentalPsychology:General,144(3):539–550,2015. 2 tection. InECCV,2020. 2
[7] Wei-LunChao,SoravitChangpinyo,BoqingGong,andFei
|     |     |     |     |     |     |     | [22] Zhi Hou, | Baosheng |     | Yu, and Dacheng | Tao. | Discovering |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | -------- | --- | --------------- | ---- | ----------- | --- |
Sha. An empirical study and analysis of generalized zero- human-object interaction concepts via self-compositional
shot learning for object recognition in the wild. In ECCV, learning. InECCV,2022. 2
pages52–68.Springer,2016. 6 [23] Cheng-YuHsieh,JieyuZhang,ZixianMa,AniruddhaKem-
[8] MehdiCherti,RomainBeaumont,RossWightman,Mitchell bhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable
Wortsman,GabrielIlharco,CadeGordon,ChristophSchuh- benchmarksforvision-languagecompositionality. NeurIPS,
| mann,LudwigSchmidt,andJeniaJitsev.Reproduciblescal- |     |     |     |     |         |     | 2023.                                                | 2   |     |     |              |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | ------- | --- | ---------------------------------------------------- | --- | --- | --- | ------------ | --- | --- |
| inglawsforcontrastivelanguage-imagelearning.        |     |     |     |     | InCVPR, |     |                                                      |     |     |     |              |     |     |
|                                                     |     |     |     |     |         |     | [24] DrewAHudsonandChristopherDManning.Compositional |     |     |     |              |     |     |
| pages2818–2829,2023.                                |     |     | 5   |     |         |     |                                                      |     |     |     |              |     |     |
|                                                     |     |     |     |     |         |     | attentionnetworksformachinereasoning.                |     |     |     | InICLR,2018. |     | 1   |
[9] KonradCzechowski,TomaszOdrzygo´z´dz´,MarekZbysin´ski, [25] PhillipIsola, JosephJ.Lim, andEdwardH.Adelson. Dis-
| Michał                    | Zawalski, | Krzysztof | Olejnik,                    | Yuhuai | Wu, | Łukasz |                                                     |     |     |     |     |     |     |
| ------------------------- | --------- | --------- | --------------------------- | ------ | --- | ------ | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                           |           |           |                             |        |     |        | coveringstatesandtransformationsinimagecollections. |     |     |     |     |     | In  |
| Kucin´ski,andPiotrMiłos´. |           |           | Subgoalsearchforcomplexrea- |        |     |        |                                                     |     |     |     |     |     |     |
|                           |           |           |                             |        |     |        | CVPR,2015.                                          |     | 6   |     |     |     |     |
soningtasks. NeurIPS,2021. 1 [26] ChaoJia,YinfeiYang,YeXia,Yi-TingChen,ZaranaParekh,
[10] SepehrDehdashtian,LanWang,andVishnuNareshBoddeti. HieuPham, QuocLe, Yun-HsuanSung, ZhenLi, andTom
| Fairerclip: |     | Debiasingzero-shotpredictionsofclipinrkhss. |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Duerig.Scalingupvisualandvision-languagerepresentation
InICLR,2024. 2,7,8 learningwithnoisytextsupervision. InICML,pages4904–
| [11] KaranDesai,MaximilianNickel,TanmayRajpurohit,Justin |     |           |             |           |     |        | 4916,2021.                                              | 1   |     |     |     |     |     |
| -------------------------------------------------------- | --- | --------- | ----------- | --------- | --- | ------ | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| Johnson,                                                 | and | Shanmukha | Ramakrishna | Vedantam. |     | Hyper- |                                                         |     |     |     |     |     |     |
|                                                          |     |           |             |           |     |        | [27] ShyamgopalKarthik,KarstenRoth,MassimilianoMancini, |     |     |     |     |     |     |
bolic image-text representations. In ICML, pages 7694– and Zeynep Akata. Vision-by-language for training-free
7731.PMLR,2023. 1 compositionalimageretrieval. InICLR,2024. 2
[12] JacobFeldman. Probabilisticoriginsofcompositionalmen- [28] Keizo Kato, Yin Li, and Abhinav Gupta. Compositional
talrepresentations. PsychologicalReview,131(3):599–624, learningforhumanobjectinteraction. InECCV,2018. 2
| 2024. | 2   |     |     |     |     |     | [29] BahjatKawar,ShiranZada,OranLang,OmerTov,Huiwen |     |     |     |     |     |     |
| ----- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
[13] Pedro Felzenszwalb, David McAllester, and Deva Ra- Chang,TaliDekel,InbarMosseri,andMichalIrani. Imagic:
manan. A discriminatively trained, multiscale, deformable Text-based real image editing with diffusion models. In
| partmodel. |     | InCVPR,2008. | 1,2 |     |     |     | CVPR,2023. |     | 2   |     |     |     |     |
| ---------- | --- | ------------ | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
24925

[30] PolinaKirichenko,PavelIzmailov,andAndrewGordonWil- [47] Bjorn Ommer and Joachim Buhmann. Learning the com-
son. Lastlayerre-trainingissufficientforrobustnesstospu- positionalnatureofvisualobjectcategoriesforrecognition.
riouscorrelations. InICLR,2023. 7,8 IEEETPAMI,32(3):501–516,2009. 1,2
[31] Jayanth Koushik, Hiroaki Hayashi, and Devendra Singh [48] BjornOmmerandJoachimMBuhmann. Learningthecom-
Sachan. Compositional reasoning for visual question an- positionalnatureofvisualobjects. InCVPR,2007. 2
swering. InICML,2017. 1,2 [49] DimPPapadopoulos,YoussefTamaazousti,FerdaOfli,In-
[32] AnanyaKumar,AditiRaghunathan,RobbieMatthewJones, gmarWeber, andAntonioTorralba. Howtomakeapizza:
Tengyu Ma, and Percy Liang. Fine-tuning can distort pre- Learningacompositionallayer-basedganmodel. InCVPR,
| trained | features | and underperform |     | out-of-distribution. |     | In  | 2019. | 1,2 |     |     |     |     |
| ------- | -------- | ---------------- | --- | -------------------- | --- | --- | ----- | --- | --- | --- | --- | --- |
ICLR,2022. 7,8 [50] BarbaraParteeetal. Lexicalsemanticsandcompositional-
[33] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew ity.Aninvitationtocognitivescience:Language,1:311–360,
| Wallingford, |     | Aditya | Sinha, Vivek | Ramanujan, |     | William | 1995. | 1   |     |     |     |     |
| ------------ | --- | ------ | ------------ | ---------- | --- | ------- | ----- | --- | --- | --- | --- | --- |
Howard-Snyder,KaifengChen,ShamKakade,PrateekJain, BarbaraHPartee.Compositionalityinformalsemantics:Se-
[51]
et al. Matryoshka representation learning. NeurIPS, 35: lectedpapers. JohnWiley&Sons,2008. 1
30233–30249,2022. 1 [52] PramudithaPerera,MatthewTrager,LucaZancato,Alessan-
[34] KevinJ.Lande. Compositionalityinperception: Aframe- dro Achille, and Stefano Soatto. Prompt algebra for task
work. WIREsCognitiveScience,15(6):e1691,2024. 2 composition. arXivpreprintarXiv:2306.00310,2023. 1,2
[35] MarthaLewis,NihalNayak,PeilinYu,JackMerullo,Qinan [53] SenthilPurushwalkam,MaximilianNickel,AbhinavGupta,
Yu,StephenBach,andElliePavlick. DoesCLIPbindcon- andMarc’AurelioRanzato. Task-drivenmodularnetworks
cepts? probingcompositionalityinlargeimagemodels. In forzero-shotcompositionallearning. InICCV,2019. 6
Findings of the Association for Computational Linguistics: [54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
EACL2024,pages1487–1500,2024. 2 Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
[36] XilaiLi,XiSong,andTianfuWu. Aognets: Compositional AmandaAskell,PamelaMishkin,JackClark,etal. Learn-
grammaticalarchitecturesfordeeplearning.InCVPR,2019. ingtransferablevisualmodelsfromnaturallanguagesuper-
| 1            |     |         |         |          |      |          | vision. | InICML,pages8748–8763.PMLR,2021. |     |     |     | 1,2,5,7, |
| ------------ | --- | ------- | ------- | -------- | ---- | -------- | ------- | -------------------------------- | --- | --- | --- | -------- |
| [37] Yong-Lu | Li, | Yue Xu, | Xiaohan | Mao, and | Cewu | Lu. Sym- | 8       |                                  |     |     |     |          |
metryandgroupinattribute-objectcompositions. InCVPR, [55] Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham,
| 2020. | 2   |     |     |     |     |     | andAjanthanThalaiyasingam. |     |     | Acceptthemodalitygap:An |     |     |
| ----- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | ----------------------- | --- | --- |
[38] Renjie Liao, Alex Schwing, Richard Zemel, and Raquel explorationinthehyperbolicspace.InCVPR,pages27263–
| Urtasun. |     | Learning | deep parsimonious |     | representations. |     | 27272,2024. |     | 1   |     |     |     |
| -------- | --- | -------- | ----------------- | --- | ---------------- | --- | ----------- | --- | --- | --- | --- | --- |
NeurIPS,2016. 1 [56] AdityaRamesh,PrafullaDhariwal,AlexNichol,CaseyChu,
[39] Giorgio Longari, Lorenzo Olearo, Simone Melzi, Rafael andMarkChen. Hierarchicaltext-conditionalimagegener-
Pen˜aloza,andAlessandroRaganato. Howtoblendconcepts ationwithcliplatents. arXivpreprintarXiv:2204.06125,1
| indiffusionmodels.arXivpreprintarXiv:2407.14280,2024. |     |     |     |     |     |     | (2):3,2022. |          | 5,8     |            |         |         |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----------- | -------- | ------- | ---------- | ------- | ------- |
| 8                                                     |     |     |     |     |     |     | [57] Robin  | Rombach, | Andreas | Blattmann, | Dominik | Lorenz, |
[40] XiaochengLu,SongGuo,ZimingLiu,andJingcaiGuo.De- PatrickEsser,andBjo¨rnOmmer.High-resolutionimagesyn-
composedsoftpromptguidedfusionenhancingforcompo- thesiswithlatentdiffusionmodels. InCVPR,pages10684–
| sitionalzero-shotlearning. |     |     | InCVPR,2023. |     | 2   |     | 10695,2022. |     | 5,8 |     |     |     |
| -------------------------- | --- | --- | ------------ | --- | --- | --- | ----------- | --- | --- | --- | --- | --- |
[41] MassimilianoMancini,MuhammadFerjadNaeem,Yongqin [58] ShioriSagawa,PangWeiKoh,TatsunoriB.Hashimoto,and
Xian, and ZeynepAkata. Openworld compositionalzero- Percy Liang. Distributionally robust neural networks. In
| shotlearning. |     | InCVPR,pages5222–5230,2021. |     |     |     | 2,6 | ICLR,2020. |     | 6,7 |     |     |     |
| ------------- | --- | --------------------------- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- |
[42] MassimilianoMancini,MuhammadFerjadNaeem,Yongqin [59] KuniakiSaito,KihyukSohn,XiangZhang,Chun-LiangLi,
Xian, and Zeynep Akata. Learning graph embeddings for Chen-Yu Lee, Kate Saenko, and Tomas Pfister. Pic2word:
openworldcompositionalzero-shotlearning. IEEETPAMI, Mapping pictures to words for zero-shot composed image
| 46(3):1545–1560,2022. |     |     |     |     |     |     | retrieval. |     | InCVPR,2023. | 2   |     |     |
| --------------------- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------ | --- | --- | --- |
[43] IshanMisra,AbhinavGupta,andMartialHebert. Fromred [60] SaschaSaralajew,LarsHoldijk,MaikeRees,EbubekirAsan,
CVPR,
wine to red tomato: Composition with context. In andThomasVillmann.Classification-by-components:Prob-
2017. abilistic modeling of reasoning over a set of components.
[44] Tushar Nagarajan and Kristen Grauman. Attributes as op- NeurIPS,2019. 1,2
erators: factorizingunseenattribute-objectcompositions. In [61] Ju¨rgenSchmidhuber.Towardscompositionallearningindy-
ECCV,2018. 1,2 namicnetworks. TechnicalUniversityofMunich(Technical
ReportFKI-129-90),1990.
| [45] NihalV.Nayak,PeilinYu,andStephenBach. |     |     |     |     |     | Learningto |     |     |     | 1   |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
composesoftpromptsforcompositionalzero-shotlearning. [62] ZhangzhangSiandSong-ChunZhu. Learningand-ortem-
InICLR,2023. 1,2 platesforobjectrecognitionanddetection.IEEETPAMI,35
[46] James Oldfield, Christos Tzelepis, Yannis Panagakis, Mi- (9):2189–2205,2013. 1
halisNicolaou,andIoannisPatras.Partsofspeech–grounded [63] Austin Stone, Huayan Wang, Michael Stark, Yi Liu, D
subspaces in vision-language models. NeurIPS, 36:2700– Scott Phoenix, and Dileep George. Teaching composition-
| 2724,2023. |     | 2,3 |     |     |     |     | alitytocnns. |     | InCVPR,2017. | 1,2 |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------------ | --- | --- | --- |
24926

[64] FuwenTan,SongFeng,andVicenteOrdonez. Text2scene:
Generating compositionalscenes fromtextual descriptions.
InCVPR,2019. 2
[65] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet
Singh, Adina Williams, Douwe Kiela, and Candace Ross.
Winoground:Probingvisionandlanguagemodelsforvisio-
linguisticcompositionality. InCVPR,2022. 2
[66] ShengbangTong,ZhuangLiu,YuexiangZhai,YiMa,Yann
LeCun, and Saining Xie. Eyes wide shut? exploring the
visualshortcomingsofmultimodalllms. InCVPR,2024. 2
[67] MatthewTrager,PramudithaPerera,LucaZancato,Alessan-
dro Achille, Parminder Bhatia, and Stefano Soatto. Lin-
earspacesofmeanings: compositionalstructuresinvision-
languagemodels. InICCV,pages15395–15404,2023. 1,2,
3,6,7,8
[68] Jianyu Wang and Alan L Yuille. Semantic part segmenta-
tion using compositional model combining shape and ap-
pearance. InCVPR,2015. 1
[69] AronYuandKristenGrauman. Fine-grainedvisualcompar-
isonswithlocallearning. InCVPR,pages192–199, 2014.
6
[70] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri,
Dan Jurafsky, and James Zou. When and why vision-
languagemodelsbehavelikebags-of-words,andwhattodo
aboutit? InICLR,2023. 2
[71] Tian Yun, Usha Bhalla, Ellie Pavlick, and Chen Sun. Do
vision-languagepretrainedmodelslearncomposableprimi-
tiveconcepts? TMLR,2023. 1
[72] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and
LucasBeyer. Sigmoidlossforlanguageimagepre-training.
InICCV,2023. 5
[73] Michael Zhang and Christopher Re´. Contrastive adapters
forfoundationmodelgrouprobustness.NeurIPS,35:21682–
21697,2022. 7,8
[74] Yizhen Zhang, Minkyu Choi, Kuan Han, and Zhongming
Liu. Explainablesemanticspacebygroundinglanguageto
vision with cross-modal contrastive learning. Advances in
Neural Information Processing Systems, 34:18513–18526,
2021. 1
[75] Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N
Metaxas,andJianRen.Sine:Singleimageeditingwithtext-
to-imagediffusionmodels. InCVPR,2023. 2
[76] BoZhao,BoChang,ZequnJie,andLeonidSigal. Modular
generativeadversarialnetworks. InECCV,2018. 2
24927