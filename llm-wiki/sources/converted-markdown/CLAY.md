|     | CLAY:    | Conditional        | Visual     | Similarity   | Modulation |     |     |
| --- | -------- | ------------------ | ---------- | ------------ | ---------- | --- | --- |
|     |          | in Vision-Language |            | Embedding    | Space      |     |     |
|     | SohwiLim |                    | LeeHyoseok | JungjoonPark | Tae-HyunOh |     |     |
KAIST
|     | [Species] |     |     |     | [Category] |     |     |
| --- | --------- | --- | --- | --- | ---------- | --- | --- |
[Location]
[Location]
[Color]
[Action]
Query Image
| Conditioned      |     |     |     |     |     | Conditioned      |     |
| ---------------- | --- | --- | --- | --- | --- | ---------------- | --- |
| retrieved images |     |     |     |     |     | retrieved images |     |
Figure1.Ourproposedconcept-basedconditionalimageretrievalmethodretrievesimagesfocusingonthesemanticaspectsspecifiedbythe
textcondition.Givenaqueryimage,ourmethodadaptivelycomputesconditionedsimilaritybymodulatingthesimilarityspacetoalignwith
variousconditions,e.g.,species,location,action,category,andcolor.
Abstract
works. Ourcodeanddatasetswillbepubliclyreleasedupon
acceptance.
Humanperceptionofvisualsimilarityisinherentlyadaptive
| andsubjective,dependingontheusers’interestsandfocus. |                 |         |                      | 1.Introduction |     |     |     |
| ---------------------------------------------------- | --------------- | ------- | -------------------- | -------------- | --- | --- | --- |
| However, most                                        | image retrieval | systems | fail to reflect this |                |     |     |     |
flexibility,relyingonafixed,monolithicmetricthatcannot Intheeraofunprecedenteddatascale,humansseektoeffi-
incorporatemultipleconditionssimultaneously. Toaddress cientlyandaccuratelyidentifytheinformationofinterestin
this,weproposeCLAY,anadaptivesimilaritycomputation theoverwhelmingdataflow. Withinthiscontext,retrieval
method that reframes the embedding space of pretrained servesasacomputationalmechanismthatenablesustofind
Vision-LanguageModels(VLMs)asatext-conditionalsim- whattrulymattersandwhatweneed. Inthecomputervi-
| ilarityspacewithoutadditionaltraining. |     |     | Thisdesignsep- |                    |                |          |                |
| -------------------------------------- | --- | --- | -------------- | ------------------ | -------------- | -------- | -------------- |
|                                        |     |     |                | sion area, despite | the remarkable | advances | in large-scale |
arates the textual conditioning process and visual feature image retrieval tasks, most approaches still rely on static
extraction,allowinghighlyefficientandmulti-conditioned definitionsofvisualsimilarity[6,12,35,47]. Bycontrast,
retrievalwithfixedvisualembeddings. Wealsoconstructa humansperceivevisualsimilarityinaflexibleandadaptive
syntheticevaluationdatasetCLAY-EVAL,forcomprehen- manner,selectivelyfocusingondifferentaspectsofanimage
siveassessmentunderdiverseconditionedretrievalsettings. dependingonauser’sinterest. Forexample,onemayseek
Experimentsonstandarddatasetsandourproposeddataset the same object itself, while another may want to see the
showthatCLAYachievesstate-of-the-artretrievalaccuracy overallmoodoftheimage. Thisunderscorestheneedfor
andnotablecomputationalefficiencycomparedtoprevious conditionalretrievalsystemsthatcanreflectvarioushuman

|     |     |     |     |     |     |     | 𝐼   | 𝐼   | 𝐼   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Table1.Ourcontextualconditionalsimilaritycomputationmethod ! # $ Textual	Subspaces
provideshighretrievalaccuracywhilemaintainingefficiencyunder
diverseconditions.Wefurthersupportamulti-conditionedretrieval
| scheme, whereas |     | previous | works | [17, 39] | do not. | † denotes a |     |     |     |     |     |     |     |
| --------------- | --- | -------- | ----- | -------- | ------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
VLM
| modified | version | of the | GeneCIS, | where | database | features are |     |                |     |                                         |     |     |     |
| -------- | ------- | ------ | -------- | ----- | -------- | ------------ | --- | -------------- | --- | --------------------------------------- | --- | --- | --- |
|          |         |        |          |       |          |              |     | Vision	Encoder |     | [Species][Action][Color][Age][Location] |     |     |     |
additionallyincorporatedwithconditionaltext.
𝑣 #
|                |               |         | GeneCIS† |     |           | CLAY |     |     |     |     |            | 𝑣 "  |      |
| -------------- | ------------- | ------- | -------- | --- | --------- | ---- | --- | --- | --- | --- | ---------- | ---- | ---- |
|                |               | GeneCIS |          |     | FocalLens |      |     | 𝑣 ! |     |     | 𝑣 "        | #    | "    |
|                |               |         |          |     |           |      |     |     | 𝑣 $ |     | $          |      | 𝑣    |
|                |               |         | ✗        | ✗   | ✗         | ✓    |     |     |     |     | "          |      | $    |
| T r a i n in g | -f r e e      |         |          |     |           |      |     |     |     | "   | 𝑣          |      |      |
|                |               |         |          |     |           |      |     |     |     | 𝑣   | #          |      | 𝑣 "  |
| R e t r ie va  | l a c c uracy |         | ✗        | ✓   | ✓         | ✓    |     |     |     | !   | [S pecies] |      | !    |
[Location]
| Dynamicefficiency |     |     | ✓   | ✗   | ✗   | ✓   |                   |     |     |     |                        |     |     |
| ----------------- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | ---------------------- | --- | --- |
|                   |     |     | ✗   | ✗   | ✗   | ✓   | Cosine	Similarity |     |     |     | Conditional	Similarity |     |     |
Multi-condition
Figure2.IllustrationoftheconceptofCLAY.Ourmethodadap-
tivelycomputesconditionalsimilaritybetweenimagesbymodulat-
attention-relatedvisualcues.
ingtheoriginalsimilarityspaceintoaconditionalsimilarityspace
Inresponsetothisdemand,priorstudieshaveexplored
withintherepresentationspaceofVLMs.
twomaindirectionsdependingonwhichaspectsofthequery
| imagearefocusedormodifiedduringretrieval: |     |     |     |     |     | onefocuses |     |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
ousmethodsdedicatedtosingle-conditionsettings[17,39].
onparticularattributeswithinthequeryimage[17,39,40],
Duetoalackofastandardbenchmarkofmulti-conditional
| while the | other | aims to | change | specific | visual | contexts of |           |       |              |             |     |            |         |
| --------- | ----- | ------- | ------ | -------- | ------ | ----------- | --------- | ----- | ------------ | ----------- | --- | ---------- | ------- |
|           |       |         |        |          |        |             | retrieval | task, | we construct | a synthetic |     | evaluation | dataset |
queryimagetomatchthetargetimages[1,14,18,20,37,48].
|     |     |     |     |     |     |     | containing |     | diverse human | and | object images |     | and various |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------- | --- | ------------- | --- | ----------- |
Althoughthelatterdirectionhasbeenactivelystudied,the
conceptualconditionpairs,allowingevaluationsundermulti-
formerdirection,whichseekstocapturespecificattributes
|     |     |     |     |     |     |     | conditionscenarios. |     | Evaluationondiverserealandsynthetic |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ----------------------------------- | --- | --- | --- | --- |
inadynamicmanner,remainsrelativelyunderexploredde-
datasetsshowsthatourmethodachievesstrongretrievalac-
| spite its | importance. | Most | studies | have | addressed | the lat- |        |     |                       |     |             |     |             |
| --------- | ----------- | ---- | ------- | ---- | --------- | -------- | ------ | --- | --------------------- | --- | ----------- | --- | ----------- |
|           |             |      |         |      |           |          | curacy | and | notable computational |     | efficiency, |     | pushing the |
terdirectioninatraining-basedmanner,wheremodelsare
Pareto-frontofthetrade-offbetweenperformanceandeffi-
trainedtoperformretrievalundergivenconditionsbylearn- ciencyinpractice. Wesummarizeourcontributionsas:
| ing condition-specific |     |     | representations |     | [17, 39]. | However, |     |     |     |     |     |     |     |
| ---------------------- | --- | --- | --------------- | --- | --------- | -------- | --- | --- | --- | --- | --- | --- | --- |
• Weproposeanefficient,training-free,state-of-the-artcon-
thesemethodsrequirenotonlycomputationalresourcesfor
ditionalvisualsimilaritycomputationmethodthatcancon-
thetrainingstagebutalsopairedquery-targetimagedatafor
textuallyadapttovariousconditionswithoutrecomputing
| eachcondition[39], |     | whichlimitstheirworkingregimeto |     |     |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
databasefeaturesatinference.
| closed-setconditions. |     |     | Moreover,tomaintainretrievalaccu- |     |     |     |     |     |     |     |     |     |     |
| --------------------- | --- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
racy,thesetrainedconditionalfeatureextractorsincurcom- • Our method supports multi-conditioned image retrieval
putationaloverheadatinferencetimebecauseallfeaturesof scenarios, enabling flexible retrieval settings beyond
single-conditionsettings.
databaseimagesneedtoberecomputedthroughthemodel
fromscratch,whenevertheuserconditionchanges[17,40]. • Weconstructasyntheticevaluationdatasetthatcontains
These limitations hinder the applicability and capacity of diverse humans and objects with conceptual condition
conditionalretrievalsystemsindiverse,real-worldscenarios. pairstofacilitatetheevaluationunderdiverseconditional
| To overcome |     | these | challenges, | we  | propose | CLAY, a |     |     |     |     |     |     |     |
| ----------- | --- | ----- | ----------- | --- | ------- | ------- | --- | --- | --- | --- | --- | --- | --- |
retrievalscenarios.
| training-free | and | adaptive | conditional |     | similarity | computa- |     |     |     |     |     |     |     |
| ------------- | --- | -------- | ----------- | --- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- |
tionmethodthattransformsthesimilaritycomputationspace
2.RelatedWork
| of pretrained | Vision-Language |     |     | Models | (VLMs) | [35, 47], |     |     |     |     |     |     |     |
| ------------- | --------------- | --- | --- | ------ | ------ | --------- | --- | --- | --- | --- | --- | --- | --- |
accordingtousers’interest. Bydecouplingthecondition- Image-to-image retrieval is a practical and fundamental
ingprocedurefromvisualfeatureextraction,ourapproach problem in the computer vision area [16]. Conventional
contextuallymodulatesthesimilaritycomputationspaceof methodsonimage-to-imageretrievalmainlyfocusedonmea-
VLMsintoaconditionalsimilarityspace. Thiseliminates suringhowvisuallysimilartwoimagesare,relyingonhand-
theneedtocompletelyrecomputeembeddingsofdatabase craftedlocaldescriptors[8,27]. Withtheemergenceofdeep
imagesundervaryinguserconditionsbykeepingtheoriginal learning,subsequentstudiesleveragedConvolutionalNeural
visualembeddingsfixed.Toachievethis,weconstructacon- Networks[5,31,32,43]toextracthigher-levelsemanticfea-
ditionalsimilarityspacewithintheVLMs’representation turesbeyondlow-levelvisualcues. However,relyingsolely
spacethatrespectsitsunderlyingnon-Euclideangeometry, onstaticvisualfeaturesimilarityfromsuchmodelscanbe
and define its textual concept subspace leveraging a sub- insufficient, as human preferences may differ on multiple
space–constructionproceduresimilarto[9,29]. Buildingon contextualfactors. Thislimitationsuggeststhattheretrieval
this,wefurtherdemonstratethatourmethodiseasilyexten- systemsneedtobedesignedtoconsiderfurtherconditions,
sibletomulti-conditionalretrievalscenarios,unlikeprevi- i.e., conditional image retrieval. In this work, we explore

conditionalimageretrieval,whichconsidersadaptivealign- 3.1.ProblemDefinition
mentwithuserintentions.
|     |     |     |     |     |     |     |     | Previous | conditional |     | retrieval | methods | [17, 39] | typically |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | --------- | ------- | -------- | --------- |
Vision-LanguageModels(VLMs)aretrainedtoembedim- computecondition-basedvisualsimilaritybymodifyingim-
ageandtextmodalitiesintoasharedembeddingspace[35, agerepresentationswithtextconditionsthroughafeature
|           |      |         |              |     |              |     |        | modulator. |     | In contrast, | we  | reformulate | the visual | similar- |
| --------- | ---- | ------- | ------------ | --- | ------------ | --- | ------ | ---------- | --- | ------------ | --- | ----------- | ---------- | -------- |
| 47]. They | also | provide | semantically |     | well-aligned |     | visual |            |     |              |     |             |            |          |
representationsbyleveraginglanguageasasemanticrefer- ity space itself, allowing conditional retrieval with fixed
ence[42]. Recentstudies[9,29]havefurtherexploredthe embeddings. Wefirstcategorizetheconditionalsimilarity
structureoftheVLMembeddingspacetoenabletext-guided computationmethodsintotwoformulations: symmetricand
controlovervisualembeddingsforpersonalizedimagegen- asymmetricmethodsbasedonhowthetextconditioncis
erationandediting. Forinstance,Dorfmanetal.[9]project incorporated into the image representations. Given query
visualembeddingintoatext-basedsubspace,therebyextract- imageI q anddatabaseimagesI d ,theconditionalsimilarity
ingthetext-relatedvisualcontextsusefulforcompositional between I and I can be computed following these two
|                  |     |                                     |     |     |     |     |     |               |     | q   | d   |     |     |     |
| ---------------- | --- | ----------------------------------- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
| imagegeneration. |     | However,theseapproachesprimarilyfo- |     |     |     |     |     | formulations: |     |     |     |     |     |     |
cusonaligningvisualrepresentationswithtextualsemantics,
|     |     |     |     |     |     |     |     |     | csim |     |     | (cid:0) |     | (cid:1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | ------- | --- | ------- |
rather than modeling their relative relationships. We fur- sym (I q ,I d |c)=d m(I q ,c), m(I d ,c) , (1)
|     |     |     |     |     |     |     |     |     |     |     |     | (cid:0) | (cid:1) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | --- |
therdevelopthisideatoeffectivelycapturetherelationships csim (I ,I |c)=d m(I ,c), I , (2)
|              |          |     |        |             |           |     |        |     |     | asym | q d |     | q d |     |
| ------------ | -------- | --- | ------ | ----------- | --------- | --- | ------ | --- | --- | ---- | --- | --- | --- | --- |
| among visual | features |     | within | the textual | subspace, |     | taking |     |     |      |     |     |     |     |
whered(·,·)denotesasimilarityfunctionsuchascosinesim-
| into account | the | hyperspherical |     | nature | of  | the embedding |     |          |                               |     |     |     |                |     |
| ------------ | --- | -------------- | --- | ------ | --- | ------------- | --- | -------- | ----------------------------- | --- | --- | --- | -------------- | --- |
|              |     |                |     |        |     |               |     | ilarity. | Them(I,c)representsamodulator |     |     |     | thatintegrates |     |
manifoldforaccuraterelationshipmodeling.
|                            |            |        |      |                         |                |     |      | inputimageI  |     | andthegivenconditionc. |     |                        |     |     |
| -------------------------- | ---------- | ------ | ---- | ----------------------- | -------------- | --- | ---- | ------------ | --- | ---------------------- | --- | ---------------------- | --- | --- |
| Conditionalimageretrieval. |            |        |      | Withtheincreasingimpor- |                |     |      |              |     |                        |     |                        |     |     |
|                            |            |        |      |                         |                |     |      | Symmetricvs. |     | asymmetric.            |     | Comparedtothesymmetric |     |     |
| tance of                   | retrieving | images | that | satisfy                 | user-specified |     | con- |              |     |                        |     |                        |     |     |
case(Eq.1),theasymmetriccomputation(Eq.2)doesnot
ditions, prior studies have proposed utilizing text condi- forwardthedatabaseimagesthroughthemodulatormwith
| tions and | incorporating |     | visual | features | to  | focus | on spe- |                   |     |     |                                  |     |     |     |
| --------- | ------------- | --- | ------ | -------- | --- | ----- | ------- | ----------------- | --- | --- | -------------------------------- | --- | --- | --- |
|           |               |     |        |          |     |       |         | thetextcondition. |     |     | ArepresentativemethodGeneCIS[39] |     |     |     |
cific attributes [17] in the image or to modify particular followstheasymmetricformulation,whereonlyqueryfea-
| contexts   | [1, 26,  | 44, 48]. | As         | an early | work,          | Conditional |        |                                                   |                 |     |        |             |          |         |
| ---------- | -------- | -------- | ---------- | -------- | -------------- | ----------- | ------ | ------------------------------------------------- | --------------- | --- | ------ | ----------- | -------- | ------- |
|            |          |          |            |          |                |             |        | tures                                             | are conditioned |     | on the | text input. | However, | despite |
| Similarity | Networks | [40]     | alleviated |          | the limitation |             | of the |                                                   |                 |     |        |             |          |         |
|            |          |          |            |          |                |             |        | therelativeefficiencyofthisasymmetricformulation, |                 |     |        |             |          | the     |
singlesimilaritymetricofembeddingmethodsbyemploy- database features will remain independent of the given
ingconditioningmaskstoselectcondition-specificembed-
|     |     |     |     |     |     |     |     | condition. |     | Consequently, |     | the retrieved | results | rely on the |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------- | --- | ------------- | ------- | ----------- |
ding dimensions. To specify this task, GeneCIS [39] in- condition-agnostic representations, potentially leading to
troducedabenchmarkthatcategorizestextconditionsinto suboptimalretrievalperformancewhenguidingthesearch.
| two types: | focus | on and | change, | and | proposed | a   | training- |     |     |     |     |     |     |     |
| ---------- | ----- | ------ | ------- | --- | -------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
Tofurthersupportthis,weexperimentallycomparetheper-
based method that fine-tunes the image encoder and con- formancebyapplyingboththesymmetricandasymmetric
| dition feature | modulator |     | from | the | paired dataset. |     | Follow- |     |     |     |     |     |     |     |
| -------------- | --------- | --- | ---- | --- | --------------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
formulationstoGeneCIS,inSec5.
ingthis,mostsubsequentstudieshavetackledthechange
|     |     |     |     |     |     |     |     | Designofmodulatorm. |     |     |     | Existingmethods[17,39]design |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | ---------------------------- | --- | --- |
type,retrievingimagesthatdifferonlyinspecificattributes
themodulatormthroughtheneuralnetworkthatincorpo-
| from the      | query | image,  | known | as      | Composed | Image   | Re-     |                          |     |     |     |                             |     |     |
| ------------- | ----- | ------- | ----- | ------- | -------- | ------- | ------- | ------------------------ | --- | --- | --- | --------------------------- | --- | --- |
|               |       |         |       |         |          |         |         | ratestextandimageinputs. |     |     |     | Inotherwords,themodulatoris |     |     |
| trieval (CIR) | [1,   | 14, 18, | 21,   | 26, 44, | 48].     | Another | line of |                          |     |     |     |                             |     |     |
coupledwiththevisualfeatureextractionandthecondition-
work[17]aimstoretrieveimagesbyfocusingonspecific
ingprocessthroughaneuralnetworkcouplertoobtaincon-
attributeswithinaqueryimage,leveragingVLMsormulti-
|                                  |     |     |     |     |     |              |     | ditionalvisualfeatures. |     |     | Thisdesignrequiresafullforward |     |     |     |
| -------------------------------- | --- | --- | --- | --- | --- | ------------ | --- | ----------------------- | --- | --- | ------------------------------ | --- | --- | --- |
| modalLargeLanguageModels(mLLMs). |     |     |     |     |     | Weexplorethe |     |                         |     |     |                                |     |     |     |
passthroughthenetworkforeachconditiontoobtainthecor-
latterdirection,whichisrelativelyunderexploredcompared
|     |     |     |     |     |     |     |     | respondingconditionalfeatures. |     |     |     | Inthesymmetricform,it |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --------------------- | --- | --- |
toCIR,andleveragethejointembeddingspaceofVLMsto
mayleadtocomputationaloverheadandlimititspracticality
effectivelyperformconditionalimageretrieval.
|     |     |     |     |     |     |     |     | foradaptiveconditionalretrieval. |     |     |     | Incontrast,wedecouple |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | --------------------- | --- | --- |
theconditioningprocessfromthevisualfeatureextraction
withinthemodulatorm,andproposeaconditioningprocess
3.CLAY:ConditionalSimilarityModulation thatdoesnotrelyontheinputvisualfeature. Weachieve
thisbyproposingtheconditionalsimilarityspacemodula-
We propose a conditional visual similarity computation tion scheme, which directly leverages the visual features
methodleveragingVLMs[35,47]thatconsiderhyperspheri- fromapretrainedVLMandprojectsthemintoaconditional
calmanifold. Webeginbydescribingtheproblemdefinition similarityspace. Thisdecouplingenablesefficientadaptive
ofconditionalretrievalinSec.3.1,followedbytheformula- retrievalinthesymmetricformbyeliminatingtheneedto
tionofourconditionalsimilarityinSec.3.2. re-encodethedatabasefeaturesforvaryingconditions.

(a)  Textual subspace construction (b) Inference: Conditional Similarity on textual subspace
|     |     |     | 𝒯𝝁! |     |     |     |     |     |     |     |     | 𝒯𝝁! |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
𝝁!
Map Query/DB
Condition	features Map to Construct Query/DBfeatures Align 𝝁 & to𝝁 ! ProjectQuery/DB
!	…𝑡 ! $ tangent space 𝒯 v 	/	v ,…,v with 𝐻(⋅) to tangent space 𝒯
| 𝑇 ! | = 𝑡 " # |     |     | 𝝁!  | textual subspace |     | '   | 1 # |     |     |     |     | 𝝁! with matrix	𝐏 | 𝐜   |
| --- | ------- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- |
: Textual / Visual features : Mean of features : Tangent space of 𝝁 : Textualsubspace  : Queryfeature
𝒄
Figure3.Conditionalsimilaritycomputationpipeline.(a)Givenacondition,weconstructthemanifold-awaretextualsubspacewiththe
conditiontextfeaturesinadvance,andgeneratethecondition-awareprojectionmatrixP .(b)Atinference,wecomputetheconditional
c
similaritybetweenthequeryanddatabaseimagesbyprojectingthevisualfeaturesontothetextualsubspacewithP c .
3.2.VisualSimilarityModulation centratedwithinaconicalregion,indicatingthatmostem-
|     |     |     |     |     |     |     |     | beddings | lie closer | to  | a common | mean | direction. | Hence, |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | -------- | ---- | ---------- | ------ |
Asmentioned,sinceweutilizetextmodalityfordescribing
weleveragethevalidityofthelocaltangentspaceapproxi-
thecondition,weadoptpretrainedVLMstoleveragetheir
|                      |                 |     |                               |                |     |           |     | mationunderthisgeometricproperty. |       |                             |          |         | Formally,anypoint |       |
| -------------------- | --------------- | --- | ----------------------------- | -------------- | --- | --------- | --- | --------------------------------- | ----- | --------------------------- | -------- | ------- | ----------------- | ----- |
| jointembeddingspace. |                 |     | VLMstypicallyconsistofavision |                |     |           |     |                                   |       |                             |          |         |                   |       |
|                      |                 |     |                               |                |     |           |     | x ∈ Sd−1                          | \{−µ} | can                         | bemapped | ontothe | tangent           | space |
| encoderf             | andtextencoderf |     |                               | .ForeachqueryI |     | ,database |     |                                   |       |                             |          |         |                   |       |
|                      | I               |     |                               | T              |     | q         |     | T ={x∈Rd                          |       | :x⊤µ=0}vialogarithmmap[15]: |          |         |                   |       |
µ
| image                        | I d and | condition      | c,  | we define | v q =           | f I (I q | ), v d = |     |     |                 |     |     |        |     |
| ---------------------------- | ------- | -------------- | --- | --------- | --------------- | -------- | -------- | --- | --- | --------------- | --- | --- | ------ | --- |
| f (I                         | ),andt  | = f (c),wheref |     | andf      | denotethevision |          |          |     |     |                 |     |     |        |     |
| I d                          |         | T              |     | I         | T               |          |          |     |     |                 |     |     | θ      |     |
| andtextencoder,respectively. |         |                |     |           |                 |          |          |     | log | (x):=(x−µ(x⊤µ)) |     |     | ,      |     |
|                              |         |                |     |           |                 |          |          |     |     | µ               |     |     | sin(θ) |     |
(3)
Themainideaofconditionalsimilarityspacemodulation
|     |     |     |     |     |     |     |     |     |     | θ =arccos(x⊤µ). |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --- | --- | --- |
istoprojectvisualfeaturesontothetextualsubspace,which
capturesthecondition-awarerelationshipsamongvisualfea-
|        |                                          |     |     |     |     |     |     | Weconsiderthenormalizedmeanµ                        |     |     |     | oftextfeaturesT |     | asa |
| ------ | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --------------- | --- | --- |
| tures. | Toachievethis,wederiveaprojectionmatrixP |     |     |     |     |     | by  |                                                     |     |     |     | c               |     | c   |
|        |                                          |     |     |     |     |     | c   | referencepointfordefiningthetangentspace,andmaptext |     |     |     |                 |     |     |
performingsingularvaluedecomposition(SVD)onthetext
featuresfromtheunithyperspheremanifoldtothetangent
| conditionfeaturematrix. |          |          | Specifically, |                   | followingprevious |     |         |          |                   |     |     |                   |     |     |
| ----------------------- | -------- | -------- | ------------- | ----------------- | ----------------- | --- | ------- | -------- | ----------------- | --- | --- | ----------------- | --- | --- |
|                         |          |          |               |                   |                   |     |         | spaceofµ | withlogarithmmap. |     |     | Then,weapplySVDon |     |     |
| works                   | [9, 29], | we first | generate      | condition-related |                   |     | textual |          | c                 |     |     |                   |     |     |
thesemappedtextfeatures:
| prompts   | through | Large-Language |        |                         | Model (LLM) |     | with the |     |     |     |         |            |               |     |
| --------- | ------- | -------------- | ------ | ----------------------- | ----------- | --- | -------- | --- | --- | --- | ------- | ---------- | ------------- | --- |
| template: | a       | photo          | of{c}. | Thegeneratedtextprompts |             |     |          |     |     |     |         |            |               |     |
|           |         |                |        |                         |             |     |          |     |     |     | (cid:2) | (tc)···log | (tc) (cid:3)⊤ |     |
arethenencodedthroughatextencoderf andtheirtext log (T c )= log ,
|     |     |     |     |     |     | T   |     |     | µ c |     | µ c | 1   | µ c n | (4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
embeddingstareconcatenatedtoformthetextfeaturema- log (T )=UΣV⊤.
µ c
| trixT | = [tc,...,tc]⊤ |     | ∈   | Rn×d. | Previousworksdirectly |     |     |     |     |     | c   |     |     |     |
| ----- | -------------- | --- | --- | ----- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|       | c 1            |     | n   |       |                       |     |     |     |     |     |     |     |     |     |
applySVDonthesetextfeaturestogeneratetheprojection Weutilizethetop-krightsingularvectorstoconstructtex-
matrix;however,thisoperationassumesEuclideanstructure tualsubspace,definedasspan(V ),andobtaintheprojec-
k
andthereforeignorestheintrinsicgeometryoftheembed- tionmatrixP =V V ⊤. Byprecomputingthesetextual
|     |     |     |     |     |     |     |     |     | c   | k   | k   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
dingmanifold. Incontrast,sinceVLMembeddingslieona subspacesforeachcondition(i.e.,projectionmatrices),we
unithypersphere,thisEuclideanassumptionfailstocapture onlyneedtomodulatetheconditionalsimilarityspaceusing
theirunderlyinggeometry. precomputedprojectionmatricesatinference,accordingly.
Specifically,weprojectthequeryanddatabasevisualfea-
| Manifold-aware |     | textual | subspace |     | construction. |     | To ad- |     |     |     |     |     |     |     |
| -------------- | --- | ------- | -------- | --- | ------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
turesfrompre-trainedVLMsintothespacethatsatisfiesthe
dressthis,wetakeintoaccountthehypersphericalnatureof
desiredconditionalrelationships,withoutre-encoding.
theembeddingmanifoldinVLMstoaccuratelymodelthe
relationshipandderiveamanifold-awareprojectionmatrix Inference. Now we have a projection matrix P which
c
P . Weconstructthetextualsubspaceaslocallygeodesic projects features onto the textual subspace within the tan-
c
submanifold rather than linear subspace, to alleviate the gentspaceofµ . Ourultimategoalistomodeltheintra-
c
distortion from Euclidean projection similar to previous relationship among visual features within this condition-
works[2,11]. Undermildcurvatureassumptions,thissub- awaretextualsubspace. Asmentionedabove,thelogarithm
manifoldcanbelocallyapproximatedbyitstangentspace mapontothetangentspaceisvalidundermildcurvatureas-
atareferencepoint. Previousworks[10,25]observethat sumptions. However,duetotheconiceffect[10,25],naive
theembeddinggeometryofeachmodalitytendstobecon- projectionofvisualfeaturescannotmaintainthisproperty.

|     |     |     |     |     |     |     |     | (a) CLAY-Object |     | (b) CLAY-Human |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | -------------- | --- | --- |
Tomitigatethis,weapplyrotationH(·)toalignthemean
| of the database |     | visual | features | µ with | the | mean | of text |     |     |     |     |     |
| --------------- | --- | ------ | -------- | ------ | --- | ---- | ------- | --- | --- | --- | --- | --- |
vd
| features | µ , without |     | altering | the intra-relationship |     |     | among |     |     |     |     |     |
| -------- | ----------- | --- | -------- | ---------------------- | --- | --- | ----- | --- | --- | --- | --- | --- |
c
| the visual | features. |     | We then | apply the | logarithm |     | map to |     |     |     |     |     |
| ---------- | --------- | --- | ------- | --------- | --------- | --- | ------ | --- | --- | --- | --- | --- |
maptherotatedvisualfeaturesH(v)ontothetangentspace
atµ ,andsubsequentlyprojectthemontothetextualsub-
c
| spacewithprojectionmatrixP |     |     |     | . Finally,similartostandard |     |     |     |     |     |     |     |     |
| -------------------------- | --- | --- | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
c
similaritycomputation,wecomputethesimilaritybetween
| thequeryfeaturev    |     |     | andthedatabasefeaturesv |             |     | byusing    |         |     |     |     |     |     |
| ------------------- | --- | --- | ----------------------- | ----------- | --- | ---------- | ------- | --- | --- | --- | --- | --- |
|                     |     | q   |                         |             |     | d          |         |     |     |     |     |     |
| cosine similarity.  |     | The | proposed                | conditional |     | similarity | be-     |     |     |     |     |     |
| tweenthequeryimageI |     |     | q andthedatabaseimageI  |             |     |            | d under |     |     |     |     |     |
theconditioncisasfollows:
|                                              | m    | (v,c):=P |        | log         | (H(v)), |             |        |     |     |     |     |     |
| -------------------------------------------- | ---- | -------- | ------ | ----------- | ------- | ----------- | ------ | --- | --- | --- | --- | --- |
|                                              |      | CLAY     |        | c µ         |         |             |        |     |     |     |     |     |
|                                              |      |          |        | c           |         |             | (5)    |     |     |     |     |     |
| csim                                         |      | (I ,I    | |c):=d | (cid:0) m(I | ,c),m(I | ,c) (cid:1) | ,      |     |     |     |     |     |
|                                              | CLAY | q        | d      |             | q       | d           |        |     |     |     |     |     |
| whered(·,·)correspondstothecosinesimilarity. |      |          |        |             |         |             | Thede- |     |     |     |     |     |
tailedpipelineisprovidedinthesupplementarymaterial.
|     |     |     |     |     |     |     |     | Figure4. | OurCLAY-EVALdatasetstatistics. |     | Weconstructa |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------------------------------ | --- | ------------ | --- |
4.OurCLAY-EVALDataset
syntheticdatasetwithdiverseconditionannotations,consistingof
WeprovideanovelsyntheticevaluationdatasetCLAY-EVAL (a)objectentityand(b)humanentity. Forboth,theleftcolumn
for conditional image retrieval. To evaluate our context- showssampleimagesdemonstratingvisualnaturalness,andthe
rightcolumnvisualizesthedistributionsofkeyattributes,showing
| aware conditional |     | image | retrieval | setting, | each | image | in  |     |     |     |     |     |
| ----------------- | --- | ----- | --------- | -------- | ---- | ----- | --- | --- | --- | --- | --- | --- |
diversity.Percentagesaretruncatedtoonedecimalplaceandanno-
thedatasetshouldbeannotatedwithmultiplelabels,allow-
tationtextlabelsareabbreviated.Seesupplementarymaterialfor
| ing flexible | clustering |     | under | different | conditions. |     | While |     |     |     |     |     |
| ------------ | ---------- | --- | ----- | --------- | ----------- | --- | ----- | --- | --- | --- | --- | --- |
fulldetails.
| several | datasets | are | classified | with | particular | conditions, |     |     |     |     |     |     |
| ------- | -------- | --- | ---------- | ---- | ---------- | ----------- | --- | --- | --- | --- | --- | --- |
theirscalabilityislimitedbecausereal-worldimageswere humanimages,anddetailedlistsofallattributeinstancesand
curationprocessareprovidedinthesupplementarymaterial.
| manually                                           | annotated |     | [23], they | only                    | have a | single | condi- |               |     |     |     |     |
| -------------------------------------------------- | --------- | --- | ---------- | ----------------------- | ------ | ------ | ------ | ------------- | --- | --- | --- | --- |
| tion [30,                                          | 33, 41,   | 45, | 46], or    | they include            | only   | simple | sim-   |               |     |     |     |     |
| ulated3Dgeometricobjects(e.g.,cone,sphere,andcube) |           |     |            |                         |        |        |        | 5.Experiments |     |     |     |     |
| withasimplebackground[19].                         |           |     |            | Toaddressthis,wefurther |        |        |        |               |     |     |     |     |
5.1.ExperimentalSetup
buildasyntheticdatasetusingapowerfulopen-sourcedif-
|     |     |     |     |     |     |     |     | Evaluation | datasets | and metric. | We evaluate | the condi- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ----------- | ----------- | ---------- |
fusionmodel[3],whosereliabilityallowsforthecontrolled
generationofhigh-qualityanddiverseimagesamples. tional retrieval performance on a wide range of datasets,
includingbothreal-worlddatasetsandoursyntheticdataset.
Toconstructanevaluationdatasetalignedwithourobjec-
Forreal-worlddatasets,weutilizefine-grainedimageclas-
| tives,weestablishthreekeydesignprinciples: |     |     |     |     |     | disentangle- |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- |
ment,compositionality,andnaturalness,inspiredby Lietal. sification datasets [4, 22, 28, 30, 33, 41, 45], and further
|     |     |     |     |     |     |     |     | conduct | on Stanford40 | [46] with | the human annotated | la- |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------- | --------- | ------------------- | --- |
[24]. Basedonthese,westructureourdatasetintotwomain
entities, object and human. Each dataset consists of core belsfrom[23]. Inthesefine-grainedclassificationdatasets,
attributes,whichserveastheprimaryqueryconditions,and we consider each category as condition, for example, in
|           |             |       |         |     |        |          |     | Stanford40,theconditionwesupposedisaction. |     |     |     | Forthe |
| --------- | ----------- | ----- | ------- | --- | ------ | -------- | --- | ------------------------------------------ | --- | --- | --- | ------ |
| diversity | attributes, | which | control | for | visual | variance | and |                                            |     |     |     |        |
bias. Wedefinethreecoreattributesforeachentity,namely syntheticevaluation,weemployCLEVR4[19]withthecon-
|     |     |     |     |     |     |     |     | ditions | shape, color, | texture, | and count, | as well |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------------- | -------- | ---------- | ------- |
category,sub-category,andcolorfortheobjectentity,and
age,actionandbackgroundforthehumanentity. as our generated synthetic dataset. We split each dataset
Wegenerateimageswithstructuredpromptsderivedfrom into query and database with 1:9 ratio, and in the evalu-
|               |               |     |     |               |         |     |          | ation with | subcategory | condition, | we report | the aver- |
| ------------- | ------------- | --- | --- | ------------- | ------- | --- | -------- | ---------- | ----------- | ---------- | --------- | --------- |
| all attribute | combinations. |     |     | For stringent | quality |     | control, |            |             |            |           |           |
we first exclude contradictory combinations via schema- agedperformancefromeachcategorizeddatabase. Forthe
evaluationmetric,weadoptthestandardmetricmeanAver-
levelfiltering,andsubsequentlymanuallyremovegenerated
imagesexhibitinglowtext-visualalignment. agePrecision(mAP)followingconventionalimageretrieval
AsshowninFig.4,oursyntheticdatasetcoversawide tasks[13,34,36],unlessotherwisenoted.
rangeofinstancesandpairedimages,enablingcomprehen- Competing methods. To the best of our knowledge,
sivecontext-awareconditionalretrievalevaluation. Ourfinal GeneCIS [39] is the only work that is closely related to
syntheticdatasetconsistsof7,325objectimagesand6,745 ourproblem. Asmentionedabove,theyfollowtheasymmet-

Table2.QuantitativecomparisonofmeanAveragePrecision(mAP)onsingleconditionaldatasets.Weevaluatesingleconditional
retrievalperformanceacrossvariousdatasets,whereeachconditioncorrespondstoaspecificattribute(e.g.,action,cat species,
flower type).(a)Real-worlddatasetsand(b)syntheticdatasetsarepresentedseparately.WecomparethebaselineVision-Language
Models(VLMs)andcompetingmethodsthatsupportconditionalretrieval.Forafaircomparison,weadditionallyreportGeneCIS†,the
extendedresultsofGeneCISusingasymmetricconditionalsimilaritycomputation,whereastheoriginalGeneCISadoptsanasymmetric
| formulation.The | best | and | second-best | resultsarehighlighted. |     |     |     |                                 |     |     |     |     |
| --------------- | ---- | --- | ----------- | ---------------------- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- |
|                 |      |     |             | Stanford40             |     |     |     | Fine-grainedImageClassification |     |     |     |     |
Method
Action Location Mood Cat Dog Instrument Flower Car Aircraft Food
| CLIP-B   |     |     | 43.0 | 47.0 | 53.0 | 37.5 | 37.9 | 27.1 | 70.1 | 30.5 | 20.9 | 47.4 |
| -------- | --- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| SigLIP-B |     |     | 54.8 | 52.7 | 56.4 | 55.9 | 59.3 | 40.6 | 86.6 | 64.6 | 46.7 | 61.3 |
| GeneCIS  |     |     | 50.0 | 50.9 | 51.8 | 24.7 | 24.7 | 24.7 | 58.5 | 16.9 | 17.2 | 44.8 |
| GeneCIS† |     |     | 63.5 | 52.3 | 57.3 | 32.7 | 33.9 | 42.0 | 72.3 | 29.8 | 23.5 | 50.8 |
InstructBLIP 63.1 54.4 60.3 50.6 56.8 37.1 76.5 27.6 15.7 50.6
Ours(CLIP-B) 66.0 55.4 57.9 72.7 79.4 58.0 80.4 51.2 28.4 58.8
Ours(SigLIP-B) 66.2 59.5 58.7 82.1 84.7 63.4 92.7 78.0 57.9 66.1
(a)real-worlddatasets.
|     |     |     |     | Clever4 |     |     | CLAY-Object |     |     | CLAY-Human |     |     |
| --- | --- | --- | --- | ------- | --- | --- | ----------- | --- | --- | ---------- | --- | --- |
Method
Shape Color Texture Count Color Category Subcategory Age Action Background
| CLIP-B   |     | 61.7 |     | 19.9 | 18.1 13.9 | 12.9 | 68.9 | 42.4 |     | 50.4 54.4 |     | 41.8 |
| -------- | --- | ---- | --- | ---- | --------- | ---- | ---- | ---- | --- | --------- | --- | ---- |
| SigLIP-B |     | 77.7 |     | 20.0 | 18.6 13.0 | 14.8 | 69.6 | 62.0 |     | 47.7 56.7 |     | 53.6 |
| GeneCIS  |     | 47.9 |     | 16.6 | 16.0 11.5 | 12.5 | 80.1 | 38.4 |     | 45.1 62.4 |     | 46.2 |
| GeneCIS† |     | 72.6 |     | 17.4 | 18.1 12.7 | 11.9 | 84.3 | 39.6 |     | 44.1 71.3 |     | 48.9 |
InstructBLIP 83.4 27.4 22.8 17.9 14.4 75.6 61.4 45.0 72.2 73.7
Ours(CLIP-B) 72.1 67.9 22.2 21.8 47.8 93.4 65.8 71.3 81.3 75.5
Ours(SigLIP-B) 88.6 73.2 26.1 22.7 63.3 94.3 81.9 61.0 80.4 81.9
(b)syntheticdatasets.
Table3.QuantitativecomparisonofRecall@1,2,3onGeneCIS Table4.QuantitativecomparisonofmAPonmulti-conditional
benchmark.Weevaluatetheretrievalperformanceonthe“Focus datasetsCLAY.Weprovidetheconditioncombinationssuchas
Attribute”subset,whereeachqueryhasasingleground-truthmatch. color and category.WereportmAPasanevaluationmet-
Recall@1,@2,and@3arereportedastheperformancemetric.For ric.Thebestresultsarehighlighted.
CIRmethods,wereportthevaluesprovidedfromMagicLens[48].
|              |     |          |     |          |          |     |                | CLAY-Object        |             |                   | CLAY-Human   |            |
| ------------ | --- | -------- | --- | -------- | -------- | --- | -------------- | ------------------ | ----------- | ----------------- | ------------ | ---------- |
| Method       |     | Recall@1 |     | Recall@2 | Recall@3 |     | Method         |                    |             |                   |              |            |
|              |     |          |     |          |          |     |                | Color              | Color       | Age               | Age          | Action All |
|              |     |          |     |          |          |     |                | Category           | Subcategory | Action Background |              | Background |
| CLIP-B       |     | 17.8     |     | 30.0     | 40.4     |     | CLIP-B         | 11.6               | 10.9        | 35.9              | 27.1         | 37.0 32.0  |
|              |     |          |     |          |          |     | SigLIP-B       | 13.9               | 19.7        | 34.5              | 32.1         | 50.5 39.5  |
| GeneCIS      |     | 19.5     |     | 31.8     | 42.2     |     |                |                    |             |                   |              |            |
|              |     |          |     |          |          |     | InstructBLIP   | 12.5               | 14.7        | 38.6              | 31.2         | 74.7 44.1  |
| CIReVL       |     | 17.9     |     | 29.4     | 40.4     |     |                |                    |             |                   |              |            |
|              |     |          |     |          |          |     | Ours(CLIP-B)   | 35.9               | 38.2        | 57.0              | 56.6         | 78.6 58.0  |
| MagicLens    |     | 15.5     |     | 28.4     | 39.1     |     |                |                    |             |                   |              |            |
|              |     |          |     |          |          |     | Ours(SigLIP-B) | 44.7               | 55.0        | 49.0              | 55.7         | 81.5 52.0  |
| SEARLE       |     | 17.0     |     | 29.7     | 40.7     |     |                |                    |             |                   |              |            |
| Ours(CLIP-B) |     | 24.4     |     | 38.3     | 50.5     |     |                |                    |             |                   |              |            |
|              |     |          |     |          |          |     | condition      | and implementation |             | details           | are provided | in the     |
supplementarymaterials.
onlyconditioningthequeryfeatures. Implementation details. In our method, we utilize two
| ricformulation, |         |     |           |             |         | We  |                             |     |     |                      |     |     |
| --------------- | ------- | --- | --------- | ----------- | ------- | --- | --------------------------- | --- | --- | -------------------- | --- | --- |
|                 |         |     |           |             |         |     | VLMs,CLIP[35]andSigLIP[47]. |     |     | Togeneratecondition- |     |     |
| further apply   | GeneCIS | in  | symmetric | formulation | by also |     |                             |     |     |                      |     |     |
feed-forwardingthedatabasefeatureswiththeconditions relatedtextualprompts,weutilizeChatGPT-5. Wetruncate
GeneCIS†. thetop-ksingularvectorsandselectk = 50forallexperi-
| through | the modulator, | noted | as  |     | In addition, |     |                                                         |     |     |     |     |     |
| ------- | -------------- | ----- | --- | --- | ------------ | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- |
|         |                |       |     |     |              |     | ments. Additionaldetailsareinthesupplementarymaterials. |     |     |     |     |     |
wealsoincludeInstructBLIP[7]asourcompetingmethod,
whichutilizesCLIPvisionencoderwithLLMtoperform
5.2.ComparisonofRetrievalAccuracy
instruction-followingtasks.Toextracttheconditionedvisual
featuresinInstructBLIP,weaverageQ-formeroutputtokens, Table 2 presents the single conditional retrieval accuracy
followingFocalLens[17]. Thedetailedinstructionsforeach ofourmethodandcompetingmethodsacrossvariousreal

Conditionalretrieved images
| Query | Top-1                 |     |     | Top-5 | Top-1                      |     |     | Top-5 |
| ----- | --------------------- | --- | --- | ----- | -------------------------- | --- | --- | ----- |
|       | (a) CLIP-B (AP:0.125) |     |     |       | (b) InstructBLIP(AP:0.071) |     |     |       |
[Instrument]
|     | (c) GeneCIS(AP:0.069) |     |     |     | (d) Ours (AP:0.768) |     |     |     |
| --- | --------------------- | --- | --- | --- | ------------------- | --- | --- | --- |
Query
|     | (a) CLIP-B (AP:0.140) |     |     |     | (b) InstructBLIP(AP:0.148) |     |     |     |
| --- | --------------------- | --- | --- | --- | -------------------------- | --- | --- | --- |
[Color]
|     | (c) GeneCIS(AP:0.124) |     |     |     | (d) Ours(AP:0.835) |     |     |     |
| --- | --------------------- | --- | --- | --- | ------------------ | --- | --- | --- |
Figure5.Qualitativecomparisonofourmethodwithcompetingmethods.Foreachqueryimageandconditiontextpair,wecomparethe
top-5retrievedresultsfrom(a)CLIP-B,(b)InstructBLIP,(c)GeneCIS,and(d)ourmethod.WealsoreportAveragePrecision(AP)ineach
result.Greenboxesindicatecorrectlyretrievedimages,whileincorrectretrievalsareshowninredboxes.
[Species] Top-5 Table 5. Effectiveness of manifold modeling. We ablate the
|     |     |     |     | underlyingmanifoldmodelingacrossdatasets. |     |     |     | Tobettercapture |
| --- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --------------- |
generaltrends,weaggregatetheresultsfromdatasets,andreportthe
averagedmAP.Duetospacelimitations,weabbreviatemanifold
|     | [Location] |     | Top-5 |          |                  |            |     |                |
| --- | ---------- | --- | ----- | -------- | ---------------- | ---------- | --- | -------------- |
|     |            |     |       | modeling | as manifold, and | CLAY-Human | and | CLAY-Object as |
HumanandObject,respectively.
|       |                              |     |     |             | Stanford40 | Fine-grained. | Clever4 | Object Human |
| ----- | ---------------------------- | --- | --- | ----------- | ---------- | ------------- | ------- | ------------ |
| Query | Conditional retrieved images |     |     |             |            |               |         |              |
|       |                              |     |     | w/omanifold | 57.9       | 60.5          | 43.8    | 65.4 76.8    |
Figure6. QualitativeresultonOxfordpetsdataset. Wevisual- w/manifold 59.8 61.3 46.0 66.9 78.3
| izethetop-5retrievedresultswithconditiondog              |     | speciesand |     |     |     |     |     |     |
| -------------------------------------------------------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
| location. Sincenoground-truthlocationlabelsareavailable, |     |            |     |     |     |     |     |     |
bilityofpre-trainedVLMs,achievingstate-of-the-artperfor-
wepresentqualitativeexamplesonly.
manceacrossdiversedatasetsandvaryingconditiontypes
andsyntheticevaluationdatasets. Forclarity,wealsoreport withoutincurringhighlycomputationaloverheadinthesym-
thebaselineperformanceofCLIP-BandSigLIP-B,which metric setting. We further present the quantitative results
onGeneCISbenchmark“FocusAttribute”subset,compared
| cannotinputtextconditions. | Asshowninthistable,CLAY |     |     |     |     |     |     |     |
| -------------------------- | ----------------------- | --- | --- | --- | --- | --- | --- | --- |
consistentlyoutperformscompetingmethods,exhibitingthe toComposedImageRetrievalmethods[1,21,48]. Table3
strongadaptabilityacrossdiversedatasetsandvaryingcon- showsCLAYstillachievesthebestperformancewhenutiliz-
ditions, highlighting that similarity modulation can yield ingthesameorevensmallerbackbone(i.e.,ViT-B).InFig.5,
highlyeffectiveresults. GeneCIS,arepresentativetraining- we visualize conditional retrieved results, and it demon-
|     |     |     |     | strates that | our method | achieves | better condition-aligned |     |
| --- | --- | --- | --- | ------------ | ---------- | -------- | ------------------------ | --- |
basedmethod,performspoorlyintheasymmetricsetting;as
discussedinSec.3.1,itssymmetricvariant(i.e.,GeneCIS†) retrievalthanthecomparisonmethods.
| generallyachievesbetterperformance. |     | AlthoughGeneCIS |     |              |      |            |             |           |
| ----------------------------------- | --- | --------------- | --- | ------------ | ---- | ---------- | ----------- | --------- |
|                                     |     |                 |     | In addition, | CLAY | can easily | be extended | to multi- |
showsreasonablequantitativeresultsinthesymmetriccase, conditionalretrievalbyconstructingtextfeaturematrixfrom
it remains impractical due to its computational overhead. multiple condition-related textual prompts. Table 4 indi-
| Moreover, while | it performs well | on several cases | (Stan- |     |     |     |     |     |
| --------------- | ---------------- | ---------------- | ------ | --- | --- | --- | --- | --- |
catesCLAYperformsreliablyunderthemulti-conditional
ford40), its performance degrades under dog species retrievalsetting,achievinghighaccuracyonCLAY-Object
andcar modelconditionscomparedtothebaseline. This andCLAY-Humandatasets.Furthermore,asshowninFig.6,
impliesthatwhileGeneCISbenefitsfromspecifictypesof CLAYeffectivelyreflectsdiverseconditionswithinthesame
textconditions,ithaslimitedgeneralizationcapability. query. Figure 7 also shows how the representation space
Incontrast,ourmethodinheritsthestrongzero-shotcapa- variesunderdifferentconditions. Thiswiderangeofcover-

CLIP-B Ours CLIP-B Ours CLIP-B Ours
(a) condition 𝑐: Action (b) condition 𝑐: Background (c) condition 𝑐: Age
Figure7. Representationspacevisualizationwitht-SNE.Wereportt-SNEofCLIP-Bandours(CLIP-B)onCLAY-Humanunder
condition(a)action,(b)background,and(c)age.Thefeatureswiththesamelabelareshowninthesamecolorforeasyinterpretation.
ComparedtothefixedrepresentationspaceinCLIP-B,ourmethodformsmorediscriminativespacescompliantwithgivenconditions.
ageiscrucialinreal-worldscenarios,whereusersmayhave (b) Ours
variousintentionsthatneedtobesatisfied.
Table5showstheperformancegainsachievedbyconsid-
eringthegeometryofVLM’sembeddingspacetopreserve
relativerelationships. Accountingforthehypersphericalna-
tureoftherepresentationspacefurtherreducesdistortiondur-
ingspacemodulation,leadingtoconsistentimprovements
inretrievalaccuracyandhighlightingtheadvantageofcon-
sideringtheunderlyinggeometryofVLMrepresentations.
The number of conditions
5.3.ComparisonofInferenceTime
As mentioned above, a symmetric formulation can be
beneficial for improving retrieval accuracy by capturing
richer condition-aware visual representations from query
anddatabaseimages. However,couplingthevisualfeature
extractionprocesswithconditioningmoduleintroducesnon-
negligible computational demands, since condition-aware
visualrepresentationsneedtobere-encodedwheneverthe
conditionchangesoranewoneisintroduced.Incontrast,our
methoddecouplestheconditionalfeatureextractionwithvi-
sualfeatureextractionandconditioningprocess,eliminating
theneedforre-encodingduringinference. Figure8demon-
stratestheadvantageofCLAYregardingthecomputingtime
asadditionalconditionsaregiven. Inthisexperiment,we
sample100databaseimagesfromCLAY-Objectandreport
theaveragedinferencetimeof10retrievalsforasinglequery
onCPU.Asthenumberofconditionsincreased,theinfer-
encetimeofGeneCIS†increasesaccordinglysinceitneeds
toextractthedatabasefeaturesforeachconditionagain.
5.4.AnalysisofRepresentationSpace
Toconfirmtheeffectivenessofsimilarityspacemodulation,
wevisualizet-SNEresultscomparingwithCLIP-BandOurs
(CLIP-B)onourCLAY-Humandataset. AsshowninFig7,
thevisualrepresentationspaceisdistinctlyseparatedwith
each action and background condition, compared to
thebaseline. Thisresultshowshowourmethodadaptively
modulates the original visual features onto the condition-
awaresubspace,achievinghighconditionalretrievalperfor-
mance. Interestingly,applyingourmethodshowsarankable
property along the condition axis (i.e., age) in Fig 7 (c).
)ces(
emit
ecnerefnI
(a) GeneCIS†
Figure8.ComparisonofinferencetimeonsampledCLAY-shoes
dataset. We compare the inference time between the symmet-
ricsimilarityformulationofGeneCIS(i.e.,GeneCIS†),andours.
GeneCIS†showshighinferencetimewheneverconditionisgiven,
whileourmethodreducesafterthefirstcondition,benefitingfrom
thedecouplingfeatureextractionfromtheconditioningprocess.
Previouswork[38]studiedthisrankabilitybyidentifying
therankableaxis,CLAYcanachievethispropertynaturally
throughconditionalsimilarityspacemodulation. Wealso
findthatthisrankablepropertyholdsinClevr4-count,and
provideinsupplementarymaterials.
6.Conclusion
Inthiswork,weintroduceCLAY,anoveltraining-freecon-
ditionalvisualsimilaritycomputationmethodthatadaptively
modulatesthefixedsimilarityofexistingpre-trainedVLMs
toatext-conditionalsimilarity. CLAY’sadaptivecondition-
ing strikes a sweet spot between accuracy and efficiency.
Thesetwofactorsarethatpriormethodsstruggledtoachieve
simultaneously,enablingpracticalandeffectiveconditional
retrieval. Byleveragingtheunderlyinghypersphericalgeom-
etry,ourmanifold-awaretextualsubspaceenablestheoreti-
callygroundedmodelingoftheconditionalrelationshipbe-
tweenimages. Tosupportacomprehensiveevaluationofthe
multi-facetedaspectsofconditionalretrieval,weconstruct
asyntheticevaluationdatasetcontainingdiverseobjectand
human images annotated with conceptual conditions. We
believethisworkopensuppromisingdirectionsforbuilding
practicalretrievalsystemscomplianttohumanintentions.

References tuningenableszero-shotconditionalimagerepresentations.
|             |           |         |            |       |          |     | arXivpreprintarXiv:2504.08368,2025.                 |     |     |     |     | 2,3,6 |     |
| ----------- | --------- | ------- | ---------- | ----- | -------- | --- | --------------------------------------------------- | --- | --- | --- | --- | ----- | --- |
| [1] Alberto | Baldrati, | Lorenzo | Agnolucci, | Marco | Bertini, | and |                                                     |     |     |     |     |       |     |
|             |           |         |            |       |          |     | [18] YoungKyunJang,DatHuynh,AshishShah,Wen-KaiChen, |     |     |     |     |       |     |
AlbertoDelBimbo.Zero-shotcomposedimageretrievalwith
|                   |     |              |     |       |     |     | andSer-NamLim.                               |     | Sphericallinearinterpolationandtext- |     |     |     |         |
| ----------------- | --- | ------------ | --- | ----- | --- | --- | -------------------------------------------- | --- | ------------------------------------ | --- | --- | --- | ------- |
| textualinversion. |     | InICCV,2023. |     | 2,3,7 |     |     |                                              |     |                                      |     |     |     |         |
|                   |     |              |     |       |     |     | anchoringforzero-shotcomposedimageretrieval. |     |                                      |     |     |     | InECCV, |
[2] DavideBerasi,MatteoFarina,MassimilianoMancini,Elisa
|     |     |     |     |     |     |     | 2024. 2,3 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- |
Ricci, and Nicola Strisciuglio. Not only text: Exploring [19] JustinJohnson,BharathHariharan,LaurensVanDerMaaten,
compositionalityofvisualrepresentationsinvision-language
|         |              |     |     |     |     |     | LiFei-Fei,CLawrenceZitnick,andRossGirshick. |     |     |     |     |     | Clevr:A |
| ------- | ------------ | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | ------- |
| models. | InCVPR,2025. |     | 4   |     |     |     |                                             |     |     |     |     |     |         |
diagnosticdatasetforcompositionallanguageandelementary
| [3] Black Forest | Labs. | Flux.1. | https://huggingface. |     |     |     |                  |     |              |     |     |     |     |
| ---------------- | ----- | ------- | -------------------- | --- | --- | --- | ---------------- | --- | ------------ | --- | --- | --- | --- |
|                  |       |         |                      |     |     |     | visualreasoning. |     | InCVPR,2017. |     | 5   |     |     |
co/black-forest-labs/FLUX.1-dev,2024.
|     |     |     |     |     |     | 1-dev. | [20] ShyamgopalKarthik,KarstenRoth,MassimilianoMancini, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------ | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
5
|                    |          |     |             |     |         |       | andZeynepAkata.           |     | Vision-by-languagefortraining-freecom- |              |     |     |     |
| ------------------ | -------- | --- | ----------- | --- | ------- | ----- | ------------------------- | --- | -------------------------------------- | ------------ | --- | --- | --- |
| [4] Lukas Bossard, | Matthieu |     | Guillaumin, | and | Luc Van | Gool. |                           |     |                                        |              |     |     |     |
|                    |          |     |             |     |         |       | positionalimageretrieval. |     |                                        | InICLR,2024. |     | 2   |     |
Food-101–miningdiscriminativecomponentswithrandom
|     |     |     |     |     |     |     | [21] ShyamgopalKarthik,KarstenRoth,MassimilianoMancini, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
forests. InECCV,2014. 5 andZeynepAkata. Vision-by-languagefortraining-freecom-
| [5] Bingyi Cao,                       | Andre | Araujo, | and JackSim. |              | Unifying | deep |                           |     |               |              |          |               |     |
| ------------------------------------- | ----- | ------- | ------------ | ------------ | -------- | ---- | ------------------------- | --- | ------------- | ------------ | -------- | ------------- | --- |
|                                       |       |         |              |              |          |      | positionalimageretrieval. |     |               | InICLR,2024. |          | 3,7           |     |
| localandglobalfeaturesforimagesearch. |       |         |              | InECCV,2020. |          | 2    |                           |     |               |              |          |               |     |
|                                       |       |         |              |              |          |      | [22] JonathanKrause,      |     | MichaelStark, |              | JiaDeng, | andLiFei-Fei. |     |
[6] MathildeCaron,HugoTouvron,IshanMisra,Herve´Je´gou, 3dobjectrepresentationsforfine-grainedcategorization. In
| JulienMairal,PiotrBojanowski,andArmandJoulin. |     |     |     |     |     | Emerg- | ICCVW,2013. |     | 5   |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | ------ | ----------- | --- | --- | --- | --- | --- | --- |
ingpropertiesinself-supervisedvisiontransformers.InICCV,
|     |     |     |     |     |     |     | [23] SehyunKwon,JaeseungPark,MinkyuKim,JaewoongCho, |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
2021. 1
|     |     |     |     |     |     |     | ErnestK.Ryu,andKangwookLee. |     |     |     | Imageclusteringcondi- |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------- | --- | --- | --- | --------------------- | --- | --- |
[7] WenliangDai,JunnanLi,DongxuLi,AnthonyTiong,Junqi tionedontextcriteria. InICLR,2024. 5
Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and [24] YangyangLi,DaqingLiu,WuLiu,AllenHe,XinchenLiu,
| StevenHoi.                           | Instructblip: |     | Towardsgeneral-purposevision- |                 |     |     |                                              |     |     |     |     |            |          |
| ------------------------------------ | ------------- | --- | ----------------------------- | --------------- | --- | --- | -------------------------------------------- | --- | --- | --- | --- | ---------- | -------- |
|                                      |               |     |                               |                 |     |     | YongdongZhang,andGuoqingJin.                 |     |     |     |     | Omniprism: | Learning |
| languagemodelswithinstructiontuning. |               |     |                               | InNeurIPS,2023. |     |     |                                              |     |     |     |     |            |          |
|                                      |               |     |                               |                 |     |     | disentangledvisualconceptforimagegeneration. |     |     |     |     |            | InCoRR,  |
6
|     |     |     |     |     |     |     | 2024. 5 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
[8] NavneetDalalandBillTriggs. Histogramsoforientedgradi- [25] VictorWeixinLiang,YuhuiZhang,YongchanKwon,Serena
| entsforhumandetection. |      |            | InCVPR,2005. | 2          |     |        |        |           |     |      |          |                    |     |
| ---------------------- | ---- | ---------- | ------------ | ---------- | --- | ------ | ------ | --------- | --- | ---- | -------- | ------------------ | --- |
|                        |      |            |              |            |     |        | Yeung, | and James | Y   | Zou. | Mind the | gap: Understanding |     |
| [9] Sara Dorfman,      | Dana | Cohen-Bar, |              | Rinon Gal, | and | Daniel |        |           |     |      |          |                    |     |
themodalitygapinmulti-modalcontrastiverepresentation
Cohen-Or. Ip-composer: Semantic composition of visual learning. InNeurIPS,2022. 4
concepts. InACMTransactionsonGraphics(SIGGRAPH), [26] ZheyuanLiu,CristianRodriguez-Opazo,DamienTeney,and
| pages1–11,2025. |     | 2,3,4 |     |     |     |     |               |     |                                         |     |     |     |     |
| --------------- | --- | ----- | --- | --- | --- | --- | ------------- | --- | --------------------------------------- | --- | --- | --- | --- |
|                 |     |       |     |     |     |     | StephenGould. |     | Imageretrievalonreal-lifeimageswithpre- |     |     |     |     |
[10] Sedigheh Eslami and Gerard de Melo. Mitigate the gap: trainedvision-and-languagemodels. InICCV,2021. 3
Improvingcross-modalalignmentinCLIP. InICLR,2025. 4 [27] David G Lowe. Distinctive image features from scale-
[11] P.T.Fletcher,ConglinLu,S.M.Pizer,andSarangJoshi. Prin- invariantkeypoints. IJCV,60(2):91–110,2004. 2
cipalgeodesicanalysisforthestudyofnonlinearstatistics
|     |      |              |     |         |          |     | [28] Subhransu | Maji, | Esa | Rahtu, | Juho | Kannala, | Matthew |
| --- | ---- | ------------ | --- | ------- | -------- | --- | -------------- | ----- | --- | ------ | ---- | -------- | ------- |
|     | IEEE | Transactions | on  | Medical | Imaging, |     |                |       |     |        |      |          |         |
of shape. 23(8): Blaschko, and Andrea Vedaldi. Fine-grained visual clas-
995–1005,2004. 4 sificationofaircraft. arXivpreprintarXiv:1306.5151,2013.
| [12] StephanieFu,NetanelTamir,ShobhitaSundaram,LucyChai, |             |        |     |                |           |     | 5                                                  |     |     |     |     |     |     |
| -------------------------------------------------------- | ----------- | ------ | --- | -------------- | --------- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| Richard                                                  | Zhang, Tali | Dekel, | and | Phillip Isola. | Dreamsim: |     |                                                    |     |     |     |     |     |     |
|                                                          |             |        |     |                |           |     | [29] Quang-BinhNguyen,MinhLuu,QuangNguyen,AnhTran, |     |     |     |     |     |     |
Learningnewdimensionsofhumanvisualsimilarityusing andKhoiNguyen. Csd-var:Content-styledecompositionin
syntheticdata. InNeurIPS,2023. 1 visualautoregressivemodels. InICCV,2025. 2,3,4
[13] AlbertGordo,JonAlmazan,JeromeRevaud,andDianeLar- [30] Maria-ElenaNilsbackandAndrewZisserman. Automated
lus. End-to-endlearningofdeepvisualrepresentationsfor
|     |     |     |     |     |     |     | flowerclassificationoveralargenumberofclasses. |     |     |     |     |     | InIn- |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | ----- |
IJCV,124(2):237–254,2017.
imageretrieval. 5 dianConferenceonComputerVision,GraphicsandImage
[14] GeonmoGu,SanghyukChun,WonjaeKim,YoohoonKang, Processing,2008. 5
andSangdooYun. Language-onlytrainingofzero-shotcom- [31] HyeonwooNoh,AndreAraujo,JackSim,TobiasWeyand,
| posedimageretrieval. |     | InCVPR,2024. |     | 2,3 |     |     |                |     |                                        |     |     |     |     |
| -------------------- | --- | ------------ | --- | --- | --- | --- | -------------- | --- | -------------------------------------- | --- | --- | --- | --- |
|                      |     |              |     |     |     |     | andBohyungHan. |     | Large-scaleimageretrievalwithattentive |     |     |     |     |
[15] SørenHauberg. Directionalstatisticswiththesphericalnor- deeplocalfeatures. InICCV,2017. 2
maldistribution. In201821stinternationalconferenceon [32] MaximeOquab,LeonBottou,IvanLaptev,andJosefSivic.
informationfusion(FUSION),2018. 4 Learning andtransferring mid-levelimage representations
|                                |     |     |     |                      |     |     | usingconvolutionalneuralnetworks. |     |     |     |     | InCVPR,2014. | 2   |
| ------------------------------ | --- | --- | --- | -------------------- | --- | --- | --------------------------------- | --- | --- | --- | --- | ------------ | --- |
| [16] JamesHaysandAlexeiAEfros. |     |     |     | Scenecompletionusing |     |     |                                   |     |     |     |     |              |     |
millions of photographs. ACM Transactions on graphics [33] OmkarM.Parkhi,AndreaVedaldi,AndrewZisserman,and
(TOG),26(3):4–es,2007. 2 C.V.Jawahar. Catsanddogs. InCVPR,2012. 5
[17] Cheng-Yu Hsieh, Pavan Kumar Anasosalu Vasu, Fartash [34] Filip Radenovic´, Ahmet Iscen, Giorgos Tolias, Yannis
Faghri,RavitejaVemulapalli,Chun-LiangLi,RanjayKrishna, Avrithis, and Ondˇrej Chum. Revisiting oxford and paris:
Oncel Tuzel, and Hadi Pouransari. Focallens: Instruction Large-scaleimageretrievalbenchmarking. InCVPR,2018. 5

[35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
AmandaAskell,PamelaMishkin,JackClark,etal. Learning
transferablevisualmodelsfromnaturallanguagesupervision.
InICML,2021. 1,2,3,6
[36] JeromeRevaud,JonAlmaza´n,RafaelSRezende,andCesar
RobertodeSouza. Learningwithaverageprecision:Training
imageretrievalwithalistwiseloss. InICCV,2019. 5
[37] KuniakiSaito,KihyukSohn,XiangZhang,Chun-LiangLi,
Chen-YuLee, KateSaenko,andTomasPfister. Pic2word:
Mappingpicturestowordsforzero-shotcomposedimage
retrieval. InCVPR,2023. 2
[38] Ankit Sonthalia, Arnas Uselis, and Seong Joon Oh. On
the rankability of visual embeddings. arXiv preprint
arXiv:2507.03683,2025. 8
[39] SagarVaze,NicolasCarion,andIshanMisra. Genecis: A
benchmarkforgeneralconditionalimagesimilarity.InCVPR,
2023. 2,3,5
[40] AndreasVeit,SergeBelongie,andTheofanisKaraletsos.Con-
ditionalsimilaritynetworks. InCVPR,2017. 2,3
[41] JinjunWang,JianchaoYang,KaiYu,FengjunLv,Thomas
Huang,andYihongGong. Locality-constrainedlinearcoding
forimageclassification. InCVPR,2010. 5
[42] ShijieWang,JianlongChang,HaojieLi,ZhihuiWang,Wanli
Ouyang, and Qi Tian. Open-set fine-grained retrieval via
promptingvision-languageevaluator. InCVPR,2023. 3
[43] YunchaoWei,YaoZhao,CanyiLu,ShikuiWei,LuoqiLiu,
ZhenfengZhu,andShuichengYan. Cross-modalretrieval
withcnnvisualfeatures:Anewbaseline. IEEEtransactions
oncybernetics,47(2):449–460,2016. 2
[44] HuiWu,YupengGao,XiaoxiaoGuo,ZiadAl-Halah,Steven
Rennie,KristenGrauman,andRogerioFeris. Thefashion
iqdataset:Retrievingimagesbycombiningsideinformation
andrelativenaturallanguagefeedback. InCVPR,2021. 3
[45] BangpengYaoandLiFei-Fei. Grouplet:Astructuredimage
representationforrecognizinghumanandobjectinteractions.
InCVPR,2010. 5
[46] BangpengYao,XiaoyeJiang,AdityaKhosla,AndyLaiLin,
LeonidasGuibas,andLiFei-Fei. Humanactionrecognition
by learning bases of action attributes and parts. In ICCV,
2011. 5
[47] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and
LucasBeyer. Sigmoidlossforlanguageimagepre-training.
InICCV,2023. 1,2,3,6
[48] KaiZhang,YiLuan,HexiangHu,KentonLee,SiyuanQiao,
WenhuChen,YuSu,andMing-WeiChang. Magiclens:self-
supervisedimageretrievalwithopen-endedinstructions. In
ICML,2024. 2,3,6,7