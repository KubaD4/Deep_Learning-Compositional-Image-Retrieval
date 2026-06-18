Learning Transferable Visual Models From Natural Language Supervision
AlecRadford*1 JongWookKim*1 ChrisHallacy1 AdityaRamesh1 GabrielGoh1 SandhiniAgarwal1
GirishSastry1 AmandaAskell1 PamelaMishkin1 JackClark1 GretchenKrueger1 IlyaSutskever1
Abstract Task-agnosticobjectivessuchasautoregressiveandmasked
languagemodelinghavescaledacrossmanyordersofmag-
| State-of-the-art |     |     | computer | vision | systems | are |     |     |     |     |     |     |     |
| ---------------- | --- | --- | -------- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
1202 beF 62  ]VC.sc[  1v02000.3012:viXra
|                   |     |         |                            |                      |     |     | nitudeincompute,     |     | modelcapacity,                   |     | anddata, | steadilyim- |     |
| ----------------- | --- | ------- | -------------------------- | -------------------- | --- | --- | -------------------- | --- | -------------------------------- | --- | -------- | ----------- | --- |
| trained           | to  | predict | a fixed                    | set of predetermined |     |     |                      |     |                                  |     |          |             |     |
|                   |     |         |                            |                      |     |     | provingcapabilities. |     | Thedevelopmentof“text-to-text”as |     |          |             |     |
| objectcategories. |     |         | Thisrestrictedformofsuper- |                      |     |     |                      |     |                                  |     |          |             |     |
astandardizedinput-outputinterface(McCannetal.,2018;
visionlimitstheirgeneralityandusabilitysince
Radfordetal.,2019;Raffeletal.,2019)hasenabledtask-
additionallabeleddataisneededtospecifyany
agnosticarchitecturestozero-shottransfertodownstream
| othervisualconcept. |     |     | Learningdirectlyfromraw |     |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
datasetsremovingtheneedforspecializedoutputheadsor
textaboutimagesisapromisingalternativewhich
|     |     |     |     |     |     |     | datasetspecificcustomization. |     |     | FlagshipsystemslikeGPT-3 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | ------------------------ | --- | --- | --- |
leveragesamuchbroadersourceofsupervision.
(Brownetal.,2020)arenowcompetitiveacrossmanytasks
Wedemonstratethatthesimplepre-trainingtask
|     |     |     |     |     |     |     | with bespoke | models |     | while requiring |     | little to no | dataset |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | --- | --------------- | --- | ------------ | ------- |
ofpredictingwhichcaptiongoeswithwhichim-
specifictrainingdata.
ageisanefficientandscalablewaytolearnSOTA
imagerepresentationsfromscratchonadataset Theseresultssuggestthattheaggregatesupervisionacces-
of400million(image,text)pairscollectedfrom sibletomodernpre-trainingmethodswithinweb-scalecol-
theinternet. Afterpre-training,naturallanguage lectionsoftextsurpassesthatofhigh-qualitycrowd-labeled
is used to reference learned visual concepts (or NLPdatasets. However, inotherfieldssuchascomputer
describe new ones) enabling zero-shot transfer vision it is still standard practice to pre-train models on
of the model to downstream tasks. We study crowd-labeleddatasetssuchasImageNet(Dengetal.,2009).
theperformanceofthisapproachbybenchmark- Could scalable pre-training methods which learn directly
ing on over 30 different existing computer vi- fromwebtextresultinasimilarbreakthroughincomputer
sion datasets, spanning tasks such as OCR, ac- vision? Priorworkisencouraging.
tionrecognitioninvideos,geo-localization,and
Over20yearsagoMorietal.(1999)exploredimproving
manytypesoffine-grainedobjectclassification.
|     |       |           |               |     |         |       | content based | image |     | retrieval by | training | a model | to pre- |
| --- | ----- | --------- | ------------- | --- | ------- | ----- | ------------- | ----- | --- | ------------ | -------- | ------- | ------- |
| The | model | transfers | non-trivially |     | to most | tasks |               |       |     |              |          |         |         |
dictthenounsandadjectivesintextdocumentspairedwith
andisoftencompetitivewithafullysupervised
images. Quattonietal.(2007)demonstrateditwaspossi-
| baseline | without |     | the need | for any | dataset | spe- |     |     |     |     |     |     |     |
| -------- | ------- | --- | -------- | ------- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- |
bletolearnmoredataefficientimagerepresentationsvia
| cific | training. | For | instance, | we  | match | the ac- |     |     |     |     |     |     |     |
| ----- | --------- | --- | --------- | --- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- |
manifoldlearningintheweightspaceofclassifierstrained
| curacy | of  | the original | ResNet-50 |     | on ImageNet |     |            |       |             |            |     |              |      |
| ------ | --- | ------------ | --------- | --- | ----------- | --- | ---------- | ----- | ----------- | ---------- | --- | ------------ | ---- |
|        |     |              |           |     |             |     | to predict | words | in captions | associated |     | with images. | Sri- |
zero-shotwithoutneedingtouseanyofthe1.28
|                                        |     |     |     |     |     |     | vastava &       | Salakhutdinov |             | (2012)     | explored | deep represen- |     |
| -------------------------------------- | --- | --- | --- | --- | --- | --- | --------------- | ------------- | ----------- | ---------- | -------- | -------------- | --- |
| milliontrainingexamplesitwastrainedon. |     |     |     |     |     | We  |                 |               |             |            |          |                |     |
|                                        |     |     |     |     |     |     | tation learning |               | by training | multimodal |          | Deep Boltzmann |     |
releaseourcodeandpre-trainedmodelweightsat
Machinesontopoflow-levelimageandtexttagfeatures.
https://github.com/OpenAI/CLIP.
Joulinetal.(2016)modernizedthislineofworkanddemon-
|     |     |     |     |     |     |     | strated that | CNNs   | trained | to predict       | words | in image       | cap- |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | ------- | ---------------- | ----- | -------------- | ---- |
|     |     |     |     |     |     |     | tions learn  | useful | image   | representations. |       | They converted |      |
1.IntroductionandMotivatingWork
thetitle,description,andhashtagmetadataofimagesinthe
|              |         |     |             |          |      |          | YFCC100M | dataset | (Thomee | et  | al., 2016) | into a | bag-of- |
| ------------ | ------- | --- | ----------- | -------- | ---- | -------- | -------- | ------- | ------- | --- | ---------- | ------ | ------- |
| Pre-training | methods |     | which learn | directly | from | raw text |          |         |         |     |            |        |         |
wordsmulti-labelclassificationtaskandshowedthatpre-
| have revolutionized |     | NLP | over | the last | few years | (Dai & |     |     |     |     |     |     |     |
| ------------------- | --- | --- | ---- | -------- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
Le,2015;Petersetal.,2018;Howard&Ruder,2018;Rad- trainingAlexNet(Krizhevskyetal.,2012)topredictthese
|                    |       |                                   |         |              |     |                | labels learned                               |     | representations | which | preformed |     | similarly |
| ------------------ | ----- | --------------------------------- | ------- | ------------ | --- | -------------- | -------------------------------------------- | --- | --------------- | ----- | --------- | --- | --------- |
| ford et al.,       | 2018; | Devlin                            | et al., | 2018; Raffel |     | et al., 2019). |                                              |     |                 |       |           |     |           |
|                    |       |                                   |         |              |     |                | toImageNet-basedpre-trainingontransfertasks. |     |                 |       |           |     | Lietal.   |
| *Equalcontribution |       | 1OpenAI,SanFrancisco,CA94110,USA. |         |              |     |                |                                              |     |                 |       |           |     |           |
(2017)thenextendedthisapproachtopredictingphrasen-
Correspondenceto: <{alec,jongwook}@openai.com>. gramsinadditiontoindividualwordsanddemonstratedthe
abilityoftheirsystemtozero-shottransfertootherimage

2
LearningTransferableVisualModelsFromNaturalLanguageSupervision
(1) Contrastive pre-training (2) Create dataset classifier from label text
plane
car
| P P e e p p p p e e r r 	 	 t t h h e e | T e x t |     |     |     |     |     |     |
| --------------------------------------- | ------- | --- | --- | --- | --- | --- | --- |
a uP sP e se p ip p ep e 	e r pr 	 u	 t pt h ph e e A 	 p h o t o 	 o f T e x t
a a u a u s u s s s s i s i e i e 	 e 	 p 	 p u p u u p p En c o d er … dog a 	 { o b j e c t } . En c o d er
|     |     |     |          | ⋮ ⋮ |     |     | …   |
| --- | --- | --- | -------- | --- | --- | --- | --- |
|     |     |     | T1 T2 T3 | TN  |     |     |     |
…
bird
|     |     | I1 I1·T1 | I1·T2 I1·T3 | … I1·TN |     |     |     |
| --- | --- | -------- | ----------- | ------- | --- | --- | --- |
(3) Use for zero-shot prediction
|     |              | I2 I2·T1 | I2·T2 I2·T3 | I2·TN   |           | T1 T2                | T3 TN   |
| --- | ------------ | -------- | ----------- | ------- | --------- | -------------------- | ------- |
|     |              |          |             | …       |           |                      | …       |
|     | I m a g e    | I3 I3·T1 | I3·T2 I3·T3 | … I3·TN |           |                      |         |
|     | E n c o d er |          |             |         | I m a g e | I1 I1·T1 I1·T2 I1·T3 | … I1·TN |
E n c o d er
|     |     | ⋮ ⋮      | ⋮ ⋮ ⋮       | ⋱ ⋮     |     |        |            |
| --- | --- | -------- | ----------- | ------- | --- | ------ | ---------- |
|     |     | IN IN·T1 | IN·T2 IN·T3 | … IN·TN |     | A	 p h | o t o 	 of |
|     |     |          |             |         |     | 	 a    | 	 d o g .  |
Figure1.Summaryofourapproach.Whilestandardimagemodelsjointlytrainanimagefeatureextractorandalinearclassifiertopredict
somelabel,CLIPjointlytrainsanimageencoderandatextencodertopredictthecorrectpairingsofabatchof(image,text)training
examples.Attesttimethelearnedtextencodersynthesizesazero-shotlinearclassifierbyembeddingthenamesordescriptionsofthe
targetdataset’sclasses.
classification datasets by scoring target classes based on mises. Bothworkscarefullydesign,andintheprocesslimit,
theirdictionaryoflearnedvisualn-gramsandpredictingthe their supervision to 1000 and 18291 classes respectively.
onewiththehighestscore. Adoptingmorerecentarchitec- Naturallanguageisabletoexpress,andthereforesupervise,
turesandpre-trainingapproaches,VirTex(Desai&Johnson, a much wider set of visual concepts through its general-
2020), ICMLM(BulentSariyildizetal.,2020), andCon- ity. Bothapproachesalsousestaticsoftmaxclassifiersto
VIRT(Zhangetal.,2020)haverecentlydemonstratedthe performpredictionandlackamechanismfordynamicout-
potentialoftransformer-basedlanguagemodeling,masked puts. Thisseverelycurtailstheirflexibilityandlimitstheir
languagemodeling,andcontrastiveobjectivestolearnim- “zero-shot”capabilities.
agerepresentationsfromtext.
Acrucialdifferencebetweentheseweaklysupervisedmod-
Whileexcitingasproofsofconcept,usingnaturallanguage elsandrecentexplorationsoflearningimagerepresentations
supervisionforimagerepresentationlearningisstillrare. directlyfromnaturallanguageisscale. WhileMahajanetal.
Thisislikelybecausedemonstratedperformanceoncom- (2018)andKolesnikovetal.(2019)trainedtheirmodelsfor
monbenchmarksismuchlowerthanalternativeapproaches. acceleratoryearsonmillionstobillionsofimages,VirTex,
For example, Li et al. (2017) reach only 11.5% accuracy ICMLM,andConVIRTtrainedforacceleratordaysonone
onImageNetinazero-shotsetting. Thisiswellbelowthe to two hundred thousand images. In this work, we close
88.4% accuracy of the current state of the art (Xie et al., thisgapandstudythebehaviorsofimageclassifierstrained
2020). Itisevenbelowthe50%accuracyofclassiccom- withnaturallanguagesupervisionatlargescale. Enabled
putervisionapproaches(Dengetal.,2012). Instead,more bythelargeamountsofpubliclyavailabledataofthisform
narrowlyscopedbutwell-targetedusesofweaksupervision ontheinternet,wecreateanewdatasetof400million(im-
haveimprovedperformance. Mahajanetal.(2018)showed age,text)pairsanddemonstratethatasimplifiedversionof
thatpredictingImageNet-relatedhashtagsonInstagramim- ConVIRTtrainedfromscratch,whichwecallCLIP,forCon-
agesisaneffectivepre-trainingtask. Whenfine-tunedto trastiveLanguage-ImagePre-training,isanefficientmethod
ImageNetthesepre-trainedmodelsincreasedaccuracyby of learning from natural language supervision. We study
over5%andimprovedtheoverallstateoftheartatthetime. thescalabilityofCLIPbytrainingaseriesofeightmodels
Kolesnikovetal.(2019)andDosovitskiyetal.(2020)have spanningalmost2ordersofmagnitudeofcomputeandob-
alsodemonstratedlargegainsonabroadersetoftransfer servethattransferperformanceisasmoothlypredictable
benchmarksbypre-trainingmodelstopredicttheclassesof functionofcompute(Hestnessetal.,2017;Kaplanetal.,
thenoisilylabeledJFT-300Mdataset. 2020). WefindthatCLIP,similartotheGPTfamily,learns
toperformawidesetoftasksduringpre-trainingincluding
Thislineofworkrepresentsthecurrentpragmaticmiddle
OCR,geo-localization,actionrecognition,andmanyothers.
groundbetweenlearningfromalimitedamountofsuper-
|     |     |     |     | We measure | this by benchmarking | the zero-shot | transfer |
| --- | --- | --- | --- | ---------- | -------------------- | ------------- | -------- |
vised“gold-labels”andlearningfrompracticallyunlimited
performanceofCLIPonover30existingdatasetsandfind
| amounts of | raw text. However, | it is not | without compro- |     |     |     |     |
| ---------- | ------------------ | --------- | --------------- | --- | --- | --- | --- |

3
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     |     |     |     |     | vision. Althoughearlyworkwrestledwiththecomplexity |     |     |     |
| --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- |
40
|     |     |     |     |     | of natural | language when | using topic model | and n-gram |
| --- | --- | --- | --- | --- | ---------- | ------------- | ----------------- | ---------- |
ycaruccA teNegamI tohS-oreZ 35
representations,improvementsindeepcontextualrepresen-
| 30  |     |     |     |     | tationlearningsuggestwenowhavethetoolstoeffectively |     |     |     |
| --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- |
leveragethisabundantsourceofsupervision(McCannetal.,
25
2017).
20
4X efficiency 3X efficiency Learning from natural language has several potential
15
|     |     |     |     |     | strengths | over other training | methods. | It’s much easier |
| --- | --- | --- | --- | --- | --------- | ------------------- | -------- | ---------------- |
10 Bag of Words Contrastive (CLIP) toscalenaturallanguagesupervisioncomparedtostandard
Bag of Words Prediction
| 5   |     |     |     |     | crowd-sourcedlabelingforimageclassificationsinceitdoes |     |     |     |
| --- | --- | --- | --- | --- | ------------------------------------------------------ | --- | --- | --- |
Transformer Language Model
notrequireannotationstobeinaclassic“machinelearning
0
2M 33M 67M 134M 268M 400M compatibleformat”suchasthecanonical1-of-Nmajority
# of images processed
|     |     |     |     |     | vote“goldlabel”. | Instead,methodswhichworkonnatural |     |     |
| --- | --- | --- | --- | --- | ---------------- | --------------------------------- | --- | --- |
languagecanlearnpassivelyfromthesupervisioncontained
| Figure2.CLIP | is much | more efficient | at zero-shot |     | transfer |     |     |     |
| ------------ | ------- | -------------- | ------------ | --- | -------- | --- | --- | --- |
thanourimagecaptionbaseline. Althoughhighlyexpressive, inthevastamountoftextontheinternet. Learningfrom
wefoundthattransformer-basedlanguagemodelsarerelatively naturallanguagealsohasanimportantadvantageovermost
weakatzero-shotImageNetclassification. Here,weseethatit unsupervisedorself-supervisedlearningapproachesinthat
learns3xslowerthanabaselinewhichpredictsabag-of-words
itdoesn’t“just”learnarepresentationbutalsoconnectsthat
(BoW)encodingofthetext(Joulinetal.,2016). Swappingthe representationtolanguagewhichenablesflexiblezero-shot
predictionobjectiveforthecontrastiveobjectiveofCLIPfurther transfer. Inthefollowingsubsections,wedetailthespecific
improvesefficiencyanother4x.
approachwesettledon.
2.2.CreatingaSufficientlyLargeDataset
| it can be | competitive | with prior | task-specific | supervised |     |     |     |     |
| --------- | ----------- | ---------- | ------------- | ---------- | --- | --- | --- | --- |
models. Wealsoconfirmthesefindingswithlinear-probe Existingworkhasmainlyusedthreedatasets,MS-COCO
representation learning analysis and show that CLIP out- (Linetal.,2014),VisualGenome(Krishnaetal.,2017),and
|     |     |     |     |     | YFCC100M(Thomeeetal.,2016). |     | WhileMS-COCOand |     |
| --- | --- | --- | --- | --- | --------------------------- | --- | --------------- | --- |
performsthebestpubliclyavailableImageNetmodelwhile
alsobeingmorecomputationallyefficient. Weadditionally VisualGenomearehighqualitycrowd-labeleddatasets,they
aresmallbymodernstandardswithapproximately100,000
findthatzero-shotCLIPmodelsaremuchmorerobustthan
equivalent accuracy supervised ImageNet models which trainingphotoseach. Bycomparison,othercomputervision
suggeststhatzero-shotevaluationoftask-agnosticmodelsis systemsaretrainedonupto3.5billionInstagramphotos
|                                             |     |     |     |          | (Mahajanetal.,2018). | YFCC100M,at100millionphotos, |     |     |
| ------------------------------------------- | --- | --- | --- | -------- | -------------------- | ---------------------------- | --- | --- |
| muchmorerepresentativeofamodel’scapability. |     |     |     | Thesere- |                      |                              |     |     |
sultshavesignificantpolicyandethicalimplications,which isapossiblealternative,butthemetadataforeachimageis
weconsiderinSection7. sparseandofvaryingquality. Manyimagesuseautomati-
|     |     |     |     |     | callygeneratedfilenameslike20160716 |                           |     | 113957.JPG      |
| --- | --- | --- | --- | --- | ----------------------------------- | ------------------------- | --- | --------------- |
|     |     |     |     |     | as “titles”                         | or contain “descriptions” | of  | camera exposure |
2.Approach
|     |     |     |     |     | settings. | After filtering to | keep only images | with natural |
| --- | --- | --- | --- | --- | --------- | ------------------ | ---------------- | ------------ |
2.1.NaturalLanguageSupervision languagetitlesand/ordescriptionsinEnglish,thedataset
|     |     |     |     |     | shrunkbyafactorof6toonly15millionphotos. |     |     | Thisis |
| --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | ------ |
Atthecoreofourapproachistheideaoflearningpercep- approximatelythesamesizeasImageNet.
| tion from | supervision | contained | in natural | language. | As  |     |     |     |
| --------- | ----------- | --------- | ---------- | --------- | --- | --- | --- | --- |
Amajormotivationfornaturallanguagesupervisionisthe
discussedintheintroduction,thisisnotatallanewidea,
largequantitiesofdataofthisformavailablepubliclyonthe
howeverterminologyusedtodescribeworkinthisspace
isvaried,evenseeminglycontradictory,andstatedmotiva- internet. Sinceexistingdatasetsdonotadequatelyreflect
thispossibility,consideringresultsonlyonthemwouldun-
| tionsarediverse. | Zhangetal.(2020),Gomezetal.(2017), |     |     |     |     |     |     |     |
| ---------------- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Joulinetal.(2016),andDesai&Johnson(2020)allintro- derestimatethepotentialofthislineofresearch. Toaddress
this,weconstructedanewdatasetof400million(image,
ducemethodswhichlearnvisualrepresentationsfromtext
|     |     |     |     |     | text) pairs | collected form | a variety of publicly | available |
| --- | --- | --- | --- | --- | ----------- | -------------- | --------------------- | --------- |
pairedwithimagesbutdescribetheirapproachesasunsuper-
vised,self-supervised,weaklysupervised,andsupervised sourcesontheInternet. Toattempttocoverasbroadaset
ofvisualconceptsaspossible,wesearchfor(image,text)
respectively.
pairsaspartoftheconstructionprocesswhosetextincludes
Weemphasizethatwhatiscommonacrossthislineofwork
|     |     |     |     |     | oneofasetof500,000queries.1 |     | Weapproximatelyclass |     |
| --- | --- | --- | --- | --- | --------------------------- | --- | -------------------- | --- |
isnotanyofthedetailsoftheparticularmethodsusedbut
theappreciationofnaturallanguageasatrainingsignal. All 1Thebasequerylistisallwordsoccurringatleast100timesin
theseapproachesarelearningfromnaturallanguagesuper- theEnglishversionofWikipedia.Thisisaugmentedwithbi-grams

4
LearningTransferableVisualModelsFromNaturalLanguageSupervision
balancetheresultsbyincludingupto20,000(image,text) multi-modalembeddingspacebyjointlytraininganimage
pairs per query. The resulting dataset has a similar total encoderandtextencodertomaximizethecosinesimilar-
wordcountastheWebTextdatasetusedtotrainGPT-2. We ity of the image and text embeddings of the N real pairs
refertothisdatasetasWITforWebImageText. inthebatchwhileminimizingthecosinesimilarityofthe
|     |     |     |     |     |     |     |     | embeddings | of the N2 | −N  | incorrect | pairings. | We  | opti- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | --- | --------- | --------- | --- | ----- |
2.3.SelectinganEfficientPre-TrainingMethod mizeasymmetriccrossentropylossoverthesesimilarity
|                  |     |          |        |         |     |          |       | scores. InFigure3weincludepseudocodeofthecoreofan |     |     |     |     |     |     |
| ---------------- | --- | -------- | ------ | ------- | --- | -------- | ----- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| State-of-the-art |     | computer | vision | systems |     | use very | large |                                                   |     |     |     |     |     |     |
implementationofCLIP.Toourknowledgethisbatchcon-
| amounts | of compute. |     | Mahajan | et al. | (2018) | required | 19  |     |     |     |     |     |     |     |
| ------- | ----------- | --- | ------- | ------ | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
structiontechniqueandobjectivewasfirstintroducedinthe
GPUyearstotraintheirResNeXt101-32x48dandXieetal. areaofdeepmetriclearningasthemulti-classN-pairloss
(2020)required33TPUv3core-yearstotraintheirNoisy
Sohn(2016),waspopularizedforcontrastiverepresentation
| StudentEfficientNet-L2. |     |     | Whenconsideringthatboththese |     |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
learningbyOordetal.(2018)astheInfoNCEloss,andwas
systemsweretrainedtopredictonly1000ImageNetclasses,
recentlyadaptedforcontrastive(text,image)representation
| the task | of learning | an  | open | set of | visual | concepts | from |     |     |     |     |     |     |     |
| -------- | ----------- | --- | ---- | ------ | ------ | -------- | ---- | --- | --- | --- | --- | --- | --- | --- |
learninginthedomainofmedicalimagingbyZhangetal.
| naturallanguageseemsdaunting. |     |     |     | Inthecourseofouref- |     |     |     | (2020). |     |     |     |     |     |     |
| ----------------------------- | --- | --- | --- | ------------------- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
forts,wefoundtrainingefficiencywaskeytosuccessfully
scalingnaturallanguagesupervisionandweselectedour Duetothelargesizeofourpre-trainingdataset,over-fitting
isnotamajorconcernandthedetailsoftrainingCLIPare
finalpre-trainingmethodbasedonthismetric.
simplifiedcomparedtotheimplementationofZhangetal.
| Ourinitialapproach, |     | similartoVirTex, |     |     | jointlytrainedan |     |     |                                                      |     |     |     |     |     |     |
| ------------------- | --- | ---------------- | --- | --- | ---------------- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                     |     |                  |     |     |                  |     |     | (2020). WetrainCLIPfromscratchwithoutinitializingthe |     |     |     |     |     |     |
imageCNNandtexttransformerfromscratchtopredictthe imageencoderwithImageNetweightsorthetextencoder
| captionofanimage.             |     | However,weencountereddifficulties |     |                      |     |     |     |                  |          |     |       |         |                |     |
| ----------------------------- | --- | --------------------------------- | --- | -------------------- | --- | --- | --- | ---------------- | -------- | --- | ----- | ------- | -------------- | --- |
|                               |     |                                   |     |                      |     |     |     | with pre-trained | weights. |     | We do | not use | the non-linear |     |
| efficientlyscalingthismethod. |     |                                   |     | InFigure2weshowthata |     |     |     |                  |          |     |       |         |                |     |
projectionbetweentherepresentationandthecontrastive
63millionparametertransformerlanguagemodel,which
embeddingspace,achangewhichwasintroducedbyBach-
| already | uses twice | the | compute | of  | its ResNet-50 |     | image |     |     |     |     |     |     |     |
| ------- | ---------- | --- | ------- | --- | ------------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
manetal.(2019)andpopularizedbyChenetal.(2020b).
encoder,learnstorecognizeImageNetclassesthreetimes Weinsteaduseonlyalinearprojectiontomapfromeachen-
slowerthanamuchsimplerbaselinethatpredictsabag-of-
coder’srepresentationtothemulti-modalembeddingspace.
wordsencodingofthesametext.
Wedidnotnoticeadifferenceintrainingefficiencybetween
Boththeseapproachesshareakeysimilarity.Theytrytopre- thetwoversionsandspeculatethatnon-linearprojections
dicttheexactwordsofthetextaccompanyingeachimage. may be co-adapted with details of current image only in
|     |     |     |     |     |     |     |     | self-supervisedrepresentationlearningmethods. |     |     |     |     | Wealso |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | ------ | --- |
Thisisadifficulttaskduetothewidevarietyofdescriptions,
comments,andrelatedtextthatco-occurwithimages. Re- removethetexttransformationfunctiont fromZhangetal.
u
|     |     |     |     |     |     |     |     | (2020) which | samples | a single | sentence | at  | uniform | from |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------- | -------- | -------- | --- | ------- | ---- |
centworkincontrastiverepresentationlearningforimages
hasfoundthatcontrastiveobjectivescanlearnbetterrepre- thetextsincemanyofthe(image,text)pairsinCLIP’spre-
sentationsthantheirequivalentpredictiveobjective(Tian trainingdatasetareonlyasinglesentence. Wealsosimplify
|              |                                         |     |     |     |     |     |     | the image | transformation |     | function | t . A | random | square |
| ------------ | --------------------------------------- | --- | --- | --- | --- | --- | --- | --------- | -------------- | --- | -------- | ----- | ------ | ------ |
| etal.,2019). | Otherworkhasfoundthatalthoughgenerative |     |     |     |     |     |     |           |                |     |          | v     |        |        |
modelsofimagescanlearnhighqualityimagerepresenta- crop from resized images is the only data augmentation
|     |     |     |     |     |     |     |     | used during | training. | Finally, | the | temperature | parameter |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | -------- | --- | ----------- | --------- | --- |
tions,theyrequireoveranorderofmagnitudemorecompute
whichcontrolstherangeofthelogitsinthesoftmax,τ,is
thancontrastivemodelswiththesameperformance(Chen
etal.,2020a). Notingthesefindings,weexploredtraining directlyoptimizedduringtrainingasalog-parameterized
multiplicativescalartoavoidturningasahyper-parameter.
asystemtosolvethepotentiallyeasierproxytaskofpre-
| dicting only | which | text | as a | whole | is paired | with | which |     |     |     |     |     |     |     |
| ------------ | ----- | ---- | ---- | ----- | --------- | ---- | ----- | --- | --- | --- | --- | --- | --- | --- |
imageandnottheexactwordsofthattext. Startingwith 2.4.ChoosingandScalingaModel
thesamebag-of-wordsencodingbaseline,weswappedthe
|     |     |     |     |     |     |     |     | We consider | two different |     | architectures | for | the image | en- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------------- | --- | ------------- | --- | --------- | --- |
predictiveobjectiveforacontrastiveobjectiveinFigure2
|     |     |     |     |     |     |     |     | coder. For | the first, | we use | ResNet-50 | (He | et al., | 2016a) |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---------- | ------ | --------- | --- | ------- | ------ |
andobservedafurther4xefficiencyimprovementintherate
|     |     |     |     |     |     |     |     | as the base | architecture | for | the image | encoder | due | to its |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------------ | --- | --------- | ------- | --- | ------ |
ofzero-shottransfertoImageNet.
widespreadadoptionandprovenperformance.Wemakesev-
GivenabatchofN (image,text)pairs,CLIPistrainedto eralmodificationstotheoriginalversionusingtheResNet-
predictwhichoftheN ×N possible(image,text)pairings DimprovementsfromHeetal.(2019)andtheantialiased
acrossabatchactuallyoccurred. Todothis,CLIPlearnsa rect-2 blur pooling from Zhang (2019). We also replace
theglobalaveragepoolinglayerwithanattentionpooling
withhighpointwisemutualinformationaswellasthenamesof
|                                                |     |     |     |     |     |            |     | mechanism. | Theattentionpoolingisimplementedasasin- |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | ---------- | --- | ---------- | --------------------------------------- | --- | --- | --- | --- | --- |
| allWikipediaarticlesaboveacertainsearchvolume. |     |     |     |     |     | Finallyall |     |            |                                         |     |     |     |     |     |
WordNetsynsetsnotalreadyinthequerylistareadded. glelayerof“transformer-style”multi-headQKVattention
wherethequeryisconditionedontheglobalaverage-pooled

5
LearningTransferableVisualModelsFromNaturalLanguageSupervision
# image_encoder - ResNet or Vision Transformer onedimensionofthemodel. WhileTan&Le(2019)tune
# text_encoder  - CBOW or Text Transformer
theratioofcomputeallocatedtoeachdimensionfortheir
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts EfficientNetarchitecture,weuseasimplebaselineofallo-
# W_i[d_i, d_e] - learned proj of image to embed catingadditionalcomputeequallytoincreasingthewidth,
# W_t[d_t, d_e] - learned proj of text to embed depth,andresolutionofthemodel. Forthetextencoder,we
# t             - learned temperature parameter
onlyscalethewidthofthemodeltobeproportionaltothe
# extract feature representations of each modality calculatedincreaseinwidthoftheResNetanddonotscale
I_f = image_encoder(I) #[n, d_i]
thedepthatall,aswefoundCLIP’sperformancetobeless
T_f = text_encoder(T)  #[n, d_t]
sensitivetothecapacityofthetextencoder.
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
2.5.Training
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
Wetrainaseriesof5ResNetsand3VisionTransformers.
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t) FortheResNetswetrainaResNet-50,aResNet-101,and
then3morewhichfollowEfficientNet-stylemodelscaling
# symmetric loss function
anduseapproximately4x,16x,and64xthecomputeofa
labels = np.arange(n)
|     |     |     |     | ResNet-50. | TheyaredenotedasRN50x4,RN50x16,and |     |     |     |     |
| --- | --- | --- | --- | ---------- | ---------------------------------- | --- | --- | --- | --- |
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
|     |     |     |     | RN50x64 | respectively. | For | the Vision | Transformers | we  |
| --- | --- | --- | --- | ------- | ------------- | --- | ---------- | ------------ | --- |
loss   = (loss_i + loss_t)/2
|     |     |     |     | trainaViT-B/32,aViT-B/16,andaViT-L/14. |     |                              |     | Wetrainall |     |
| --- | --- | --- | --- | -------------------------------------- | --- | ---------------------------- | --- | ---------- | --- |
|     |     |     |     | modelsfor32epochs.                     |     | WeusetheAdamoptimizer(Kingma |     |            |     |
Figure3.Numpy-likepseudocodeforthecoreofanimplementa-
|     |     |     |     | & Ba, 2014) | with decoupled |     | weight | decay regularization |     |
| --- | --- | --- | --- | ----------- | -------------- | --- | ------ | -------------------- | --- |
tionofCLIP.
(Loshchilov&Hutter,2017)appliedtoallweightsthatare
|     |     |     |     | not gains                               | or biases, | and decay | the | learning rate | using a |
| --- | --- | --- | --- | --------------------------------------- | ---------- | --------- | --- | ------------- | ------- |
|     |     |     |     | cosineschedule(Loshchilov&Hutter,2016). |            |           |     | Initialhyper- |         |
representationoftheimage. Forthesecondarchitecture,we parametersweresetusingacombinationofgridsearches,
experimentwiththerecentlyintroducedVisionTransformer
randomsearch,andmanualtuningonthebaselineResNet-
(ViT) (Dosovitskiy et al., 2020). We closely follow their 50modelwhentrainedfor1epoch. Hyper-parameterswere
implementationwithonlytheminormodificationofadding
thenadaptedheuristicallyforlargermodelsduetocompu-
| an additional | layer normalization | to the combined | patch |                      |     |                                  |     |     |     |
| ------------- | ------------------- | --------------- | ----- | -------------------- | --- | -------------------------------- | --- | --- | --- |
|               |                     |                 |       | tationalconstraints. |     | Thelearnabletemperatureparameter |     |     |     |
andpositionembeddingsbeforethetransformerandusea τ wasinitializedtotheequivalentof0.07from(Wuetal.,
slightlydifferentinitializationscheme.
|     |     |     |     | 2018) and | clipped to | prevent | scaling | the logits | by more |
| --- | --- | --- | --- | --------- | ---------- | ------- | ------- | ---------- | ------- |
than100whichwefoundnecessarytopreventtrainingin-
| The text encoder | is a Transformer | (Vaswani | et al., 2017) |     |     |     |     |     |     |
| ---------------- | ---------------- | -------- | ------------- | --- | --- | --- | --- | --- | --- |
with the architecture modifications described in Radford stability. We use a very large minibatch size of 32,768.
etal.(2019). Asabasesizeweusea63M-parameter12- Mixed-precision(Micikeviciusetal.,2017)wasusedtoac-
|                |            |                    |            | celeratetrainingandsavememory. |     |     | Tosaveadditionalmem- |     |     |
| -------------- | ---------- | ------------------ | ---------- | ------------------------------ | --- | --- | -------------------- | --- | --- |
| layer 512-wide | model with | 8 attention heads. | The trans- |                                |     |     |                      |     |     |
formeroperatesonalower-casedbytepairencoding(BPE) ory, gradient checkpointing (Griewank & Walther, 2000;
Chenetal.,2016),half-precisionAdamstatistics(Dhariwal
| representation | of the text | with a 49,152 vocab | size (Sen- |     |     |     |     |     |     |
| -------------- | ----------- | ------------------- | ---------- | --- | --- | --- | --- | --- | --- |
nrichetal.,2015). Forcomputationalefficiency,themax etal.,2020),andhalf-precisionstochasticallyroundedtext
sequence length was capped at 76. The text sequence is encoderweightswereused. Thecalculationofembedding
similaritieswasalsoshardedwithindividualGPUscomput-
bracketedwith[SOS]and[EOS]tokensandtheactiva-
tionsofthehighestlayerofthetransformeratthe[EOS] ingonlythesubsetofthepairwisesimilaritiesnecessaryfor
|           |                        |                |             | theirlocalbatchofembeddings. |     |     | ThelargestResNetmodel, |     |     |
| --------- | ---------------------- | -------------- | ----------- | ---------------------------- | --- | --- | ---------------------- | --- | --- |
| token are | treated as the feature | representation | of the text |                              |     |     |                        |     |     |
RN50x64,took18daystotrainon592V100GPUswhile
whichislayernormalizedandthenlinearlyprojectedinto
themulti-modalembeddingspace. Maskedself-attention thelargestVisionTransformertook12dayson256V100
GPUs. FortheViT-L/14wealsopre-trainatahigher336
wasusedinthetextencodertopreservetheabilitytoini-
tializewithapre-trainedlanguagemodeloraddlanguage pixel resolution for one additional epoch to boost perfor-
|     |     |     |     | mancesimilartoFixRes(Touvronetal.,2019). |     |     |     | Wedenote |     |
| --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | -------- | --- |
modelingasanauxiliaryobjective,thoughexplorationof
|     |     |     |     | thismodelasViT-L/14@336px. |     |     | Unlessotherwisespecified, |     |     |
| --- | --- | --- | --- | -------------------------- | --- | --- | ------------------------- | --- | --- |
thisisleftasfuturework.
allresultsreportedinthispaperas“CLIP”usethismodel
Whilepreviouscomputervisionresearchhasoftenscaled
whichwefoundtoperformbest.
modelsbyincreasingthewidth(Mahajanetal.,2018)or
depth(Heetal.,2016a)inisolation,fortheResNetimage
encodersweadapttheapproachofTan&Le(2019)which
foundthatallocatingadditionalcomputeacrossallofwidth,
depth,andresolutionoutperformsonlyallocatingittoonly

LearningTransferableVisualModelsFromNaturalLanguageSupervision 6
3.Experiments trainingasatransferlearningmethodtoimprovesupervised
fine-tuning,italsoincludedanablationstudydemonstrat-
3.1.Zero-ShotTransfer
ingthattheperformanceoffourheuristiczero-shottransfer
3.1.1.MOTIVATION methodsimprovedsteadilyoverthecourseofpre-training,
withoutanysupervisedadaption.Thisanalysisservedasthe
Incomputervision,zero-shotlearningusuallyreferstothe
basisforGPT-2(Radfordetal.,2019)whichfocusedexclu-
studyofgeneralizingtounseenobjectcategoriesinimage
sivelyonstudyingthetask-learningcapabilitiesoflanguage
classification (Lampert et al., 2009). We instead use the
modelsviazero-shottransfer.
terminabroadersenseandstudygeneralizationtounseen
datasets. We motivate this as a proxy for performing un- 3.1.2.USINGCLIPFORZERO-SHOTTRANSFER
seentasks,asaspiredtointhezero-datalearningpaperof
Larochelleetal.(2008). Whilemuchresearchinthefieldof CLIPispre-trainedtopredictifanimageandatextsnippet
unsupervisedlearningfocusesontherepresentationlearn- arepairedtogetherinitsdataset. Toperformzero-shotclas-
ingcapabilitiesofmachinelearningsystems,wemotivate sification,wereusethiscapability. Foreachdataset,weuse
studyingzero-shottransferasawayofmeasuringthetask- thenamesofalltheclassesinthedatasetasthesetofpoten-
learningcapabilitiesofmachinelearningsystems. Inthis tialtextpairingsandpredictthemostprobable(image,text)
view, a dataset evaluates performance on a task on a spe- pairaccordingtoCLIP.Inabitmoredetail,wefirstcompute
cificdistribution. However,manypopularcomputervision thefeatureembeddingoftheimageandthefeatureembed-
datasetswerecreatedbytheresearchcommunityprimarily dingofthesetofpossibletextsbytheirrespectiveencoders.
asbenchmarkstoguidethedevelopmentofgenericimage Thecosinesimilarityoftheseembeddingsisthencalculated,
classificationmethodsratherthanmeasuringperformance scaledbyatemperatureparameterτ,andnormalizedintoa
on a specific task. While it is reasonable to say that the probabilitydistributionviaasoftmax. Notethatthispredic-
SVHNdatasetmeasuresthetaskofstreetnumbertranscrip- tionlayerisamultinomiallogisticregressionclassifierwith
tiononthedistributionofGoogleStreetViewphotos,itis L2-normalizedinputs,L2-normalizedweights,nobias,and
unclear what “real” task the CIFAR-10 dataset measures. temperaturescaling. Wheninterpretedthisway,theimage
Itisclear,however,whatdistributionCIFAR-10isdrawn encoderisthecomputervisionbackbonewhichcomputesa
from-TinyImages(Torralbaetal.,2008). Onthesekindsof featurerepresentationfortheimageandthetextencoderisa
datasets,zero-shottransferismoreanevaluationofCLIP’s hypernetwork(Haetal.,2016)whichgeneratestheweights
robustnesstodistributionshiftanddomaingeneralization ofalinearclassifierbasedonthetextspecifyingthevisual
ratherthantaskgeneralization. PleaseseeSection3.3for conceptsthattheclassesrepresent. LeiBaetal.(2015)first
analysisfocusedonthis. introducedazero-shotimageclassifierofthisformwhile
the idea of generating a classifier from natural language
To our knowledge, Visual N-Grams (Li et al., 2017) first
datesbacktoatleastElhoseinyetal.(2013). Continuing
studiedzero-shottransfertoexistingimageclassification
withthisinterpretation,everystepofCLIPpre-trainingcan
datasetsinthemannerdescribedabove. Itisalsotheonly
be viewed as optimizing the performance of a randomly
otherworkweareawareofthathasstudiedzero-shottrans-
createdproxytoacomputervisiondatasetwhichcontains1
fertostandardimageclassificationdatasetsusingagener-
exampleperclassandhas32,768totalclassesdefinedvia
ically pre-trained model and serves as the best reference
naturallanguagedescriptions. Forzero-shotevaluation,we
pointforcontextualizingCLIP.Theirapproachlearnsthe
cachethezero-shotclassifieronceithasbeencomputedby
parametersofadictionaryof142,806visualn-grams(span-
thetextencoderandreuseitforallsubsequentpredictions.
ning1-to5-grams)andoptimizesthesen-gramsusinga
Thisallowsthecostofgeneratingittobeamortizedacross
differentialversionofJelinek-Mercersmoothingtomaxi-
allthepredictionsinadataset.
mizetheprobabilityofalltextn-gramsforagivenimage.
Inordertoperformzero-shottransfer,theyfirstconvertthe
3.1.3.INITIALCOMPARISONTOVISUALN-GRAMS
text of each of the dataset’s class names into its n-gram
representationandthencomputeitsprobabilityaccording InTable1wecompareVisualN-GramstoCLIP.Thebest
totheirmodel,predictingtheonewiththehighestscore. CLIPmodelimprovesaccuracyonImageNetfromaproof
ofconcept11.5%to76.2%andmatchestheperformance
Ourfocusonstudyingzero-shottransferasanevaluationof
of the original ResNet-50 despite using none of the 1.28
tasklearningisinspiredbyworkdemonstratingtasklearn-
millioncrowd-labeledtrainingexamplesavailableforthis
inginthefieldofNLP.ToourknowledgeLiuetal.(2018)
dataset. Additionally,thetop-5accuracyofCLIPmodels
firstidentifiedtasklearningasan“unexpectedside-effect”
arenoticeablyhigherthantheirtop-1,andthismodelhasa
whenalanguagemodeltrainedtogenerateWikipediaar-
95%top-5accuracy,matchingInception-V4(Szegedyetal.,
ticles learned to reliably transliterate names between lan-
2016). The ability to match the performance of a strong,
guages. WhileGPT-1(Radfordetal.,2018)focusedonpre-
fully supervised baselines in a zero-shot setting suggests

7
LearningTransferableVisualModelsFromNaturalLanguageSupervision
70
RN50x64
|               |     | aYahoo |      | ImageNet | SUN  |     |     |     |     |     |
| ------------- | --- | ------ | ---- | -------- | ---- | --- | --- | --- | --- | --- |
| VisualN-Grams |     |        | 72.4 | 11.5     | 23.0 |     |     |     |     |     |
| CLIP          |     |        | 98.4 | 76.2     | 58.5 |     | 65  |     |     |     |
Table1.ComparingCLIPtopriorzero-shottransferimageclassi- )%( erocS egarevA 5 point
improvement
ficationresults.CLIPimprovesperformanceonallthreedatasets
60
| byalargeamount. |     | Thisimprovementreflectsmanydifferences |     |     |     |     |     |     |     |     |
| --------------- | --- | -------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
inthe4yearssincethedevelopmentofVisualN-Grams(Lietal., RN50x16
| 2017). |     |     |     |     |     |     |     | 4X efficiency gain |     |     |
| ------ | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- |
55
RN50x4
| CLIP is                             | a significant | step | towards | flexible          | and practical |     |     |       |     |     |
| ----------------------------------- | ------------- | ---- | ------- | ----------------- | ------------- | --- | --- | ----- | --- | --- |
| zero-shotcomputervisionclassifiers. |               |      |         | Asmentionedabove, |               |     |     |       |     |     |
|                                     |               |      |         |                   |               |     | 50  | RN101 |     |     |
thecomparisontoVisualN-Gramsismeantforcontextu-
| alizing the | performance | of  | CLIP | and should | not be inter- |     | RN50 |     |     |     |
| ----------- | ----------- | --- | ---- | ---------- | ------------- | --- | ---- | --- | --- | --- |
Prompt engineering and ensembling
pretedasadirectmethodscomparisonbetweenCLIPand Contextless class names (Li et al. 2017)
45
VisualN-Gramsasmanyperformancerelevantdifferences
|         |                 |     |          |            |              |     | 6.1 | 9.9 21.5 | 75.3 | 265.9 |
| ------- | --------------- | --- | -------- | ---------- | ------------ | --- | --- | -------- | ---- | ----- |
| between | the two systems |     | were not | controlled | for. For in- |     |     |          |      |       |
Model GFLOPs
stance,wetrainonadatasetthatis10xlarger,useavision
modelthatrequiresnearly100xmorecomputeperpredic- Figure4.Prompt engineering and ensembling improve zero-
tion, likely used over 1000x their training compute, and shotperformance.Comparedtothebaselineofusingcontextless
classnames,promptengineeringandensemblingboostzero-shot
| use a transformer-based |     | model | which | did | not exist when |     |     |     |     |     |
| ----------------------- | --- | ----- | ----- | --- | -------------- | --- | --- | --- | --- | --- |
classificationperformancebyalmost5pointsonaverageacross
| VisualN-Gramswaspublished. |     |     | Asaclosercomparison,we |     |     |             |     |                                            |     |     |
| -------------------------- | --- | --- | ---------------------- | --- | --- | ----------- | --- | ------------------------------------------ | --- | --- |
|                            |     |     |                        |     |     | 36datasets. |     | Thisimprovementissimilartothegainfromusing |     |     |
trainedaCLIPResNet-50onthesameYFCC100Mdataset
4timesmorecomputewiththebaselinezero-shotmethodbutis
thatVisualN-Gramswastrainedonandfounditmatched
“free”whenamortizedovermanypredictions.
theirreportedImageNetperformancewithinaV100GPU
day. Thisbaselinewasalsotrainedfromscratchinsteadof
beinginitializedfrompre-trainedImageNetweightsasin
VisualN-Grams.
chosensomewhathaphazardlyanddonotanticipateissues
relatedtozero-shottransferwhichreliesontaskdescription
CLIPalsooutperformsVisualN-Gramsontheother2re-
porteddatasets.OnaYahoo,CLIPachievesa95%reduction inordertotransfersuccessfully.
inthenumberoferrors,andonSUN,CLIPmorethandou- Acommonissueispolysemy. Whenthenameofaclass
| blestheaccuracyofVisualN-Grams. |     |     |     | Toconductamore |     |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
istheonlyinformationprovidedtoCLIP’stextencoderit
comprehensive analysis and stress test, we implement a isunabletodifferentiatewhichwordsenseismeantdueto
| much larger | evaluation | suite | detailed | in Appendix | A. In |                   |     |                                  |     |     |
| ----------- | ---------- | ----- | -------- | ----------- | ----- | ----------------- | --- | -------------------------------- | --- | --- |
|             |            |       |          |             |       | thelackofcontext. |     | Insomecasesmultiplemeaningsofthe |     |     |
totalweexpandfromthe3datasetsreportedinVisualN-
samewordmightbeincludedasdifferentclassesinthesame
Gramstoincludeover30datasetsandcomparetoover50 dataset! This happens in ImageNet which contains both
existingcomputervisionsystemstocontextualizeresults.
|     |     |     |     |     |     | constructioncranesandcranesthatfly. |     |     | Anotherexampleis |     |
| --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | ---------------- | --- |
foundinclassesoftheOxford-IIITPetdatasetwherethe
3.1.4.PROMPTENGINEERINGANDENSEMBLING
wordboxeris,fromcontext,clearlyreferringtoabreedof
dog,buttoatextencoderlackingcontextcouldjustaslikely
Moststandardimageclassificationdatasetstreattheinfor-
refertoatypeofathlete.
mationnamingordescribingclasseswhichenablesnatural
| languagebasedzero-shottransferasanafterthought. |     |     |     |     | The |     |     |     |     |     |
| ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Anotherissueweencounteredisthatit’srelativelyrarein
vastmajorityofdatasetsannotateimageswithjustanumeric ourpre-trainingdatasetforthetextpairedwiththeimage
idofthelabelandcontainafilemappingtheseidsbackto
|                      |     |                               |     |     |     | tobejustasingleword. |     | Usuallythetextisafullsentence |              |             |
| -------------------- | --- | ----------------------------- | --- | --- | --- | -------------------- | --- | ----------------------------- | ------------ | ----------- |
| theirnamesinEnglish. |     | Somedatasets,suchasFlowers102 |     |     |     |                      |     |                               |              |             |
|                      |     |                               |     |     |     | describing           |     | the image in some             | way. To help | bridge this |
and GTSRB, don’t appear to include this mapping at all distributiongap,wefoundthatusingtheprompttemplate
intheirreleasedversionspreventingzero-shottransferen-
|     |     |     |     |     |     | “A  | photo | of a {label}.” | tobeagooddefaultthat |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | -------------- | -------------------- | --- |
tirely.2
Formanydatasets,weobservedtheselabelsmaybe helpsspecifythetextisaboutthecontentoftheimage. This
2AleclearnedmuchmoreaboutflowerspeciesandGerman oftenimprovesperformanceoverthebaselineofusingonly
|     |     |     |     |     |     | thelabeltext. |     | Forinstance,justusingthispromptimproves |     |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | --------------------------------------- | --- | --- |
trafficsignsoverthecourseofthisprojectthanheoriginallyantic-
accuracyonImageNetby1.3%.
ipated.

8
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Similartothe“promptengineering”discussionaroundGPT- StanfordCars +28.9
3 (Brown et al., 2020; Gao et al., 2020), we have also Country211 +23.2
|     |     |     |     |     |     |     | Food101 |     |     | +22.5 |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | ----- | --- |
observed that zero-shot performance can be significantly Kinetics700 +14.5
| improvedbycustomizingtheprompttexttoeachtask. |     |     |     |     | A   |     | SST2 |     | +12.4 |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- | --- | ----- | --- | --- |
few,nonexhaustive,examplesfollow. Wefoundonseveral SUN397 +7.8
|     |     |     |     |     |     |     | UCF101 |     | +7.7 |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | ---- | --- | --- |
fine-grainedimageclassificationdatasetsthatithelpedto HatefulMemes +6.7
specifythecategory. ForexampleonOxford-IIITPets,us- CIFAR10 +3.9
|             |     |            |        |     |       |     | CIFAR100 | +3.0 |     |     |     |
| ----------- | --- | ---------- | ------ | --- | ----- | --- | -------- | ---- | --- | --- | --- |
| ing“A photo | of  | a {label}, | a type | of  | pet.” |     |          |      |     |     |     |
|             |     |            |        |     |       |     | STL10    | +3.0 |     |     |     |
tohelpprovidecontextworkedwell. Likewise,onFood101 FER2013 +2.8
|     |     |     |     |     |     | Caltech101 |     | +2.0 |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------- | --- | ---- | --- | --- | --- |
specifyingatypeoffoodandonFGVCAircraftatypeof
|     |     |     |     |     |     |     | ImageNet | +1.9 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---- | --- | --- | --- |
aircrafthelpedtoo.ForOCRdatasets,wefoundthatputting OxfordPets +1.1
|     |     |     |     |     |     | PascalVOC2007 |     | +0.5 |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ---- | --- | --- | --- |
quotesaroundthetextornumbertoberecognizedimproved
|     |     |     |     |     |     |     | -3.2 | Birdsnap |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | -------- | --- | --- | --- |
performance.Finally,wefoundthatonsatelliteimageclassi- -10.0 MNIST
|     |     |     |     |     |     | -11.3 |     | FGVCAircraft |     |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | ------------ | --- | --- | --- |
ficationdatasetsithelpedtospecifythattheimageswereof
|                              |     |     |           |       |     | -11.9 |     | RESISC45   |     |     |     |
| ---------------------------- | --- | --- | --------- | ----- | --- | ----- | --- | ---------- | --- | --- | --- |
| thisformandweusevariantsof“a |     |     | satellite | photo |     |       |     |            |     |     |     |
|                              |     |     |           |       |     | -12.5 |     | Flowers102 |     |     |     |
|                              |     |     |           |       |     | -16.6 |     | DTD        |     |     |     |
of a {label}.”.
|     |     |     |     |     |     | -18.2 |     | CLEVRCounts |     |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | ----------- | --- | --- | --- |
Wealsoexperimentedwithensemblingovermultiplezero- -18.4 GTSRB
|     |     |     |     |     |     | -19.5 |     | PatchCamelyon |     |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | ------------- | --- | --- | --- |
shotclassifiersasanotherwayofimprovingperformance. -34.0 KITTI Distance
Theseclassifiersarecomputedbyusingdifferentcontext -37.1 EuroSAT
promptssuchas‘A photo of a big {label}”and 40 30 20 10 0 10 20 30 40
| “A photo | of a | small {label}”. | Weconstructthe |     |     |     |  Score (%) |     |     |     |     |
| -------- | ---- | --------------- | -------------- | --- | --- | --- | ---------- | --- | --- | --- | --- |
ensembleovertheembeddingspaceinsteadofprobability Zero-Shot CLIP vs. Linear Probe on ResNet50
space. Thisallowsustocacheasinglesetofaveragedtext
|     |     |     |     |     | Figure5.Zero-shot |     | CLIP is | competitive | with | a fully | super- |
| --- | --- | --- | --- | --- | ----------------- | --- | ------- | ----------- | ---- | ------- | ------ |
embeddingssothatthecomputecostoftheensembleisthe
|     |     |     |     |     | visedbaseline. | Acrossa27datasetevalsuite,azero-shotCLIP |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------------- | ---------------------------------------- | --- | --- | --- | --- | --- |
sameasusingasingleclassifierwhenamortizedovermany
classifieroutperformsafullysupervisedlinearclassifierfittedon
predictions. We’veobservedensemblingacrossmanygen- ResNet-50featureson16datasets,includingImageNet.
eratedzero-shotclassifierstoreliablyimproveperformance
| anduseitforthemajorityofdatasets. |     |     | OnImageNet, |     | we  |     |     |     |     |     |     |
| --------------------------------- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
ensemble80differentcontextpromptsandthisimproves
|     |     |     |     |     | tenthannotandwinson16ofthe27datasets. |     |     |     |     | Lookingat |     |
| --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --------- | --- |
performancebyanadditional3.5%overthesingledefault
|                       |     |                               |     |     | individualdatasetsrevealssomeinterestingbehavior. |     |     |     |     |     | On  |
| --------------------- | --- | ----------------------------- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| promptdiscussedabove. |     | Whenconsideredtogether,prompt |     |     |                                                   |     |     |     |     |     |     |
fine-grainedclassificationtasks,weobserveawidespread
| engineering | and ensembling | improve | ImageNet | accuracy |     |     |     |     |     |     |     |
| ----------- | -------------- | ------- | -------- | -------- | --- | --- | --- | --- | --- | --- | --- |
inperformance.Ontwoofthesedatasets,StanfordCarsand
byalmost5%. InFigure4wevisualizehowpromptengi- Food101,zero-shotCLIPoutperformslogisticregression
neeringandensemblingchangetheperformanceofasetof
|     |     |     |     |     | on ResNet-50 | features | by  | over 20% | while | on two | others, |
| --- | --- | --- | --- | --- | ------------ | -------- | --- | -------- | ----- | ------ | ------- |
CLIPmodelscomparedtothecontextlessbaselineapproach
Flowers102andFGVCAircraft,zero-shotCLIPunderper-
| of directly | embedding | the class | name as done | in Li | et al.   |           |     |            |     |           |      |
| ----------- | --------- | --------- | ------------ | ----- | -------- | --------- | --- | ---------- | --- | --------- | ---- |
|             |           |           |              |       | forms by | over 10%. | On  | OxfordPets | and | Birdsnap, | per- |
(2017).
|     |     |     |     |     | formanceismuchcloser. |     |     | Wesuspectthesedifferenceare |     |     |     |
| --- | --- | --- | --- | --- | --------------------- | --- | --- | --------------------------- | --- | --- | --- |
primarilyduetovaryingamountsofper-tasksupervision
3.1.5.ANALYSISOFZERO-SHOTCLIPPERFORMANCE
|     |     |     |     |     | betweenWITandImageNet. |     |     | On“general”objectclassifica- |     |     |     |
| --- | --- | --- | --- | --- | ---------------------- | --- | --- | ---------------------------- | --- | --- | --- |
tiondatasetssuchasImageNet,CIFAR10/100,STL10,and
Sincetask-agnosticzero-shotclassifiersforcomputervision
havebeenunderstudied,CLIPprovidesapromisingoppor- PascalVOC2007 performance is relatively similar with a
tunitytogainabetterunderstandingofthistypeofmodel. slightadvantageforzero-shotCLIPinallcases. OnSTL10,
|     |     |     |     |     | CLIP achieves | 99.3% | overall | which | appears | to  | be a new |
| --- | --- | --- | --- | --- | ------------- | ----- | ------- | ----- | ------- | --- | -------- |
Inthissection,weconductastudyofvariouspropertiesof
CLIP’s zero-shot classifiers. As a first question, we look stateoftheartdespitenotusinganytrainingexamples.Zero-
|                                             |     |     |     |        | shot CLIP | significantly | outperforms |     | a ResNet-50 |     | on two |
| ------------------------------------------- | --- | --- | --- | ------ | --------- | ------------- | ----------- | --- | ----------- | --- | ------ |
| simplyathowwellzero-shotclassifiersperform. |     |     |     | Tocon- |           |               |             |     |             |     |        |
textualizethis,wecomparetotheperformanceofasimple datasetsmeasuringactionrecognitioninvideos. OnKinet-
off-the-shelfbaseline: fittingafullysupervised,regularized, ics700, CLIP outperforms a ResNet-50 by 14.5%. Zero-
shotCLIPalsooutperformsaResNet-50’sfeaturesby7.7%
logisticregressionclassifieronthefeaturesofthecanonical
ResNet-50. InFigure5weshowthiscomparisonacross27 onUCF101. Wespeculatethisisduetonaturallanguage
datasets. PleaseseeAppendixAfordetailsofdatasetsand providingwidersupervisionforvisualconceptsinvolving
verbs,comparedtothenoun-centricobjectsupervisionin
setup.
ImageNet.
Zero-shotCLIPoutperformsthisbaselineslightlymoreof-
Lookingatwherezero-shotCLIPnotablyunderperforms,

9
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     | 75  |     |     |                   |     | expectzero-shottounderperformone-shot,weinsteadfind |     |     |     |     |     |     |
| --- | --- | --- | --- | ----------------- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | Linear Probe CLIP |     | thatzero-shotCLIPmatchestheperformanceof4-shotlo-   |     |     |     |     |     |     |
70
|     |     |     |     |     |     | gisticregressiononthesamefeaturespace. |     |     |     |     | Thisislikely |     |
| --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | ------------ | --- |
duetoanimportantdifferencebetweenthezero-shotand
65 Zero-Shot BiT-M (ImageNet-21K) few-shotapproach. First,CLIP’szero-shotclassifierisgen-
CLIP
|     |     |     |     |     | SimCLRv2 | eratedvianaturallanguagewhichallowsforvisualconcepts |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
)%( erocS egarevA 60
|     |     |     |     |     |     | to be directly | specified |     | (“communicated”). |     | By  | contrast, |
| --- | --- | --- | --- | --- | --- | -------------- | --------- | --- | ----------------- | --- | --- | --------- |
55 ResNet50 “normal”supervisedlearningmustinferconceptsindirectly
|     |     |     |     |     |     | fromtrainingexamples. |              |      | Context-lessexample-basedlearn- |           |            |     |
| --- | --- | --- | --- | --- | --- | --------------------- | ------------ | ---- | ------------------------------- | --------- | ---------- | --- |
|     |     |     |     |     |     | ing has               | the drawback | that | many                            | different | hypotheses | can |
50
beconsistentwiththedata,especiallyintheone-shotcase.
|     | 45  |     |     |     |     | A single | image | often contains | many | different | visual | con- |
| --- | --- | --- | --- | --- | --- | -------- | ----- | -------------- | ---- | --------- | ------ | ---- |
cepts. Althoughacapablelearnerisabletoexploitvisual
|     | 40  |     |     |     |     | cuesandheuristics,suchasassumingthattheconceptbeing |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
demonstratedistheprimaryobjectinanimage,thereisno
35
guarantee.
|     |     |     |     |     |     | A potential | resolution | of  | this discrepancy |     | between | zero- |
| --- | --- | --- | --- | --- | --- | ----------- | ---------- | --- | ---------------- | --- | ------- | ----- |
30
|     | 0   | 1 2 4 |     | 8   | 16  |     |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
shotandfew-shotperformanceistouseCLIP’szero-shot
# of labeled training examples per class
classifierasapriorfortheweightsofthefew-shotclassifier.
WhileaddinganL2penaltytowardsthegeneratedweights
Figure6.Zero-shotCLIPoutperformsfew-shotlinearprobes.
Zero-shotCLIPmatchestheaverageperformanceofa4-shotlinear isastraightforwardimplementationofthisidea,wefound
classifiertrainedonthesamefeaturespaceandnearlymatchesthe that hyperparameter optimization would often select for
bestresultsofa16-shotlinearclassifieracrosspubliclyavailable suchalargevalueofthisregularizerthattheresultingfew-
models. For both BiT-M and SimCLRv2, the best performing shotclassifierwas“just”thezero-shotclassifier. Research
modelishighlighted.Lightgraylinesareothermodelsintheeval intobettermethodsofcombiningthestrengthofzero-shot
suite. The20datasetswithatleast16examplesperclasswere
transferwithflexibilityoffew-shotlearningisapromising
usedinthisanalysis.
directionforfuturework.
|     |     |     |     |     |     | When comparing |     | zero-shot | CLIP | to few-shot | logistic | re- |
| --- | --- | --- | --- | --- | --- | -------------- | --- | --------- | ---- | ----------- | -------- | --- |
we see that zero-shot CLIP is quite weak on several spe- gression on the features of other models, zero-shot CLIP
cialized,complex,orabstracttaskssuchassatelliteimage
|     |     |     |     |     |     | roughly | matches | the performance |     | of the | best performing |     |
| --- | --- | --- | --- | --- | --- | ------- | ------- | --------------- | --- | ------ | --------------- | --- |
classification(EuroSATandRESISC45),lymphnodetumor 16-shotclassifierinourevaluationsuite,whichusesthefea-
detection(PatchCamelyon),countingobjectsinsynthetic turesofaBiT-MResNet-152x2trainedonImageNet-21K.
scenes(CLEVRCounts),self-drivingrelatedtaskssuchas
|     |     |     |     |     |     | We are | certain | that a BiT-L | model | trained | on  | JFT-300M |
| --- | --- | --- | --- | --- | --- | ------ | ------- | ------------ | ----- | ------- | --- | -------- |
Germantrafficsignrecognition(GTSRB),recognizingdis- wouldperformevenbetterbutthesemodelshavenotbeen
| tance | to the | nearest car | (KITTI | Distance). These | results |                   |     |                                |     |     |     |     |
| ----- | ------ | ----------- | ------ | ---------------- | ------- | ----------------- | --- | ------------------------------ | --- | --- | --- | --- |
|       |        |             |        |                  |         | publiclyreleased. |     | ThataBiT-MResNet-152x2performs |     |     |     |     |
highlight the poor capability of zero-shot CLIP on more best in a 16-shot setting is somewhat surprising since, as
complextasks.Bycontrast,non-experthumanscanrobustly analyzedinSection3.2,theNoisyStudentEfficientNet-L2
| performseveralofthesetasks, |     |     |     | suchascounting, | satellite |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | --------------- | --------- | --- | --- | --- | --- | --- | --- | --- |
outperformsitinafullysupervisedsettingbyalmost5%on
imageclassification,andtrafficsignrecognition,suggesting averageacross27datasets.
| significant |     | room for improvement. |     | However, | we caution |     |     |     |     |     |     |     |
| ----------- | --- | --------------------- | --- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
Inadditiontostudyingtheaverageperformanceofzero-shot
thatitisunclearwhethermeasuringzero-shottransfer,as
|     |     |     |     |     |     | CLIP and | few-shot | logistic | regression, | we  | also | examine |
| --- | --- | --- | --- | --- | --- | -------- | -------- | -------- | ----------- | --- | ---- | ------- |
opposedtofew-shottransfer,isameaningfulevaluationfor
|           |       |                |     |                     |       | performanceonindividualdatasets. |     |     |     | InFigure7,weshow |     |     |
| --------- | ----- | -------------- | --- | ------------------- | ----- | -------------------------------- | --- | --- | --- | ---------------- | --- | --- |
| difficult | tasks | that a learner | has | no prior experience | with, |                                  |     |     |     |                  |     |     |
estimatesforthenumberoflabeledexamplesperclassthat
suchaslymphnodetumorclassificationforalmostallhu-
|     |     |     |     |     |     | a logistic | regression | classifier | on  | the same | feature | space |
| --- | --- | --- | --- | --- | --- | ---------- | ---------- | ---------- | --- | -------- | ------- | ----- |
mans(andpossiblyCLIP).
requirestomatchtheperformanceofzero-shotCLIP.Since
Whilecomparingzero-shotperformancetofullysupervised zero-shotCLIPisalsoalinearclassifier,thisestimatesthe
modelscontextualizesthetask-learningcapabilitiesofCLIP, effectivedataefficiencyofzero-shottransferinthissetting.
comparingtofew-shotmethodsisamoredirectcompari-
|     |     |     |     |     |     | In order | to avoid | training | thousands | of  | linear | classifiers, |
| --- | --- | --- | --- | --- | --- | -------- | -------- | -------- | --------- | --- | ------ | ------------ |
son,sincezero-shotisitslimit. InFigure6,wevisualize we estimate the effective data efficiency based on a log-
howzero-shotCLIPcomparestofew-shotlogisticregres-
linearinterpolationoftheperformanceofa1,2,4,8,16-
siononthefeaturesofmanyimagemodelsincludingthe shot(whenpossible),andafullysupervisedlinearclassifier
bestpubliclyavailableImageNetmodels,self-supervised trainedoneachdataset. Wefindthatzero-shottransfercan
| learningmethods,andCLIPitself. |     |     |     | Whileitisintuitiveto |     |     |     |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- |

10
LearningTransferableVisualModelsFromNaturalLanguageSupervision
| FER2013 |     |     |     |     |     | 184 | 100 |     |     |     |     | STL10 |
| ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- |
| CIFAR10 |     |     |     | 81  |     |     |     |     |     |     |     |       |
CIFAR10
| Food101    |     |      | 64  |     |     |     |     |     |     |     |     | CaFltoeocdh110011OxfordPets |
| ---------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------------------- |
| OxfordPets |     |      | 48  |     |     |     | 90  |     |     |     |     |                             |
| Country211 |     | 32   |     |     |     |     |     |     |     |     |     | MNIST                       |
| ImageNet   |     | 16.0 |     |     |     |     |     |     |     |     |     | VOC2007                     |
| PCam       |     | 14.7 |     |     |     |     |     |     |     |     |     |                             |
SST2 14.4 ecnamrofreP PILC tohS-oreZ 80 SIgFt aAN nRf e1o t0rd0 Cars
|             |     |      |     |     |     |     |     |     |     |     | ImCa       | UCF101 Flowers102 |
| ----------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----------------- |
| Kinetics700 |     | 13.6 |     |     |     |     |     |     |     |     | e          |                   |
| STL10       |     | 12.7 |     |     |     |     |     |     |     |     |            |                   |
| CIFAR100    |     | 12.0 |     |     |     |     | 70  |     |     |     |            | RESISC45          |
|             |     | 9.8  |     |     |     |     |     |     |     |     | SUN397SST2 |                   |
HatefulMemes
| StanfordCars |     | 6.0 |     |     |     |     |     |     |     | HatefulMemes |     | PCAM    |
| ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------- |
| MNIST        | 4.8 |     |     |     |     |     |     |     |     | Kinetics700  |     |         |
| SUN397       | 3.9 |     |     |     |     |     | 60  |     |     |              |     | EuroSAT |
FER2013
| Caltech101     | 3.5 |     |     |     |     |     |     |     |     |     | DTD |       |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- |
| KITTI Distance | 2.9 |     |     |     |     |     |     |     |     |     |     | GTSRB |
| UCF101         | 2.9 |     |     |     |     |     | 50  |     |     |     |     |       |
Birdsnap
| Birdsnap     | 2.7     |     |     |     |     |     |     |     |     |     |     |     |
| ------------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|              | DTD 2.6 |     |     |     |     |     |     |     |     |     |     |     |
| FGVCAircraft | 2.0     |     |     |     |     |     |     |     |     |     |     |     |
40
| GTSRB       | 1.6 |     |       |     |         |             |     |     |            |             | FGVCAircraft   |          |
| ----------- | --- | --- | ----- | --- | ------- | ----------- | --- | --- | ---------- | ----------- | -------------- | -------- |
| CLEVRCounts | 1.5 |     |       |     |         |             |     |     | Country211 |             |                |          |
| RESISC45    | 1.5 |     |       |     |         |             |     |     |            |             | KITTI Distance |          |
|             |     |     |       |     |         | Mean:  20.8 | 30  |     |            |             |                |          |
| EuroSAT     | 0.9 |     |       |     |         | Median: 5.4 |     |     |            |             |                |          |
| Flowers102  | 0.9 |     |       |     |         |             |     |     |            |             |                |          |
|             |     |     |       |     |         |             |     |     |            | CLEVRCounts |                | r = 0.82 |
|             | 0   | 25  | 50 75 | 100 | 125 150 | 175 200     |     |     |            |             |                |          |
20
|     |     |     | # of labeled examples per class |     |     |     | 20  | 30 40 | 50  | 60 70 | 80  | 90 100 |
| --- | --- | --- | ------------------------------- | --- | --- | --- | --- | ----- | --- | ----- | --- | ------ |
required to match zero-shot
Linear Probe CLIP Performance
| Figure7.The | data        | efficiency |        | of zero-shot | transfer | varies    |                   |     |             |                     |            |             |
| ----------- | ----------- | ---------- | ------ | ------------ | -------- | --------- | ----------------- | --- | ----------- | ------------------- | ---------- | ----------- |
|             |             |            |        |              |          |           | Figure8.Zero-shot |     | performance | is                  | correlated | with linear |
| widely.     | Calculating | the        | number | of labeled   | examples | per class |                   |     |             |                     |            |             |
|             |             |            |        |              |          |           | probe performance |     | but still   | mostly sub-optimal. |            | Comparing   |
alinearclassifieronthesameCLIPfeaturespacerequirestomatch
zero-shotandlinearprobeperformanceacrossdatasetsshowsa
theperformanceofthezero-shotclassifiercontextualizestheef- strongcorrelationwithzero-shotperformancemostlyshifted10to
| fectivenessofzero-shottransfer. |     |     |     | Valuesareestimatedbasedon |     |     |                |                                         |     |     |     |     |
| ------------------------------- | --- | --- | --- | ------------------------- | --- | --- | -------------- | --------------------------------------- | --- | --- | --- | --- |
|                                 |     |     |     |                           |     |     | 25pointslower. | Ononly5datasetsdoeszero-shotperformance |     |     |     |     |
log-linearinterpolationof1,2,4,8,16-shotandfullysupervised
approachlinearprobeperformance(≤3pointdifference).
results. Performancevarieswidelyfromstillunderperforminga
one-shotclassifierontwodatasetstomatchinganestimated184
mance,suggestingthatCLIPisrelativelyconsistentatcon-
labeledexamplesperclass.
nectingunderlyingrepresentationandtasklearningtozero-
|     |     |     |     |     |     |     | shot transfer. | However, | zero-shot |     | CLIP only | approaches |
| --- | --- | --- | --- | --- | --- | --- | -------------- | -------- | --------- | --- | --------- | ---------- |
havewidelyvaryingefficiencyperdatasetfromlessthan1
|                                       |     |     |     |                        |     |           | fully supervised | performance |             | on 5 | datasets:   | STL10, CI- |
| ------------------------------------- | --- | --- | --- | ---------------------- | --- | --------- | ---------------- | ----------- | ----------- | ---- | ----------- | ---------- |
| labeledexampleperclassto184.          |     |     |     | Twodatasets,Flowers102 |     |           |                  |             |             |      |             |            |
|                                       |     |     |     |                        |     |           | FAR10, Food101,  |             | OxfordPets, | and  | Caltech101. | On all 5   |
| andEuroSATunderperformone-shotmodels. |     |     |     |                        |     | Halfofthe |                  |             |             |      |             |            |
datasets,bothzero-shotaccuracyandfullysupervisedaccu-
datasetsrequirelessthan5examplesperclasswithamedian
|     |     |     |     |     |     |     | racyareover90%. |     | ThissuggeststhatCLIPmaybemore |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ----------------------------- | --- | --- | --- |
of5.4. However,themeanestimateddataefficiencyis20.8
effectiveatzero-shottransferfortaskswhereitsunderly-
| examples | per | class. | This is | due to | the 20% | of datasets |                     |     |          |               |     |                |
| -------- | --- | ------ | ------- | ------ | ------- | ----------- | ------------------- | --- | -------- | ------------- | --- | -------------- |
|          |     |        |         |        |         |             | ing representations |     | are also | high quality. |     | The slope of a |
wheresupervisedclassifiersrequiremanylabeledexamples
linearregressionmodelpredictingzero-shotperformance
| per class | in order | to  | match | performance. | On  | ImageNet, |     |     |     |     |     |     |
| --------- | -------- | --- | ----- | ------------ | --- | --------- | --- | --- | --- | --- | --- | --- |
asafunctionoffullysupervisedperformanceestimatesthat
zero-shotCLIPmatchestheperformanceofa16-shotlinear
forevery1%improvementinfullysupervisedperformance,
classifiertrainedonthesamefeaturespace.
|     |     |     |     |     |     |     | zero-shotperformanceimprovesby1.28%. |     |     |     |     | However,the |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- | ----------- |
Ifweassumethatevaluationdatasetsarelargeenoughthat 95th-percentileconfidenceintervalsstillincludevaluesof
theparametersoflinearclassifierstrainedonthemarewell lessthan1(0.93-1.79).
estimated,then,becauseCLIP’szero-shotclassifierisalso
Overthepastfewyears,empiricalstudiesofdeeplearning
alinearclassifier,theperformanceofthefullysupervised
systemshavedocumentedthatperformanceispredictableas
classifiersroughlysetsanupperboundforwhatzero-shot
afunctionofimportantquantitiessuchastrainingcompute
| transfercanachieve. |     | InFigure8wecompareCLIP’szero- |     |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
anddatasetsize(Hestnessetal.,2017;Kaplanetal.,2020).
| shot performance |     | with | fully | supervised | linear | classifiers |     |     |     |     |     |     |
| ---------------- | --- | ---- | ----- | ---------- | ------ | ----------- | --- | --- | --- | --- | --- | --- |
TheGPTfamilyofmodelshassofardemonstratedconsis-
| acrossdatasets. |     | Thedashed,y |     | =xlinerepresentsan“op- |     |     |     |     |     |     |     |     |
| --------------- | --- | ----------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
tentimprovementsinzero-shotperformanceacrossa1000x
timal”zero-shotclassifierthatmatchestheperformanceof
|                               |     |     |     |                         |     |     | increaseintrainingcompute. |     |     | InFigure9,wecheckwhether |     |     |
| ----------------------------- | --- | --- | --- | ----------------------- | --- | --- | -------------------------- | --- | --- | ------------------------ | --- | --- |
| itsfullysupervisedequivalent. |     |     |     | Formostdatasets,theper- |     |     |                            |     |     |                          |     |     |
thezero-shotperformanceofCLIPfollowsasimilarscaling
formanceofzero-shotclassifiersstillunderperformfullysu-
pattern. Weplottheaverageerrorrateofthe5ResNetCLIP
pervisedclassifiersby10%to25%,suggestingthatthereis
modelsacross39evaluationson36differentdatasetsand
stillplentyofheadroomforimprovingCLIP’stask-learning
findthatasimilarlog-loglinearscalingtrendholdsforCLIP
andzero-shottransfercapabilities.
|     |     |     |     |     |     |     | acrossa44xincreaseinmodelcompute. |     |     |     | Whiletheoverall |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --------------- | --- |
There is a positive correlation of 0.82 (p-value < 10−6) trendissmooth,wefoundthatperformanceonindividual
betweenzero-shotperformanceandfullysupervisedperfor- evaluations can be much noisier. We are unsure whether

11
LearningTransferableVisualModelsFromNaturalLanguageSupervision
classifiershastheaddedbenefitofbeingverysimilartothe
RN50
| 45  |     |     |     |     |     |     | approach | used | for its zero-shot | classifiers | which | enables |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---- | ----------------- | ----------- | ----- | ------- |
RN101
|     |     |        |     |     |     |     | extensivecomparisonsandanalysisinSection3.1.    |     |     |     |     | Finally, |
| --- | --- | ------ | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | -------- |
|     |     | RN50x4 |     |     |     |     | weaimtocompareCLIPtoacomprehensivesetofexisting |     |     |     |     |          |
40
|     |     |     |     |     |     |     | modelsacrossmanytasks. |     | Studying66differentmodelson |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --------------------------- | --- | --- | --- |
)%( rorrE
27differentdatasetsrequirestuning1782differentevalua-
RN50x16
| 35  |     |     |     |     |     |     | tions. Fine-tuningopensupamuchlargerdesignandhyper- |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- |
parameterspace,whichmakesitdifficulttofairlyevaluate
andcomputationallyexpensivetocompareadiversesetof
techniquesasdiscussedinotherlargescaleempiricalstudies
30
RN50x64
(Lucicetal.,2018;Choietal.,2019).Bycomparison,linear
classifiersrequireminimalhyper-parametertuningandhave
| 6.1 | 9.9 | 21.5 |     | 75.3 |     | 265.9 |     |     |     |     |     |     |
| --- | --- | ---- | --- | ---- | --- | ----- | --- | --- | --- | --- | --- | --- |
Model GFLOPs standardized implementations and evaluation procedures.
PleaseseeAppendixAforfurtherdetailsonevaluation.
| Figure9.Zero-shot |     | CLIP | performance | scales | smoothly | as a |     |     |     |     |     |     |
| ----------------- | --- | ---- | ----------- | ------ | -------- | ---- | --- | --- | --- | --- | --- | --- |
function of model compute. Across 39 evals on 36 different Figure10summarizesourfindings. Tominimizeselection
datasets,averagezero-shoterroriswellmodeledbyalog-loglin- effectsthatcouldraiseconcernsofconfirmationorreporting
eartrendacrossa44xrangeofcomputespanning5differentCLIP bias,wefirststudyperformanceonthe12datasetevaluation
models.Lightlyshadedlinesareperformanceonindividualevals,
|     |     |     |     |     |     |     | suitefromKornblithetal.(2019). |     |     | WhilesmallCLIPmod- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | ------------------ | --- | --- |
showingthatperformanceismuchmorevarieddespitethesmooth elssuchasaResNet-50andResNet-101outperformother
overalltrend. ResNetstrainedonImageNet-1K(BiT-Sandtheoriginals),
theyunderperformResNetstrainedonImageNet-21K(BiT-
M).ThesesmallCLIPmodelsalsounderperformmodels
thisiscausedbyhighvariancebetweenindividualtraining
runsonsub-tasks(asdocumentedinD’Amouretal.(2020)) in the EfficientNet family with similar compute require-
ments. However,modelstrainedwithCLIPscaleverywell
maskingasteadilyimprovingtrendorwhetherperformance
andthelargestmodelwetrained(ResNet-50x64)slightly
isactuallynon-monotonicasafunctionofcomputeonsome
| tasks. |     |     |     |     |     |     | outperformsthebestperformingexistingmodel(aNoisy |     |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- |
StudentEfficientNet-L2)onbothoverallscoreandcompute
3.2.RepresentationLearning efficiency. WealsofindthatCLIPvisiontransformersare
about3xmorecomputeefficientthanCLIPResNets,which
Whilewehaveextensivelyanalyzedthetask-learningca- allows us to reach higher overall performance within our
pabilitiesofCLIPthroughzero-shottransferintheprevi-
|     |     |     |     |     |     |     | compute | budget. | These results | qualitatively | replicate | the |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------- | ------------- | ------------- | --------- | --- |
oussection,itismorecommontostudytherepresentation findings of Dosovitskiy et al. (2020) which reported that
| learningcapabilitiesofamodel. |     |     |     | Thereexistmanywaysto |     |     |                     |     |          |         |           |           |
| ----------------------------- | --- | --- | --- | -------------------- | --- | --- | ------------------- | --- | -------- | ------- | --------- | --------- |
|                               |     |     |     |                      |     |     | vision transformers |     | are more | compute | efficient | than con- |
evaluatethequalityofrepresentationsaswellasdisagree- vnetswhentrainedonsufficientlylargedatasets. Ourbest
mentsoverwhatpropertiesan“ideal”representationshould overallmodelisaViT-L/14thatisfine-tunedatahigherres-
| have(Locatelloetal.,2020). |     |     |     | Fittingalinearclassifieron |     |     |     |     |     |     |     |     |
| -------------------------- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
olutionof336pixelsonourdatasetfor1additionalepoch.
arepresentationextractedfromthemodelandmeasuring Thismodeloutperformsthebestexistingmodelacrossthis
itsperformanceonvariousdatasetsisacommonapproach.
evaluationsuitebyanaverageof2.6%.
Analternativeismeasuringtheperformanceofend-to-end
AsFigure21qualitativelyshows,CLIPmodelslearnawider
| fine-tuning | of the | model. | This | increases | flexibility, | and |     |     |     |     |     |     |
| ----------- | ------ | ------ | ---- | --------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
setoftasksthanhaspreviouslybeendemonstratedinasin-
priorworkhasconvincinglydemonstratedthatfine-tuning
glecomputervisionmodeltrainedend-to-endfromrandom
| outperforms     | linear | classification |     | on most    | image   | classifi-   |                 |                                           |     |     |     |     |
| --------------- | ------ | -------------- | --- | ---------- | ------- | ----------- | --------------- | ----------------------------------------- | --- | --- | --- | --- |
|                 |        |                |     |            |         |             | initialization. | Thesetasksincludegeo-localization,optical |     |     |     |     |
| cation datasets |        | (Kornblith     | et  | al., 2019; | Zhai et | al., 2019). |                 |                                           |     |     |     |     |
characterrecognition,facialemotionrecognition,andaction
| While the | high | performance |     | of fine-tuning | motivates | its |              |                                         |     |     |     |     |
| --------- | ---- | ----------- | --- | -------------- | --------- | --- | ------------ | --------------------------------------- | --- | --- | --- | --- |
|           |      |             |     |                |           |     | recognition. | Noneofthesetasksaremeasuredintheevalua- |     |     |     |     |
studyforpracticalreasons,westilloptforlinearclassifier
|                                   |     |     |     |                  |     |     | tionsuiteofKornblithetal.(2019). |     |                |              | Thiscouldbeargued |              |
| --------------------------------- | --- | --- | --- | ---------------- | --- | --- | -------------------------------- | --- | -------------- | ------------ | ----------------- | ------------ |
| basedevaluationforseveralreasons. |     |     |     | Ourworkisfocused |     |     |                                  |     |                |              |                   |              |
|                                   |     |     |     |                  |     |     | to be a form                     | of  | selection bias | in Kornblith | et                | al. (2019)’s |
ondevelopingahigh-performingtaskanddataset-agnostic
|                       |     |     |                                 |     |     |     | studytowardstasksthatoverlapwithImageNet. |     |     |     |     | Toaddress |
| --------------------- | --- | --- | ------------------------------- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | --- | --------- |
| pre-trainingapproach. |     |     | Fine-tuning,becauseitadaptsrep- |     |     |     |                                           |     |     |     |     |           |
resentations to each dataset during the fine-tuning phase, this,wealsomeasureperformanceonabroader27dataset
|     |     |     |     |     |     |     | evaluationsuite. |     | Thisevaluationsuite,detailedinAppendix |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | -------------------------------------- | --- | --- | --- |
cancompensateforandpotentiallymaskfailurestolearn
Aincludesdatasetsrepresentingtheaforementionedtasks,
generalandrobustrepresentationsduringthepre-training
GermanTrafficSignsRecognitionBenchmark(Stallkamp
phase. Linearclassifiers,becauseoftheirlimitedflexibility,
etal.,2011),aswellasseveralotherdatasetsadaptedfrom
insteadhighlightthesefailuresandprovideclearfeedback
duringdevelopment. ForCLIP,trainingsupervisedlinear VTAB(Zhaietal.,2019).

12
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Linear probe average over Kornblith et al.'s 12 datasets Linear probe average over all 27 datasets
90
85
85
| )%( erocS egarevA |     |     |     |     | )%( erocS egarevA 80 |     |     |     |     |
| ----------------- | --- | --- | --- | --- | -------------------- | --- | --- | --- | --- |
80
75
75
70
|     | 100 | 101                       | 102 |     |                      | 100                       | 101                | 102 |     |
| --- | --- | ------------------------- | --- | --- | -------------------- | ------------------------- | ------------------ | --- | --- |
|     |     | Forward-pass GFLOPs/image |     |     |                      | Forward-pass GFLOPs/image |                    |     |     |
|     |     | CLIP-ViT                  |     |     | Instagram-pretrained |                           | ViT (ImageNet-21k) |     |     |
|     |     | CLIP-ResNet               |     |     | SimCLRv2             |                           | BiT-M              |     |     |
|     |     | EfficientNet-NoisyStudent |     |     | BYOL                 |                           | BiT-S              |     |     |
|     |     | EfficientNet              |     |     | MoCo                 |                           | ResNet             |     |     |
Figure10.Linear probe performance of CLIP models in comparison with state-of-the-art computer vision models, including
EfficientNet(Tan&Le,2019;Xieetal.,2020),MoCo(Chenetal.,2020d),Instagram-pretrainedResNeXtmodels(Mahajanetal.,2018;
Touvronetal.,2019),BiT(Kolesnikovetal.,2019),ViT(Dosovitskiyetal.,2020),SimCLRv2(Chenetal.,2020c),BYOL(Grilletal.,
2020),andtheoriginalResNetmodels(Heetal.,2016b).(Left)Scoresareaveragedover12datasetsstudiedbyKornblithetal.(2019).
(Right)Scoresareaveragedover27datasetsthatcontainawidervarietyofdistributions. Dottedlinesindicatemodelsfine-tunedor
evaluatedonimagesatahigher-resolutionthanpre-training.SeeTable10forindividualscoresandFigure20forplotsforeachdataset.
Onthisbroaderevaluationsuite,thebenefitsofCLIPare andHatefulMemes),geo-localizationandscenerecognition
moreclear. AllCLIPmodels,regardlessofscale,outper- (Country211,SUN397),andactivityrecognitioninvideos
formallevaluatedsystemsintermsofcomputeefficiency. (Kinetics700 and UCF101). In addition CLIP also does
Theimprovementinaveragescoreofthebestmodelover muchbetteronfine-grainedcarandtrafficsignrecognition
previoussystemsincreasesfrom2.6%to5%. Wealsofind (Stanford Cars and GTSRB). This may reflect a problem
that self-supervised systems do noticeably better on our withoverlynarrowsupervisioninImageNet. Aresultsuch
broader evaluation suite. For instance, while SimCLRv2 asthe14.7%improvementonGTSRBcouldbeindicative
still underperforms BiT-M on average on the 12 datasets ofanissuewithImageNet-1K,whichhasonlyasinglela-
ofKornblithetal.(2019),SimCLRv2outperformsBiT-M bel for all traffic and street signs. This could encourage
onour27datasetevaluationsuite. Thesefindingssuggest a supervised representation to collapse intra-class details
continuingtoexpandtaskdiversityandcoverageinorder andhurtaccuracyonafine-graineddownstreamtask. As
tobetterunderstandthe“general”performanceofsystems. mentioned, CLIP still underperforms the EfficientNet on
Wesuspectadditionalevaluationeffortsalongthelinesof several datasets. Unsurprisingly, the dataset that the Effi-
VTABtobevaluable. cientNet does best relative to CLIP on is the one it was
|             |                  |          |           |           | trainedon: | ImageNet. | TheEffcientNetalsoslightlyoutper- |     |     |
| ----------- | ---------------- | -------- | --------- | --------- | ---------- | --------- | --------------------------------- | --- | --- |
| In addition | to the aggregate | analysis | above, we | visualize |            |           |                                   |     |     |
formsCLIPonlow-resolutiondatasetssuchasCIFAR10
per-datasetdifferencesintheperformanceofthebestCLIP
|                          |                 |                         |                  |        | andCIFAR100. | Wesuspectthisisatleastpartlyduetothe |                   |               |           |
| ------------------------ | --------------- | ----------------------- | ---------------- | ------ | ------------ | ------------------------------------ | ----------------- | ------------- | --------- |
| model and                | the best        | model in our            | evaluation suite | across |              |                                      |                   |               |           |
|                          |                 |                         |                  |        | lack of      | scale-based                          | data augmentation | in CLIP.      | The Effi- |
| all27datasetsinFigure11. |                 | CLIPoutperformstheNoisy |                  |        |              |                                      |                   |               |           |
|                          |                 |                         |                  |        | cientNet     | also does slightly                   | better on         | PatchCamelyon | and       |
| Student                  | EfficientNet-L2 | on 21 of                | the 27 datasets. | CLIP   |              |                                      |                   |               |           |
CLEVRCounts,datasetswhereoverallperformanceisstill
| improves | the most | on tasks which | require OCR | (SST2 |     |     |     |     |     |
| -------- | -------- | -------------- | ----------- | ----- | --- | --- | --- | --- | --- |

13
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     | SST2 |     |     | +23.6 | combinationofthetwo?CLIPmodels,whicharetrainedvia |     |     |     |     |
| --- | ---- | --- | --- | ----- | ------------------------------------------------- | --- | --- | --- | --- |
Country211 +22.7 naturallanguagesupervisiononaverylargedatasetandare
| HatefulMemes |     |     |     | +18.8 |     |     |     |     |     |
| ------------ | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
StanfordCars +15.9 capableofhighzero-shotperformance,areanopportunity
|     | GTSRB       |      | +14.7 |     | toinvestigatethisquestionfromadifferentangle. |        |             |               |            |
| --- | ----------- | ---- | ----- | --- | --------------------------------------------- | ------ | ----------- | ------------- | ---------- |
|     | SUN397      |      | +6.5  |     |                                               |        |             |               |            |
|     | Kinetics700 | +6.2 |       |     |                                               |        |             |               |            |
|     |             |      |       |     | Taori et al.                                  | (2020) | is a recent | comprehensive | study mov- |
|     | RESISC45    | +5.1 |       |     |                                               |        |             |               |            |
FER2013 +4.5 ingtowardsquantifyingandunderstandingthesebehaviors
Food101 +3.9 for ImageNet models. Taori et al. (2020) study how the
| FGVCAircraft |        | +3.2 |     |     |                                                |     |     |     |     |
| ------------ | ------ | ---- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- |
|              | UCF101 | +3.1 |     |     | performanceofImageNetmodelschangewhenevaluated |     |     |     |     |
KITTI Distance +2.3 onnaturaldistributionshifts. Theymeasureperformance
|     | Birdsnap | +1.4 |     |     |     |     |     |     |     |
| --- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- |
Flowers102 +1.4 onasetof7distributionshifts: ImageNetV2(Rechtetal.,
Caltech101 +1.3 2019),ImageNetSketch(Wangetal.,2019),Youtube-BB
EuroSAT +0.9 andImageNet-Vid(Shankaretal.,2019),ObjectNet(Barbu
MNIST +0.6
DTD +0.5 etal.,2019),ImageNetAdversarial(Hendrycksetal.,2019),
VOC2007 +0.5 andImageNetRendition(Hendrycksetal.,2020a). They
STL10 +0.0
-0.5 OxfordPets distinguishthesedatasets,whichallconsistofnovelimages
-0.8 CIFAR10 collected from a variety of sources, from synthetic distri-
-1.2 PatchCamelyon
butionshiftssuchasImageNet-C(Hendrycks&Dietterich,
-1.7 CIFAR100
-2.4 CLEVRCounts 2019),StylizedImageNet(Geirhosetal.,2018),oradver-
|     | -3.0 | ImageNet |     |     |     |     |     |     |     |
| --- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- |
sarialattacks(Goodfellowetal.,2014)whicharecreatedby
10 5 0 5 10 15 20 25 perturbingexistingimagesinvariousways. Theypropose
 Score (%)
thisdistinctionbecauseinpartbecausetheyfindthatwhile
Logistic Regression on CLIP vs. EfficientNet L2 NS
severaltechniqueshavebeendemonstratedtoimproveper-
Figure11.CLIP’sfeaturesoutperformthefeaturesofthebest formanceonsyntheticdistributionshifts,theyoftenfailto
ImageNetmodelonawidevarietyofdatasets. Fittingalinear yieldconsistentimprovementsonnaturaldistributions.3
classifieronCLIP’sfeaturesoutperformsusingtheNoisyStudent
Acrossthesecollecteddatasets,theaccuracyofImageNet
EfficientNet-L2on21outof27datasets.
|     |     |     |     |     | models drop         | well | below                            | the expectation | set by the Ima- |
| --- | --- | --- | --- | --- | ------------------- | ---- | -------------------------------- | --------------- | --------------- |
|     |     |     |     |     | geNetvalidationset. |      | Forthefollowingsummarydiscussion |                 |                 |
lowforbothapproaches.
wereportaverageaccuracyacrossall7naturaldistribution
shiftdatasetsandaverageaccuracyacrossthecorrespond-
3.3.RobustnesstoNaturalDistributionShift
|     |     |     |     |     | ing class | subsets | of ImageNet | unless | otherwise specified. |
| --- | --- | --- | --- | --- | --------- | ------- | ----------- | ------ | -------------------- |
In2015,itwasannouncedthatadeeplearningmodelex- Additionally, for Youtube-BB and ImageNet-Vid, which
ceededhumanperformanceontheImageNettestset(He havetwodifferentevaluationsettings,weusetheaverage
| et al., | 2015). | However, research | in the subsequent | years |     |     |     |     |     |
| ------- | ------ | ----------------- | ----------------- | ----- | --- | --- | --- | --- | --- |
ofpm-0andpm-10accuracy.
hasrepeatedlyfoundthatthesemodelsstillmakemanysim-
AResNet-101makes5timesasmanymistakeswheneval-
plemistakes(Dodge&Karam,2017;Geirhosetal.,2018;
uatedonthesenaturaldistributionshiftscomparedtothe
Alcornetal.,2019),andnewbenchmarkstestingthesesys-
ImageNetvalidationset.Encouraginglyhowever,Taorietal.
temshasoftenfoundtheirperformancetobemuchlower
than both their ImageNet accuracy and human accuracy (2020)findthataccuracyunderdistributionshiftincreases
predictablywithImageNetaccuracyandiswellmodeled
| (Rechtetal.,2019;Barbuetal.,2019). |           |                                       | Whatexplainsthis |          |               |          |                      |            |                 |
| ---------------------------------- | --------- | ------------------------------------- | ---------------- | -------- | ------------- | -------- | -------------------- | ---------- | --------------- |
|                                    |           |                                       |                  |          | as a linear   | function | of logit-transformed |            | accuracy. Taori |
| discrepancy?                       |           | Variousideashavebeensuggestedandstud- |                  |          |               |          |                      |            |                 |
|                                    |           |                                       |                  |          | et al. (2020) | use      | this finding         | to propose | that robustness |
| ied                                | (Ilyas et | al., 2019; Geirhos                    | et al., 2020).   | A common |               |          |                      |            |                 |
analysisshoulddistinguishbetweeneffectiveandrelative
themeofproposedexplanationsisthatdeeplearningmodels
|     |     |     |     |     | robustness. | Effective | robustness | measures | improvements |
| --- | --- | --- | --- | --- | ----------- | --------- | ---------- | -------- | ------------ |
areexceedinglyadeptatfindingcorrelationsandpatterns
inaccuracyunderdistributionshiftabovewhatispredicted
whichholdacrosstheirtrainingdatasetandthusimprove
bythedocumentedrelationshipbetweenin-distributionand
| in-distributionperformance. |     |     | Howevermanyofthesecorre- |     |                              |     |     |                            |     |
| --------------------------- | --- | --- | ------------------------ | --- | ---------------------------- | --- | --- | -------------------------- | --- |
|                             |     |     |                          |     | out-of-distributionaccuracy. |     |     | Relativerobustnesscaptures |     |
lationsandpatternsareactuallyspuriousanddonotholdfor
|     |     |     |     |     | anyimprovementinout-of-distributionaccuracy. |     |     |     | Taorietal. |
| --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | --- | ---------- |
otherdistributionsandresultinlargedropsinperformance
(2020)arguethatrobustnesstechniquesshouldaimtoim-
onotherdatasets.
provebotheffectiverobustnessandrelativerobustness.
| We  | caution that, | to date, most | of these studies | limit their |     |     |     |     |     |
| --- | ------------- | ------------- | ---------------- | ----------- | --- | --- | --- | --- | --- |
AlmostallmodelsstudiedinTaorietal.(2020)aretrained
| evaluationtomodelstrainedonImageNet. |     |     |     | Recallingthe |     |     |     |     |     |
| ------------------------------------ | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
topicofdiscussion,itmaybeamistaketogeneralizetoo 3WereferreaderstoHendrycksetal.(2020a)foradditional
far from these initial findings. To what degree are these experimentsanddiscussiononthisclaim.
| failures | attributable | to deep | learning, ImageNet, | or some |     |     |     |     |     |
| -------- | ------------ | ------- | ------------------- | ------- | --- | --- | --- | --- | --- |

14
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Linear probe average over Kornblith et al.'s 12 datasets Linear probe average over 26 datasets
| 90                 |                           |       | 90                 |                    |       |
| ------------------ | ------------------------- | ----- | ------------------ | ------------------ | ----- |
| 85                 |                           |       | 85                 |                    |       |
| )%( erocS refsnarT |                           |       | )%( erocS refsnarT |                    |       |
| 80                 |                           |       | 80                 |                    |       |
| 75                 |                           |       | 75                 |                    |       |
| 70                 |                           |       | 70                 |                    |       |
| 65                 |                           |       | 65                 |                    |       |
| 65 70              | 75 80                     | 85 90 | 65 70              | 75 80              | 85 90 |
|                    | ImageNet Score (%)        |       |                    | ImageNet Score (%) |       |
|                    | CLIP-ViT                  |       | Instagram          | ViT (ImageNet-21k) |       |
|                    | CLIP-ResNet               |       | SimCLRv2           | BiT-M              |       |
|                    | EfficientNet-NoisyStudent |       | BYOL               | BiT-S              |       |
|                    | EfficientNet              |       | MoCo               | ResNet             |       |
Figure12.CLIP’sfeaturesaremorerobusttotaskshiftwhencomparedtomodelspre-trainedonImageNet. Forbothdataset
splits, thetransferscoresoflinearprobestrainedontherepresentationsofCLIPmodelsarehigherthanothermodelswithsimilar
ImageNetperformance.ThissuggeststhattherepresentationsofmodelstrainedonImageNetaresomewhatoverfittotheirtask.
or fine-tuned on the ImageNet dataset. Returning to the in much more robust models regardless of whether they
discussion in the introduction to this section - is training are zero-shot or fine-tuned. As an initial experiment to
oradaptingtotheImageNetdatasetdistributionthecause potentially begin narrowing this down, we also measure
of the observed robustness gap? Intuitively, a zero-shot howtheperformanceofCLIPmodelschangeafteradapting
modelshouldnotbeabletoexploitspuriouscorrelations totheImageNetdistributionviaaL2regularizedlogistic
orpatternsthatholdonlyonaspecificdistribution,sinceit regressionclassifierfittoCLIPfeaturesontheImageNet
isnottrainedonthatdistribution. 4 Thusitisreasonable trainingset. Wevisualizehowperformancechangesfrom
toexpectzero-shotmodelstohavemuchhighereffective the zero-shot classifier in Figure 14. Although adapting
robustness. InFigure13,wecomparetheperformanceof CLIPtotheImageNetdistributionincreasesitsImageNet
zero-shotCLIPwithexistingImageNetmodelsonnatural accuracyby9.2%to85.4%overall,andtiestheaccuracy
distribution shifts. All zero-shot CLIP models improve of the 2018 SOTA from Mahajan et al. (2018), average
effectiverobustnessbyalargeamountandreducethesize accuracyunderdistributionshiftslightlydecreases.
ofthegapbetweenImageNetaccuracyandaccuracyunder
Itissurprisingtoseea9.2%increaseinaccuracy,whichcor-
distributionshiftbyupto75%.
respondstoroughly3yearsofimprovementinSOTA,fail
Whiletheseresultsshowthatzero-shotmodelscanbemuch totranslateintoanyimprovementinaverageperformance
morerobust,theydonotnecessarilymeanthatsupervised underdistributionshift. Wealsobreakdownthedifferences
learningonImageNetcausesarobustnessgap. Otherdetails betweenzero-shotaccuracyandlinearclassifieraccuracy
ofCLIP,suchasitslargeanddiversepre-trainingdataset perdatasetinFigure14andfindperformancestillincreases
or use of natural language supervision could also result significantly on one dataset, ImageNetV2. ImageNetV2
closelyfollowedthecreationprocessoftheoriginalIma-
4Wecautionthatazero-shotmodelcanstillexploitspurious
geNetdatasetwhichsuggeststhatgainsinaccuracyfrom
correlationsthataresharedbetweenthepre-trainingandevaluation
supervisedadaptationarecloselyconcentratedaroundthe
distributions.
|     |     |     | ImageNetdistribution. | Performancedecreasesby4.7%on |     |
| --- | --- | --- | --------------------- | ---------------------------- | --- |

15
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     |     |     |     |     |     |                  |     |     | ImageNet  | Zero-Shot |         |
| --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --------- | --------- | ------- |
| 100 |     |     |     |     |     | Dataset Examples |     |     | ResNet101 | CLIP      | Δ Score |
Ideal robust model (y = x)
| )% ,1-pot( stesatad tfihs noitubirtsid larutan 7 no egarevA 95 | Zero-Shot CLIP                  |     |     |          |     |     |     |     |      |      |     |
| -------------------------------------------------------------- | ------------------------------- | --- | --- | -------- | --- | --- | --- | --- | ---- | ---- | --- |
|                                                                | Standard ImageNet training      |     |     | ImageNet |     |     |     |     | 76.2 | 76.2 | 0%  |
| 90                                                             | Exisiting robustness techniques |     |     |          |     |     |     |     |      |      |     |
85
80
|     |     |     |     | ImageNetV2 |     |     |     |     | 64.3 | 70.1 | +5.8% |
| --- | --- | --- | --- | ---------- | --- | --- | --- | --- | ---- | ---- | ----- |
75
70
65
|     |     |     |     | ImageNet-R |     |     |     |     | 37.7 | 88.9 +51.2% |     |
| --- | --- | --- | --- | ---------- | --- | --- | --- | --- | ---- | ----------- | --- |
60
55
| 50  |     |     |     | ObjectNet |     |     |     |     | 32.6 | 72.3 +39.7% |     |
| --- | --- | --- | --- | --------- | --- | --- | --- | --- | ---- | ----------- | --- |
45
40
ImageNet
| 35  |     |     |     |     |     |     |     |     | 25.2 | 60.2 +35.0% |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----------- | --- |
Sketch
30
25
| 20  |       |             |     | ImageNet-A |     |     |     |     | 2.7 | 77.1 +74.4% |     |
| --- | ----- | ----------- | --- | ---------- | --- | --- | --- | --- | --- | ----------- | --- |
|     | 65 70 | 75 80 85 90 | 95  | 100        |     |     |     |     |     |             |     |
Average on class subsampled ImageNet (top-1, %)
Figure13.Zero-shotCLIPismuchmorerobusttodistributionshiftthanstandardImageNetmodels.(Left)Anidealrobustmodel
(dashedline)performsequallywellontheImageNetdistributionandonothernaturalimagedistributions.Zero-shotCLIPmodelsshrink
this“robustnessgap”byupto75%.Linearfitsonlogittransformedvaluesareshownwithbootstrapestimated95%confidenceintervals.
(Right)Visualizingdistributionshiftforbananas,aclasssharedacross5ofthe7naturaldistributionshiftdatasets.Theperformanceof
thebestzero-shotCLIPmodel,ViT-L/14@336px,iscomparedwithamodelthathasthesameperformanceontheImageNetvalidation
set,ResNet-101.
ImageNet-R,3.8%onObjectNet,2.8%onImageNetSketch, poolingpredictionsacrossallsub-classesaccordingtothe
and1.9%onImageNet-A.Thechangeinaccuracyonthe ImageNetclasshierarchy. Sometimesthismappingismuch
twootherdatasets, Youtube-BBandImageNetVid, isin- lessthanperfect. ForthepersonclassinYoutube-BB,pre-
significant. dictionsaremadebypoolingovertheImageNetclassesfor
|     |     |     |     |     | a baseball |     | player, | a bridegroom, | and a scuba | diver. | With |
| --- | --- | --- | --- | --- | ---------- | --- | ------- | ------------- | ----------- | ------ | ---- |
Howisitpossibletoimproveaccuracyby9.2%ontheIm-
|     |     |     |     |     | CLIP | we  | can instead | generate | a custom | zero-shot | classi- |
| --- | --- | --- | --- | --- | ---- | --- | ----------- | -------- | -------- | --------- | ------- |
ageNetdatasetwithlittletonoincreaseinaccuracyunder
|              |        |                       |     |                  | fierforeachdatasetdirectlybasedonitsclassnames. |     |        |           |                  |           | In  |
| ------------ | ------ | --------------------- | --- | ---------------- | ----------------------------------------------- | --- | ------ | --------- | ---------------- | --------- | --- |
| distribution | shift? | Is the gain primarily |     | from “exploiting |                                                 |     |        |           |                  |           |     |
|              |        |                       |     |                  | Figure                                          | 14  | we see | that this | improves average | effective | ro- |
spuriouscorrelations”?Isthisbehavioruniquetosomecom-
bustnessby5%butisconcentratedinlargeimprovements
binationofCLIP,theImageNetdatatset,andthedistribution
|                                                   |     |     |     |            | ononlyafewdatasets.  |     |     | Curiously,accuracyonObjectNet |     |     |     |
| ------------------------------------------------- | --- | --- | --- | ---------- | -------------------- | --- | --- | ----------------------------- | --- | --- | --- |
| shiftsstudied,oramoregeneralphenomena?            |     |     |     | Doesithold |                      |     |     |                               |     |     |     |
|                                                   |     |     |     |            | alsoincreasesby2.3%. |     |     | Althoughthedatasetwasdesigned |     |     |     |
| forend-to-endfinetuningaswellaslinearclassifiers? |     |     |     | We         |                      |     |     |                               |     |     |     |
tocloselyoverlapwithImageNetclasses,usingthenames
donothaveconfidentanswerstothesequestionsatthistime.
providedforeachclassbyObjectNet’screatorsstillhelpsa
Priorworkhasalsopre-trainedmodelsondistributionsother
smallamountcomparedtousingImageNetclassnamesand
thanImageNet,butitiscommontostudyandreleasemod-
poolingpredictionswhennecessary.
| elsonlyaftertheyhavebeenfine-tunedtoImageNet. |     |     |     | Asa |     |     |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
steptowardsunderstandingwhetherpre-trainedzero-shot Whilezero-shotCLIPimproveseffectiverobustness,Figure
modelsconsistentlyhavehighereffectiverobustnessthan 14showsthatthebenefitisalmostentirelygoneinafully
fine-tunedmodels, weencouragetheauthorsofMahajan supervisedsetting. Tobetterunderstandthisdifference,we
etal.(2018),Kolesnikovetal.(2019),andDosovitskiyetal. investigatehoweffectiverobustnesschangesonthecontin-
(2020)to,ifpossible,studythesequestionsontheirmodels uumfromzero-shottofullysupervised. InFigure15we
| aswell. |     |     |     |     | visualizetheperformanceof0-shot,1-shot,2-shot,4-shot |     |     |     |     |     |     |
| ------- | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
...,128-shot,andfullysupervisedlogisticregressionclassi-
Wealsoinvestigateanotherrobustnessinterventionenabled
|     |     |     |     |     | fiersonthebestCLIPmodel’sfeatures. |     |     |     | Weseethatwhile |     |     |
| --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | -------------- | --- | --- |
byflexiblezero-shotnatural-language-basedimageclassi-
few-shotmodelsalsoshowhighereffectiverobustnessthan
fiers. Thetargetclassesacrossthe7transferdatasetsare
|                                               |     |     |     |     | existing | models, |     | this benefit | fades as in-distribution |     | per- |
| --------------------------------------------- | --- | --- | --- | --- | -------- | ------- | --- | ------------ | ------------------------ | --- | ---- |
| notalwaysperfectlyalignedwiththoseofImageNet. |     |     |     | Two |          |         |     |              |                          |     |      |
formanceincreaseswithmoretrainingdataandismostly,
datasets,Youtube-BBandImageNet-Vid,consistofsuper-
|                    |     |                                |     |     | though        | not | entirely,                            | gone for | the fully supervised |     | model. |
| ------------------ | --- | ------------------------------ | --- | --- | ------------- | --- | ------------------------------------ | -------- | -------------------- | --- | ------ |
| classesofImageNet. |     | Thispresentsaproblemwhentrying |     |     |               |     |                                      |          |                      |     |        |
|                    |     |                                |     |     | Additionally, |     | zero-shotCLIPisnotablymorerobustthan |          |                      |     |        |
tousethefixed1000-wayclassifierofanImageNetmodel
afew-shotmodelwithequivalentImageNetperformance.
| tomakepredictions. |     | Taorietal.(2020)handlethisbymax- |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

16
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Adapt to ImageNet
)% ,1-pot( stesatad tfihs noitubirtsid larutan 7 no egarevA ImageNet +9.2
| 80  |     |     |     |     | ImageNetV2 |     |     | +5.8 |     |     |
| --- | --- | --- | --- | --- | ---------- | --- | --- | ---- | --- | --- |
Adapt to class shift
|     |     |     |     |     |     | Youtube-BB | +0.6 |     |     |     |
| --- | --- | --- | --- | --- | --- | ---------- | ---- | --- | --- | --- |
-0.5 ImageNet Vid
| 75  |     |     | Adapt to ImageNet |     |     |      |                 |      |       |       |
| --- | --- | --- | ----------------- | --- | --- | ---- | --------------- | ---- | ----- | ----- |
|     |     |     |                   |     |     | -1.9 | ImageNet-A      |      |       |       |
| 70  |     |     |                   |     |     | -2.8 | ImageNet Sketch |      |       |       |
|     |     |     |                   |     |     | -3.8 | ObjectNet       |      |       |       |
| 65  |     |     |                   |     |     | -4.7 | ImageNet-R      |      |       |       |
|     |     |     |                   |     | 10  | 5    | 0               | 5 10 | 15 20 | 25 30 |
60
Change from zero-shot ImageNet classifier accuracy (%)
55
Adapt to class shift
50
|     |     |     |                            |     |                  | Youtube-BB |     |      |     | +26.9 |
| --- | --- | --- | -------------------------- | --- | ---------------- | ---------- | --- | ---- | --- | ----- |
| 45  |     |     |                            |     | ImageNet Vid     |            |     | +8.3 |     |       |
|     |     |     |                            |     |                  | ObjectNet  |     | +2.3 |     |       |
| 40  |     |     | Ideal robust model (y = x) |     |                  |            |     |      |     |       |
|     |     |     | Adaptive Zero-Shot CLIP    |     | ImageNet Sketch0 |            |     |      |     |       |
| 35  |     |     | ImageNet Zero-Shot CLIP    |     | ImageNet-R0      |            |     |      |     |       |
Logistic Regression CLIP
ImageNet-A0
Standard ImageNet training
| 30  |     |     | Robustness intervention |     | ImageNetV20 |           |     |     |     |     |
| --- | --- | --- | ----------------------- | --- | ----------- | --------- | --- | --- | --- | --- |
|     |     |     | Trained with more data  |     |             | ImageNet0 |     |     |     |     |
25
| 70  | 75 80 | 85  | 90  | 95  | 10  | 5   | 0   | 5 10 | 15 20 | 25 30 |
| --- | ----- | --- | --- | --- | --- | --- | --- | ---- | ----- | ----- |
Average on class subsampled ImageNet (top-1, %) Change from zero-shot ImageNet classifier accuracy (%)
Figure14.WhilesupervisedadaptationtoImageNetincreasesImageNetaccuracyby9.2%,itslightlyreducesaveragerobustness.
(Left)Customizingzero-shotCLIPtoeachdatasetimprovesrobustnesscomparedtousingasinglestaticzero-shotImageNetclassifier
andpoolingpredictionsacrosssimilarclassesasinTaorietal.(2020).CLIPmodelsadaptedtoImageNethavesimilareffectiverobustness
asthebestpriorImageNetmodels.(Right)Detailsofperdatasetchangesinaccuracyforthetworobustnessinterventions.Adaptingto
ImageNetincreasesaccuracyonImageNetV2noticeablybuttradesoffaccuracyonseveralotherdistributions.Datasetspecificzero-shot
classifierscanimproveaccuracybyalargeamountbutarelimitedtoonlyafewdatasetsthatincludeclasseswhichdon’tperfectlyalign
withImageNetcategories.
Acrossourexperiments,higheffectiverobustnessseemsto humansononeofourtasks. Wewantedtogetasenseof
resultfromminimizingtheamountofdistributionspecific howstronghumanzero-shotperformanceisatthesetasks,
trainingdataamodelhasaccessto,butthiscomesatacost andhowmuchhumanperformanceisimprovediftheyare
ofreducingdataset-specificperformance. shown one or two image samples. This can help us to
comparetaskdifficultyforhumansandCLIP,andidentify
| Taken together, | these | results suggest | that the | recent shift |     |     |     |     |     |     |
| --------------- | ----- | --------------- | -------- | ------------ | --- | --- | --- | --- | --- | --- |
correlationsanddifferencesbetweenthem.
towardslarge-scaletaskanddatasetagnosticpre-training
combinedwithareorientationtowardszero-shotandfew- Wehadfivedifferenthumanslookateachof3669images
shotbenchmarkingonbroadevaluationsuites(asadvocated inthetestsplitoftheOxfordIITPetsdataset(Parkhietal.,
byYogatamaetal.(2019)andLinzen(2020))promotesthe 2012) and select which of the 37 cat or dog breeds best
developmentofmorerobustsystemsandprovidesamore matched the image (or ‘I don’t know’ if they were com-
accurateassessmentofperformance. Wearecurioustosee pletelyuncertain). Inthezero-shotcasethehumanswere
if the same results hold for zero-shot models in the field given no examples of the breeds and asked to label them
of NLP such as the GPT family. While Hendrycks et al. to the best of their ability without an internet search. In
(2020b)hasreportedthatpre-trainingimprovesrelativero- theone-shotexperimentthehumansweregivenonesample
bustnessonsentimentanalysis,Milleretal.(2020)’sstudy imageofeachbreedandinthetwo-shotexperimentthey
oftherobustnessofquestionansweringmodelsundernat- weregiventwosampleimagesofeachbreed.5
| ural distribution | shift | finds, similar | to Taori et | al. (2020), |     |     |     |     |     |     |
| ----------------- | ----- | -------------- | ----------- | ----------- | --- | --- | --- | --- | --- | --- |
Onepossibleconcernwasthatthehumanworkerswerenot
littleevidenceofeffectiverobustnessimprovementstodate.
|     |     |     |     |     | sufficiently |     | motivated | in the | zero-shot task. | High human |
| --- | --- | --- | --- | --- | ------------ | --- | --------- | ------ | --------------- | ---------- |
accuracyof94%ontheSTL-10dataset(Coatesetal.,2011)
4.ComparisontoHumanPerformance
|     |     |     |     |     | 5There | is  | not a | perfect correspondence | between | the human |
| --- | --- | --- | --- | --- | ------ | --- | ----- | ---------------------- | ------- | --------- |
HowdoesCLIPcomparetohumanperformanceandhuman few-shottasksandthemodel’sfew-shotperformancesincethe
modelcannotrefertosampleimagesinthewaythatthehumans
learning? Togetabetterunderstandingofhowwellhumans
can.
performinsimilarevaluationsettingstoCLIP,weevaluated

17
LearningTransferableVisualModelsFromNaturalLanguageSupervision
75
)% ,1-pot( stesatad tfihs noitubirtsid larutan 7 no egarevA MajorityVote
|     |     |     |     | 0 shot |     |     |     |          | MajorityVote  |     | Accuracy  |           |     |
| --- | --- | --- | --- | ------ | --- | --- | --- | -------- | ------------- | --- | --------- | --------- | --- |
|     |     |     |     |        |     | all |     | Accuracy |               |     |           | Accuracy  |     |
| 70  |     |     |     |        |     | 128 |     |          | onFullDataset |     | onGuesses | onGuesses |     |
64
32
| 65  |     |     |     |        |         |     | Zero-shothuman | 53.7 |     | 57.0 |     | 69.7 63.9 |     |
| --- | --- | --- | --- | ------ | ------- | --- | -------------- | ---- | --- | ---- | --- | --------- | --- |
|     |     |     |     |        | 16 shot |     | Zero-shotCLIP  | 93.5 |     | 93.5 |     | 93.5 93.5 |     |
|     |     |     |     |        |         |     | One-shothuman  | 75.7 |     | 80.3 |     | 78.5 81.2 |     |
| 60  |     |     |     | 8 shot |         |     |                |      |     |      |     |           |     |
|     |     |     |     |        |         |     | Two-shothuman  | 75.7 |     | 85.0 |     | 79.2 86.1 |     |
4 shot
55
50 2 shot Table2.ComparisonofhumanperformanceonOxfordIITPets.
AsinParkhietal.(2012),themetricisaverageper-classclassifica-
45
tionaccuracy.Mostofthegaininperformancewhengoingfrom
1 shot
| 40  |     |     |     |     |     |     | thehumanzeroshotcasetothehumanoneshotcaseisonimages |     |     |     |     |                   |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | ----------------- | --- |
|     |     |     |     |     |     |     | thatparticipantswerehighlyuncertainon.              |     |     |     |     | “Guesses”refersto |     |
35
|     |     |     |     |     |                            |     | restricting                                            | the dataset | to where | participants |     | selected an | answer |
| --- | --- | --- | --- | --- | -------------------------- | --- | ------------------------------------------------------ | ----------- | -------- | ------------ | --- | ----------- | ------ |
|     |     |     |     |     | Ideal robust model (y = x) |     | otherthan“Idon’tknow”,the“majorityvote”istakingthemost |             |          |              |     |             |        |
30
|     |     |     |     |     | Few-Shot CLIP (best model) |     | frequent(exclusiveofties)answerperimage. |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------------------------- | --- | ---------------------------------------- | --- | --- | --- | --- | --- | --- |
Zero-Shot CLIP (best model)
| 25  |     |     |     |     | Standard ImageNet training |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Robustness intervention
Trained with more data
20 quality pre-trained model is near state-of-the-art for few
| 65  | 70 75 | 80  | 85  |     | 90  | 95  |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Average on class subsampled ImageNet (top-1, %) shotlearning(Tianetal.,2020),whichsuggeststhatthereis
agapbetweenthebestfew-shotmachinelearningmethods
Figure15.Few-shot CLIP also increases effective robustness andhumanfew-shotlearning.
comparedtoexistingImageNetmodelsbutislessrobustthan
|           |       |            |     |        |             |          | If we plot | human accuracy |     | vs CLIP’s | zero | shot accuracy |     |
| --------- | ----- | ---------- | --- | ------ | ----------- | -------- | ---------- | -------------- | --- | --------- | ---- | ------------- | --- |
| zero-shot | CLIP. | Minimizing | the | amount | of ImageNet | training |            |                |     |           |      |               |     |
(Figure16),weseethatthehardestproblemsforCLIPare
datausedforadaptionincreaseseffectiverobustnessatthecostof
decreasingrelativerobustness. 16-shotlogisticregressionCLIP alsohardforhumans.Totheextentthaterrorsareconsistent,
matcheszero-shotCLIPonImageNet,aspreviouslyreportedin our hypothesis is that this is due to at least a two factors:
noiseinthedataset(includingmislabeledimages)andoutof
Figure7,butislessrobust.
distributionimagesbeinghardforbothhumansandmodels.
5.DataOverlapAnalysis
| and 97-100% | accuracy |     | on the | subset | of attention | check |     |     |     |     |     |     |     |
| ----------- | -------- | --- | ------ | ------ | ------------ | ----- | --- | --- | --- | --- | --- | --- | --- |
imagesincreasedourtrustinthehumanworkers. Aconcernwithpre-trainingonaverylargeinternetdataset
|     |     |     |     |     |     |     | is unintentional | overlap | with | downstream |     | evals. | This is |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ------- | ---- | ---------- | --- | ------ | ------- |
Interestingly,humanswentfromaperformanceaverageof
importanttoinvestigatesince,inaworst-casescenario,a
54%to76%withjustonetrainingexampleperclass,and
completecopyofanevaluationdatasetcouldleakintothe
| the marginal | gain | from | an additional |     | training | example is |     |     |     |     |     |     |     |
| ------------ | ---- | ---- | ------------- | --- | -------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
pre-trainingdatasetandinvalidatetheevaluationasamean-
minimal. Thegaininaccuracygoingfromzerotooneshot
|           |          |           |     |             |      |           | ingfultestofgeneralization. |     |     | Oneoptiontopreventthisisto |     |     |     |
| --------- | -------- | --------- | --- | ----------- | ---- | --------- | --------------------------- | --- | --- | -------------------------- | --- | --- | --- |
| is almost | entirely | on images |     | that humans | were | uncertain |                             |     |     |                            |     |     |     |
identifyandremoveallduplicatesbeforetrainingamodel.
about. Thissuggeststhathumans“knowwhattheydon’t
Whilethisguaranteesreportingtruehold-outperformance,
know”andareabletoupdatetheirpriorsontheimagesthey
itrequiresknowingallpossibledatawhichamodelmight
| aremostuncertaininbasedonasingleexample. |            |      |      |           |          | Giventhis, |                                            |          |          |      |     |              |     |
| ---------------------------------------- | ---------- | ---- | ---- | --------- | -------- | ---------- | ------------------------------------------ | -------- | -------- | ---- | --- | ------------ | --- |
|                                          |            |      |      |           |          |            | be evaluated                               | on ahead | of time. | This | has | the downside | of  |
| it seems                                 | that while | CLIP | is a | promising | training | strategy   |                                            |          |          |      |     |              |     |
|                                          |            |      |      |           |          |            | limitingthescopeofbenchmarkingandanalysis. |          |          |      |     | Addinga      |     |
forzero-shotperformance(Figure5)anddoeswellontests
newevaluationwouldrequireanexpensivere-trainorrisk
| of natural | distribution |     | shift (Figure |     | 13), there | is a large |     |     |     |     |     |     |     |
| ---------- | ------------ | --- | ------------- | --- | ---------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
reportinganun-quantifiedbenefitduetooverlap.
differencebetweenhowhumanslearnfromafewexamples
andthefew-shotmethodsinthispaper. Instead,wedocumenthowmuchoverlapoccursandhow
|     |     |     |     |     |     |     | performancechangesduetotheseoverlaps. |     |     |     |     | Todothis,we |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | ----------- | --- |
Thissuggeststhattherearestillalgorithmicimprovements
usethefollowingprocedure:
waitingtobemadetodecreasethegapbetweenmachine
andhumansampleefficiency,asnotedbyLakeetal.(2016) 1)Foreachevaluationdataset,werunaduplicatedetector
and others. Because these few-shot evaluations of CLIP (seeAppendixC)onitsexamples.Wethenmanuallyinspect
don’tmakeeffectiveuseofpriorknowledgeandthehumans thefoundnearestneighborsandsetaperdatasetthreshold
do,wespeculatethatfindingamethodtoproperlyintegrate to keep high precision while maximizing recall. Using
priorknowledgeintofew-shotlearningisanimportantstep thisthreshold,wethencreatetwonewsubsets,Overlap,
inalgorithmicimprovementstoCLIP.Toourknowledge, which contains all examples which have a similarity to a
using a linear classifier on top of the features of a high- trainingexampleabovethethreshold,andClean,which

18
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     |     |     |     |     |     |     | ouranalysis.        | Thereisamedianoverlapof2.2%andanav- |                                |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | ----------------------------------- | ------------------------------ | --- | --- | --- |
|     |     |     |     |     |     |     | erageoverlapof3.2%. |                                     | Duetothissmallamountofoverlap, |     |     |     |
100
overallaccuracyisrarelyshiftedbymorethan0.1%with
|     |     |     |     |     |     |     | only7datasetsabovethisthreshold.                   |     |     |     | Ofthese,only2are |        |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | ---------------- | ------ |
|     | 80  |     |     |     |     |     | statisticallysignificantafterBonferronicorrection. |     |     |     |                  | Themax |
)%( ycaruccA
detectedimprovementisonly0.6%onBirdsnapwhichhas
|     |     |     |     |     |     |     | thesecondlargestoverlapat12.1%. |     |     |     | Thelargestoverlapis |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | ------------------- | --- |
60
|     |     |     |     |     |     |     | forCountry211at21.5%. |     |     | Thisisduetoitbeingconstructed |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --- | ----------------------------- | --- | --- |
outofYFCC100M,whichourpre-trainingdatasetcontains
40
|     |     |     |     |     |     |     | afilteredsubsetof. |     | Despitethislargeoverlapthereisonly |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ---------------------------------- | --- | --- | --- |
Zero-Shot CLIP a0.2%increaseinaccuracyonCountry211. Thismaybe
One-Shot Human
|     | 20  |     |     |     |     |     | becausethetrainingtextaccompanyinganexampleisoften |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- |
Zero-Shot Human
notrelatedtothespecifictaskadownstreamevalmeasures.
|     | gup | xnyhps deriahtrohs_namreg uni_abihs elgaeb seeneryp_taerg rettes_hsilgne | deyomas dranreb_tnias nainaremop dnaldnuofwen reirret_netaehw | reirret_hsittocs reirret_erihskroy esemais rehcsnip_erutainim esenavah | dnohseek yabmob nooc_eniam auhauhihc dnuoh_tessab nihc_esenapaj eulb_naissur | godllub_nacirema naisrep lagneb regrebnoel nainissyba rexob riahtrohs_hsitirb reirret_llub_erihsdroffats reirret_llub_tip_nacirema uam_naitpyge namrib leinaps_rekcoc_hsilgne llodgar |     |     |     |     |     |     |
| --- | --- | ------------------------------------------------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Country211measuresgeo-localizationability,butinspect-
ingthetrainingtextfortheseduplicatesshowedtheyoften
donotmentionthelocationoftheimage.
Weareawareoftwopotentialconcernswithouranalysis.
|     |     |     |     |     |     |     | First our     | detector | is not    | perfect. While | it achieves | near       |
| --- | --- | --- | --- | --- | --- | --- | ------------- | -------- | --------- | -------------- | ----------- | ---------- |
|     |     |     |     |     |     |     | 100% accuracy | on       | its proxy | training       | task and    | manual in- |
Figure16.ThehardestproblemsforCLIPalsotendtobethehard- spection+thresholdtuningresultsinveryhighprecision
estproblemsforhumans.Herewerankimagecategoriesbydiffi-
withgoodrecallamongthefoundnearest-neighbors,wecan
cultyforCLIPasmeasuredasprobabilityofthecorrectlabel. nottractablycheckitsrecallacross400millionexamples.
Anotherpotentialconfounderofouranalysisisthattheun-
derlyingdatadistributionmayshiftbetweentheOverlap
|          |     |              |     |          |       |                    | andCleansubsets.                             |     | Forexample, | onKinetics-700many |     |         |
| -------- | --- | ------------ | --- | -------- | ----- | ------------------ | -------------------------------------------- | --- | ----------- | ------------------ | --- | ------- |
|          |     |              |     |          |       |                    | “overlaps”areinfactallblacktransitionframes. |     |             |                    |     | Thisex- |
| contains |     | all examples |     | that are | below | this threshold. We |                                              |     |             |                    |     |         |
plainswhyKinetics-700hasanapparent20%accuracydrop
| denotetheunalteredfulldatasetAllforreference. |     |     |     |     |     | From |     |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
onOverlap.
thiswefirstrecordthedegreeofdatacontaminationasthe Wesuspectmoresubtledistributionshifts
|     |     |     |     |     |     |     | likelyexist. | OnepossibilitywenoticedonCIFAR-100is |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------------------------------ | --- | --- | --- | --- |
ratioofthenumberofexamplesinOverlaptothesizeof
|     |     |     |     |     |     |     | that, due | to the | very low | resolution | of its images, | many |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | -------- | ---------- | -------------- | ---- |
All.
duplicateswerefalsepositivesofsmallobjectssuchasbirds
2) We then compute the zero-shot accuracy of CLIP or planes. Changes in accuracy could instead be due to
| RN50x64 |     | on the | three | splits | and report | All - Clean |     |     |     |     |     |     |
| ------- | --- | ------ | ----- | ------ | ---------- | ----------- | --- | --- | --- | --- | --- | --- |
changesintheclassdistributionordifficultyofthedupli-
asourmainmetric. Thisisthedifferenceinaccuracydue cates. Unfortunately,thesedistributionanddifficultyshifts
tocontamination. Whenpositiveitisourestimateofhow couldalsomasktheeffectsofover-fitting.
muchtheoverallreportedaccuracyonthedatasetwasin-
However,theseresultscloselyfollowthefindingsofsimi-
flatedbyover-fittingtooverlappingdata.
larduplicateanalysisinpreviousworkonlargescalepre-
3)Theamountofoverlapisoftensmallsowealsoruna
training. Mahajanetal.(2018)andKolesnikovetal.(2019)
binomial significance test where we use the accuracy on detectedsimilaroverlapratesandfoundminimalchangesin
Cleanasthenullhypothesisandcomputetheone-tailed overallperformance. Importantly,Kolesnikovetal.(2019)
(greater)p-valuefortheOverlapsubset.Wealsocalculate
|     |     |     |     |     |     |     | also compared | the | alternative | de-duplication |     | strategy dis- |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----------- | -------------- | --- | ------------- |
99.5%Clopper-PearsonconfidenceintervalsonDirtyas cussedintheintroductiontothissectionwiththeapproach
anothercheck.
wesettledonandobservedlittledifferencebetweenthetwo
approaches.
| AsummaryofthisanalysisispresentedinFigure17. |     |     |     |     |     | Out |     |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
of35datasetsstudied,9datasetshavenodetectedoverlap
6.Limitations
| at     | all. Most | of            | these datasets |       | are synthetic | or specialized   |     |     |     |     |     |     |
| ------ | --------- | ------------- | -------------- | ----- | ------------- | ---------------- | --- | --- | --- | --- | --- | --- |
| making |           | them unlikely |                | to be | posted as     | normal images on |     |     |     |     |     |     |
theinternet(forinstanceMNIST,CLEVR,andGTSRB)or TherearestillmanylimitationstoCLIP.Whileseveralof
thesearediscussedaspartofanalysisinvarioussections,
areguaranteedtohavenooverlapduetocontainingnovel
wesummarizeandcollectthemhere.
datafromafterthedateourdatasetwascreated(ObjectNet
andHatefulMemes). Thisdemonstratesourdetectorhas On datasets with training splits, the performance of zero-
a low-false positive rate which is important as false posi- shot CLIP is on average competitive with the simple su-
tiveswouldunder-estimatetheeffectofcontaminationin

LearningTransferableVisualModelsFromNaturalLanguageSupervision 19
20
10
0
-10
-20
0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 22.5
Detected Data Overlap (%)
)%(
ataD
naelC
.sv
gnippalrevO
no
ycaruccA
ni
ecnereffiD
0.75
0.5
CIFAR-100
0.25
SUN397 SUN
0
ImageNet Sketch -0.25
-0.5
Kinetics-700
-0.75
0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 22.5
Detected Data Overlap (%)
)%(
palrevO
oT
euD
egnahC
ycaruccA
llarevO
Birdsnap p < 1e-3
CIFAR-100 p < 0.05
p > 0.05
FER2013
SUN397 SUN Stanford Cars
Country211
Figure17.Fewstatisticallysignificantimprovementsinaccuracyduetodetecteddataoverlap. (Left)Whileseveraldatasetshave
upto±20%apparentdifferencesinzero-shotaccuracyondetectedoverlappingvscleanexamplesonly5datasetsoutof35totalhave
99.5%Clopper-Pearsonconfidenceintervalsthatexcludea0%accuracydifference.2ofthesedatasetsdoworseonoverlappingdata.
(Right)Sincethepercentageofdetectedoverlappingexamplesisalmostalwaysinthesingledigits,theoveralltestaccuracygaindueto
overlapismuchsmallerwiththelargestestimatedincreasebeingonly0.6%onBirdsnap.Similarly,foronly6datasetsaretheaccuracy
improvementsstatisticallysignificantwhencalculatedusingaone-sidedbinomialtest.
pervisedbaselineofalinearclassifierontopofResNet-50 CLIPlearnsahighqualitysemanticOCRrepresentationthat
features. On most of these datasets, the performance of performswellondigitallyrenderedtext,whichiscommon
thisbaselineisnowwellbelowtheoverallstateoftheart. initspre-trainingdataset,asevidencedbyperformanceon
Significantworkisstillneededtoimprovethetasklearning RenderedSST2. However,CLIPonlyachieves88%accu-
andtransfercapabilitiesofCLIP.Whilescalinghassofar racyonthehandwrittendigitsofMNIST.Anembarrassingly
steadilyimprovedperformanceandsuggestsarouteforcon- simplebaselineoflogisticregressiononrawpixelsoutper-
tinuedimprovement,weestimatearounda1000xincrease forms zero-shot CLIP. Both semantic and near-duplicate
incomputeisrequiredforzero-shotCLIPtoreachoverall nearest-neighborretrievalverifythattherearealmostnoim-
state-of-the-artperformance. Thisisinfeasibletotrainwith agesthatresembleMNISTdigitsinourpre-trainingdataset.
currenthardware. Furtherresearchintoimprovinguponthe This suggests CLIP does little to address the underlying
computationalanddataefficiencyofCLIPwillbenecessary. problemofbrittlegeneralizationofdeeplearningmodels.
InsteadCLIPtriestocircumventtheproblemandhopesthat
AnalysisinSection3.1foundthatCLIP’szero-shotperfor-
bytrainingonsuchalargeandvarieddatasetthatalldata
manceisstillquiteweakonseveralkindsoftasks. When
willbeeffectivelyin-distribution.Thisisanaiveassumption
comparedtotask-specificmodels,theperformanceofCLIP
that,asMNISTdemonstrates,iseasytoviolate.
ispooronseveraltypesoffine-grainedclassificationsuch
as differentiating models of cars, species of flowers, and AlthoughCLIPcanflexiblygeneratezero-shotclassifiers
variantsofaircraft. CLIPalsostruggleswithmoreabstract forawidevarietyoftasksanddatasets,CLIPisstilllimited
andsystematictaskssuchascountingthenumberofobjects tochoosingfromonlythoseconceptsinagivenzero-shot
inanimage. Finallyfornoveltaskswhichareunlikelytobe classifier. This is a significant restriction compared to a
includedinCLIP’spre-trainingdataset,suchasclassifying trulyflexibleapproachlikeimagecaptioningwhichcould
the distance to the nearest car in a photo, CLIP’s perfor- generatenoveloutputs. Unfortunately,asdescribedinSec-
mancecanbenearrandom. Weareconfidentthatthereare tion2.3wefoundthecomputationalefficiencyoftheimage
stillmany,many,taskswhereCLIP’szero-shotperformance captionbaselinewetriedtobemuchlowerthanCLIP.A
isnearchancelevel. simple idea worth trying is joint training of a contrastive
and generative objective with the hope of combining the
Whilezero-shotCLIPgeneralizeswelltomanynaturalim-
efficiencyofCLIPwiththeflexibilityofacaptionmodel.
agedistributionsasinvestigatedinSection3.3,we’veob-
Asanotheralternative,searchcouldbeperformedatinfer-
servedthatzero-shotCLIPstillgeneralizespoorlytodata
ence time over many natural language explanations of a
thatistrulyout-of-distributionforit. Anillustrativeexam-
givenimage,similartoapproachproposedinLearningwith
pleoccursforthetaskofOCRasreportedinAppendixE.
LatentLanguageAndreasetal.(2017).

20
LearningTransferableVisualModelsFromNaturalLanguageSupervision
CLIPalsodoesnotaddressthepoordataefficiencyofdeep 7.BroaderImpacts
| learning. | InsteadCLIPcompensatesbyusingasourceof |     |     |     |     |     |     |     |     |     |     |
| --------- | -------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
CLIPhasawiderangeofcapabilitiesduetoitsabilityto
| supervision       | that can be                      | scaled to | hundreds | of millions | of  |                                            |     |     |     |            |     |
| ----------------- | -------------------------------- | --------- | -------- | ----------- | --- | ------------------------------------------ | --- | --- | --- | ---------- | --- |
|                   |                                  |           |          |             |     | carryoutarbitraryimageclassificationtasks. |     |     |     | Onecangive |     |
| trainingexamples. | Ifeveryimageseenduringtrainingof |           |          |             |     |                                            |     |     |     |            |     |
itimagesofcatsanddogsandaskittoclassifycats,orgive
| a CLIP model | was presented | at a | rate of | one per | second, |     |     |     |     |     |     |
| ------------ | ------------- | ---- | ------- | ------- | ------- | --- | --- | --- | --- | --- | --- |
itimagestakeninadepartmentstoreandaskittoclassify
itwouldtake405yearstoiteratethroughthe12.8billion
shoplifters–ataskwithsignificantsocialimplicationsand
| images seen | over 32 | training epochs. | Combining |     | CLIP |                       |     |                            |     |     |     |
| ----------- | ------- | ---------------- | --------- | --- | ---- | --------------------- | --- | -------------------------- | --- | --- | --- |
|             |         |                  |           |     |      | forwhichAImaybeunfit. |     | Likeanyimageclassification |     |     |     |
withself-supervision(Henaff,2020;Chenetal.,2020c)and
system,CLIP’sperformanceandfitnessforpurposeneedto
self-training(Lee;Xieetal.,2020)methodsisapromising
beevaluated,anditsbroaderimpactsanalyzedincontext.
directiongiventheirdemonstratedabilitytoimprovedata
CLIPalsointroducesacapabilitythatwillmagnifyandalter
efficiencyoverstandardsupervisedlearning.
|                 |     |                     |     |              |     | suchissues: | CLIPmakesitpossibletoeasilycreateyour |     |     |     |     |
| --------------- | --- | ------------------- | --- | ------------ | --- | ----------- | ------------------------------------- | --- | --- | --- | --- |
| Our methodology | has | several significant |     | limitations. | De- |             |                                       |     |     |     |     |
ownclassesforcategorization(to‘rollyourownclassifier’)
spiteourfocusonzero-shottransfer,werepeatedlyqueried withoutaneedforre-training. Thiscapabilityintroduces
performance on full validation sets to guide the develop- challenges similar to those found in characterizing other,
mentofCLIP.Thesevalidationsetsoftenhavethousands large-scale generative models like GPT-3 (Brown et al.,
of examples, which is unrealistic for true zero-shot sce- 2020); models that exhibit non-trivial zero-shot (or few-
| narios. Similar | concerns | have been | raised | in the | field of |     |     |     |     |     |     |
| --------------- | -------- | --------- | ------ | ------ | -------- | --- | --- | --- | --- | --- | --- |
shot)generalizationcanhaveavastrangeofcapabilities,
semi-supervisedlearning(Oliveretal.,2018). Anotherpo- manyofwhicharemadeclearonlyaftertestingforthem.
| tentialissueisourselectionofevaluationdatasets. |     |     |     |     | While |     |     |     |     |     |     |
| ----------------------------------------------- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
we have reported results on Kornblith et al. (2019)’s 12 Our studies of CLIP in a zero-shot setting show that the
|                    |                |                    |     |             |      | model displays                   | significant                           | promise | for widely-applicable |             |      |
| ------------------ | -------------- | ------------------ | --- | ----------- | ---- | -------------------------------- | ------------------------------------- | ------- | --------------------- | ----------- | ---- |
| dataset evaluation | suite          | as a standardized  |     | collection, | our  |                                  |                                       |         |                       |             |      |
|                    |                |                    |     |             |      | taskslikeimageretrievalorsearch. |                                       |         | Forexample,itcanfind  |             |      |
| main results       | use a somewhat | haphazardly        |     | assembled   | col- |                                  |                                       |         |                       |             |      |
|                    |                |                    |     |             |      | relevant                         | images in a database                  | given   | text,                 | or relevant | text |
| lection of         | 27 datasets    | that is undeniably |     | co-adapted  | with |                                  |                                       |         |                       |             |      |
|                    |                |                    |     |             |      | givenanimage.                    | Further,therelativeeaseofsteeringCLIP |         |                       |             |      |
thedevelopmentandcapabilitiesofCLIP.Creatinganew
benchmark of tasks designed explicitly to evaluate broad towardbespokeapplicationswithlittleornoadditionaldata
ortrainingcouldunlockavarietyofnovelapplicationsthat
zero-shottransfercapabilities,ratherthanre-usingexisting
arehardforustoenvisiontoday,ashasoccurredwithlarge
superviseddatasets,wouldhelpaddresstheseissues.
languagemodelsoverthepastfewyears.
CLIPistrainedontextpairedwithimagesontheinternet.
Inadditiontothemorethan30datasetsstudiedinearlier
| These image-text | pairs | are unfiltered | and | uncurated | and |     |     |     |     |     |     |
| ---------------- | ----- | -------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
sectionsofthispaper,weevaluateCLIP’sperformanceon
| result in | CLIP models | learning many | social | biases. | This |              |           |               |     |             |      |
| --------- | ----------- | ------------- | ------ | ------- | ---- | ------------ | --------- | ------------- | --- | ----------- | ---- |
|           |             |               |        |         |      | the FairFace | benchmark | and undertake |     | exploratory | bias |
hasbeenpreviouslydemonstratedforimagecaptionmodels
probes. Wethencharacterizethemodel’sperformancein
| (Bhargava&Forsyth,2019). |     | WereferreaderstoSection7 |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | ------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
adownstreamtask,surveillance,anddiscussitsusefulness
fordetailedanalysisandquantificationofthesebehaviorsfor
|     |     |     |     |     |     | ascomparedwithotheravailablesystems. |     |     |     | ManyofCLIP’s |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | ------------ | --- |
CLIPaswellasdiscussionofpotentialmitigationstrategies.
|     |     |     |     |     |     | capabilitiesareomni-useinnature(e.g. |     |     | OCRcanbeused |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | ------------ | --- | --- |
Whilewehaveemphasizedthroughoutthisworkthatspeci- to make scanned documents searchable, to power screen
fyingimageclassifiersthroughnaturallanguageisaflexible reading technologies, or to read license plates). Several
andgeneralinterface,ithasitsownlimitations. Manycom- ofthecapabilitiesmeasured,fromactionrecognition,ob-
plex tasks and visual concepts can be difficult to specify jectclassification,andgeo-localization,tofacialemotion
| justthroughtext. | Actualtrainingexamplesareundeniably |     |     |     |     |              |                          |     |     |                |     |
| ---------------- | ----------------------------------- | --- | --- | --- | --- | ------------ | ------------------------ | --- | --- | -------------- | --- |
|                  |                                     |     |     |     |     | recognition, | canbeusedinsurveillance. |     |     | Givenitssocial |     |
usefulbutCLIPdoesnotoptimizeforfew-shotperformance implications,weaddressthisdomainofusespecificallyin
directly. Inourwork,wefallbacktofittinglinearclassifiers theSurveillancesection.
| ontopofCLIP’sfeatures. |     | Thisresultsinacounter-intuitive |     |     |     |     |     |     |     |     |     |
| ---------------------- | --- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Wehavealsosoughttocharacterizethesocialbiasesinher-
dropinperformancewhentransitioningfromazero-shot
|               |          |              |            |     |            | enttothemodel. | Ourbiastestsrepresentourinitialefforts |     |     |     |     |
| ------------- | -------- | ------------ | ---------- | --- | ---------- | -------------- | -------------------------------------- | --- | --- | --- | --- |
| to a few-shot | setting. | As discussed | in Section |     | 4, this is |                |                                        |     |     |     |     |
toprobeaspectsofhowthemodelrespondsindifferentsce-
notablydifferentfromhumanperformancewhichshowsa
|                                          |     |     |     |            |     | narios,andarebynaturelimitedinscope. |     |     |     | CLIPandmodels |     |
| ---------------------------------------- | --- | --- | --- | ---------- | --- | ------------------------------------ | --- | --- | --- | ------------- | --- |
| largeincreasefromazerotoaoneshotsetting. |     |     |     | Futurework |     |                                      |     |     |     |               |     |
isneededtodevelopmethodsthatcombineCLIP’sstrong likeitwillneedtobeanalyzedinrelationtotheirspecific
|     |     |     |     |     |     | deployments | to understand | how | bias manifests | and | iden- |
| --- | --- | --- | --- | --- | --- | ----------- | ------------- | --- | -------------- | --- | ----- |
zero-shotperformancewithefficientfew-shotlearning.
|     |     |     |     |     |     | tifypotentialinterventions. |     | Furthercommunityexploration |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------------------- | --- | --------------------------- | --- | --- | --- |
willberequiredtodevelopbroader,morecontextual,and
morerobusttestingschemessothatAIdeveloperscanbet-
tercharacterizebiasesingeneralpurposecomputervision
models.

21
LearningTransferableVisualModelsFromNaturalLanguageSupervision
| Model           |     | Race | Gender | Age  |     | Model           |     |     | Race | Gender | Age  |     |
| --------------- | --- | ---- | ------ | ---- | --- | --------------- | --- | --- | ---- | ------ | ---- | --- |
| FairFaceModel   |     | 93.7 | 94.2   | 59.7 |     | FairFaceModel   |     |     | 75.4 | 94.4   | 60.7 |     |
|                 |     |      |        |      |     | LinearProbeCLIP |     |     | 92.8 | 97.7   | 63.1 |     |
| LinearProbeCLIP |     | 93.4 | 96.5   | 63.8 |     |                 |     |     |      |        |      |     |
| Zero-ShotCLIP   |     | 58.3 | 95.9   | 57.1 |     | Zero-ShotCLIP   |     |     | 91.3 | 97.2   | 54.3 |     |
LinearProbeInstagram 90.8 93.2 54.2 LinearProbeInstagram 87.2 93.9 54.1
Table3.PercentaccuracyonRace,Gender,andAgeclassification Table4.PercentaccuracyonRace,Gender,andAgeclassification
ofimagesinFairFacecategory‘White’ ofimagesinFairFacecategories‘Black,’‘Indian,’‘EastAsian,’
|     |     |     |     |     | ‘Southeast | Asian,’ | ‘Middle | Eastern,’ | and | ‘Latino’ | (grouped | to- |
| --- | --- | --- | --- | --- | ---------- | ------- | ------- | --------- | --- | -------- | -------- | --- |
getherasFairFacecategory‘Non-White’)
|     |     |     |     |     |     |     | Middle | Southeast | East |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | --------- | ---- | --- | --- | --- |
Model Gender Black White Indian Latino Eastern Asian Asian Average
|     |     |     | Male | 96.9 96.4 | 98.7 | 96.5 | 98.9 | 96.2 | 96.9 | 97.2 |     |     |
| --- | --- | --- | ---- | --------- | ---- | ---- | ---- | ---- | ---- | ---- | --- | --- |
LinearProbeCLIP Female 97.9 96.7 97.9 99.2 97.2 98.5 97.3 97.8
|     |               |     |        | 97.4 96.5 | 98.3 | 97.8 | 98.4 | 97.3 | 97.1 | 97.5 |     |     |
| --- | ------------- | --- | ------ | --------- | ---- | ---- | ---- | ---- | ---- | ---- | --- | --- |
|     |               |     | Male   | 96.3 96.4 | 97.7 | 97.2 | 98.3 | 95.5 | 96.8 | 96.9 |     |     |
|     | Zero-ShotCLIP |     | Female | 97.1 95.3 | 98.3 | 97.8 | 97.5 | 97.2 | 96.4 | 97.0 |     |     |
|     |               |     |        | 96.7 95.9 | 98.0 | 97.5 | 98.0 | 96.3 | 96.6 |      |     |     |
|     |               |     | Male   | 92.5 94.8 | 96.2 | 93.1 | 96.0 | 92.7 | 93.4 | 94.1 |     |     |
LinearProbeInstagram Female 90.1 91.4 95.0 94.8 95.0 94.1 94.3 93.4
|     |     |     |     | 91.3 93.2 | 95.6 | 94.0 | 95.6 | 93.4 | 93.9 |     |     |     |
| --- | --- | --- | --- | --------- | ---- | ---- | ---- | ---- | ---- | --- | --- | --- |
Table5. PercentaccuracyongenderclassificationofimagesbyFairFaceracecategory
7.1.Bias as an initial bias probe, then probe the model further to
surfaceadditionalbiasesandsourcesofbiases,including
Algorithmicdecisions,trainingdata,andchoicesabouthow
classdesign.
classesaredefinedandtaxonomized(whichwerefertoin-
formallyas“classdesign”)canallcontributetoandamplify WeevaluatedtwoversionsofCLIPontheFairFacedataset:
socialbiasesandinequalitiesresultingfromtheuseofAI azero-shotCLIPmodel(“ZSCLIP”),andalogisticregres-
sionclassifierfittedtoFairFace’sdatasetontopofCLIP’s
systems(Noble,2018;Bechmann&Bowker,2019;Bowker
&Star,2000). Classdesignisparticularlyrelevanttomod- features(“LRCLIP”).WefindthatLRCLIPgetshigher
elslikeCLIP,sinceanydevelopercandefineaclassandthe accuracyontheFairFacedatasetthanboththeResNext-101
modelwillprovidesomeresult. 32x48dInstagrammodel(“LinearProbeInstagram”)(Ma-
hajanetal.,2018)andFairFace’sownmodelonmostofthe
| In this | section, we provide | preliminary | analysis | of some |                            |     |     |     |                           |     |     |     |
| ------- | ------------------- | ----------- | -------- | ------- | -------------------------- | --- | --- | --- | ------------------------- | --- | --- | --- |
|         |                     |             |          |         | classificationtestsweran7. |     |     |     | ZSCLIP’sperformancevaries |     |     |     |
ofthebiasesinCLIP,usingbiasprobesinspiredbythose
bycategoryandisworsethanthatofFairFace’smodelfora
outlinedinBuolamwini&Gebru(2018)andKa¨rkka¨inen
|             |                                      |     |     |     | fewcategories,andbetterforothers. |     |     |     |     | (SeeTable3andTable |     |     |
| ----------- | ------------------------------------ | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | ------------------ | --- | --- |
| &Joo(2019). | Wealsoconductexploratorybiasresearch |     |     |     |                                   |     |     |     |     |                    |     |     |
4).
intendedtofindspecificexamplesofbiasesinthemodel,
similartothatconductedbySolaimanetal.(2019). andKeyes(2018)haveshown.WhileFairFace’sdatasetreduces
theproportionofWhitefaces,itstilllacksrepresentationofentire
WestartbyanalyzingtheperformanceofZero-ShotCLIPon largedemographicgroups,effectivelyerasingsuchcategories.We
thefaceimagedatasetFairFace(Ka¨rkka¨inen&Joo,2019)6 usethe2gendercategoriesand7racecategoriesdefinedinthe
FairFacedatasetinanumberofourexperimentsnotinorderto
6FairFaceisafaceimagedatasetdesignedtobalanceage,gen- reinforceorendorsetheuseofsuchreductivecategories,butin
der,andrace,inordertoreduceasymmetriescommoninprevious ordertoenableustomakecomparisonstopriorwork.
facedatasets.Itcategorizesgenderinto2groups:femaleandmale 7OnechallengewiththiscomparisonisthattheFairFacemodel
andraceinto7groups:White,Black,Indian,EastAsian,Southeast usesbinaryclassesforrace(“White”and“Non-White”),instead
Asian,MiddleEastern,andLatino.Thereareinherentproblems ofbreakingdownracesintofiner-grainedsub-groups.
withraceandgenderclassifications,ase.g.Bowker&Star(2000)

22
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     |                         |             |               | Middle  | Southeast | East  |     |
| --- | ----------------------- | ----------- | ------------- | ------- | --------- | ----- | --- |
|     | Category                | Black White | Indian Latino | Eastern | Asian     | Asian |     |
|     | Crime-relatedCategories | 16.4 24.9   | 24.4 10.8     | 19.7    | 4.4       | 1.3   |     |
|     | Non-humanCategories     | 14.4 5.5    | 7.6           | 3.7 2.0 | 1.9       | 0.0   |     |
Table6.Percentofimagesclassifiedintocrime-relatedandnon-humancategoriesbyFairFaceRacecategory.Thelabelsetincluded7
FairFaceracecategorieseachformenandwomen(foratotalof14),aswellas3crime-relatedcategoriesand4non-humancategories.
CategoryLabelSet 0-2 3-9 10-19 20-29 30-39 40-49 50-59 60-69 over70
| DefaultLabelSet |     | 30.3 35.0 | 29.5 16.3 | 13.9 | 18.5 19.1 | 16.2 | 10.4 |
| --------------- | --- | --------- | --------- | ---- | --------- | ---- | ---- |
DefaultLabelSet+‘child’category 2.3 4.3 14.7 15.0 13.4 18.2 18.6 15.5 9.4
Table7.Percentofimagesclassifiedintocrime-relatedandnon-humancategoriesbyFairFaceAgecategory,showingcomparisonbetween
resultsobtainedusingadefaultlabelsetandalabelsettowhichthelabel’child’hasbeenadded.Thedefaultlabelsetincluded7FairFace
racecategorieseachformenandwomen(foratotalof14),3crime-relatedcategoriesand4non-humancategories.
Additionally,wetesttheperformanceoftheLRCLIPand We found that 4.9% (confidence intervals between 4.6%
ZSCLIPmodelsacrossintersectionalraceandgendercate- and 5.4%) of the images were misclassified into one of
goriesastheyaredefinedintheFairFacedataset. Wefind the non-human classes we used in our probes (‘animal’,
thatmodelperformanceongenderclassificationisabove ‘chimpanzee’,‘gorilla’,‘orangutan’). Outofthese,‘Black’
95%forallracecategories. Table5summarizesthesere- imageshadthehighestmisclassificationrate(approximately
| sults. |     |     | 14%; confidence | intervals | between | [12.6% | and 16.4%]) |
| ------ | --- | --- | --------------- | --------- | ------- | ------ | ----------- |
whileallotherraceshadmisclassificationratesunder8%.
WhileLRCLIPachieveshigheraccuracythantheLinear
|     |     |     | People | aged 0-20 years | had the | highest proportion | being |
| --- | --- | --- | ------ | --------------- | ------- | ------------------ | ----- |
ProbeInstagrammodelontheFairFacebenchmarkdataset
classifiedintothiscategoryat14%.
forgender,raceandageclassificationofimagesbyintersec-
tionalcategories,accuracyonbenchmarksoffersonlyone Wealsofoundthat16.5%ofmaleimagesweremisclassified
approximationofalgorithmicfairness,asRajietal.(2020) intoclassesrelatedtocrime(‘thief’,‘suspiciousperson’and
haveshown,andoftenfailsasameaningfulmeasureoffair- ‘criminal’) as compared to 9.8% of female images. Inter-
nessinrealworldcontexts. Evenifamodelhasbothhigher estingly, we found that people aged 0-20 years old were
accuracyandlowerdisparitiesinperformanceondifferent morelikelytofallunderthesecrime-relatedclasses(approx-
sub-groups,thisdoesnotmeanitwillhavelowerdisparities imately 18%) compared to images of people in different
inimpact(Scheuermanetal.,2019). Forexample,higher ageranges(approximately12%forpeopleaged20-60and
performanceonunderrepresentedgroupsmightbeusedby 0%forpeopleover70). Wefoundsignificantdisparitiesin
acompanytojustifytheiruseoffacialrecognition,andto classificationsacrossracesforcrimerelatedterms,whichis
| thendeployitwaysthataffectdemographicgroupsdispro- |                                          |     | capturedinTable6. |     |     |     |     |
| -------------------------------------------------- | ---------------------------------------- | --- | ----------------- | --- | --- | --- | --- |
| portionately.                                      | Ouruseoffacialclassificationbenchmarksto |     |                   |     |     |     |     |
Giventhatweobservedthatpeopleunder20werethemost
probeforbiasesisnotintendedtoimplythatfacialclassi-
|     |     |     | likely to | be classified | in both | the crime-related | and non- |
| --- | --- | --- | --------- | ------------- | ------- | ----------------- | -------- |
ficationisanunproblematictask,nortoendorsetheuseof
humananimalcategories,wecarriedoutclassificationfor
race,age,orgenderclassificationindeployedcontexts.
|     |     |     | the images | with the | same classes | but with | an additional |
| --- | --- | --- | ---------- | -------- | ------------ | -------- | ------------- |
Wealsoprobedthemodelusingclassificationtermswith category ‘child’ added to the categories. Our goal here
highpotentialtocauserepresentationalharm,focusingon wastoseeifthiscategorywouldsignificantlychangethe
denigrationharmsinparticular(Crawford,2017). Wecar- behaviourofthemodelandshifthowthedenigrationharms
riedoutanexperimentinwhichtheZSCLIPmodelwas aredistributedbyage.Wefoundthatthisdrasticallyreduced
requiredtoclassify10,000imagesfromtheFairFacedataset. thenumberofimagesofpeopleunder20classifiedineither
InadditiontotheFairFaceclasses,weaddedinthefollow- crime-related categories or non-human animal categories
ingclasses: ‘animal’,‘gorilla’,‘chimpanzee’,‘orangutan’, (Table7). Thispointstohowclassdesignhasthepotential
‘thief’,‘criminal’and‘suspiciousperson’. Thegoalofthis tobeakeyfactordeterminingboththemodelperformance
experimentwastocheckifharmsofdenigrationdispropor- andtheunwantedbiasesorbehaviourthemodelmayexhibit
tionatelyimpactcertaindemographicsubgroups. whilealsoasksoverarchingquestionsabouttheuseofface

23
LearningTransferableVisualModelsFromNaturalLanguageSupervision
images to automatically classify people along such lines likefordeployingsuchsystems.
(yArcasetal.,2017).
WhengiventhecombinedsetoflabelsthatGoogleCloud
Theresultsoftheseprobescanchangebasedontheclass Vision(GCV),AmazonRekognitionandMicrosoftreturned
categories one chooses to include as well as the specific foralltheimages,similartothebiasesSchwemmeretal.
languageoneusestodescribeeachclass. Poorclassdesign (2020)foundinGCVsystems,wefoundoursystemalso
can lead to poor real world performance; this concern is disproportionatelyattachedlabelstodowithhairandap-
particularlyrelevanttoamodellikeCLIP,givenhoweasily pearance in general to women more than men. For ex-
developerscandesigntheirownclasses. ample, labels such as ‘brown hair’, ‘blonde’ and ‘blond’
|     |     |     |     |     |     |     | appearedsignificantlymoreoftenforwomen. |     |     | Additionally, |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | ------------- |
Wealsocarriedoutexperimentssimilartothoseoutlinedby
CLIPattachedsomelabelsthatdescribedhighstatusoccu-
Schwemmeretal.(2020)totesthowCLIPtreatedimages
pationsdisproportionatelymoreoftentomensuchas‘ex-
ofmenandwomendifferentlyusingimagesofMembers
|              |     |            |       |              |     |            | ecutive’and‘doctor’. | Outoftheonlyfouroccupationsthat |     |     |
| ------------ | --- | ---------- | ----- | ------------ | --- | ---------- | -------------------- | ------------------------------- | --- | --- |
| of Congress. |     | As part of | these | experiments, |     | we studied |                      |                                 |     |     |
itattachedmoreoftentowomen,threewere‘newscaster’,
| how certain | additional | design |     | decisions | such | as deciding |     |     |     |     |
| ----------- | ---------- | ------ | --- | --------- | ---- | ----------- | --- | --- | --- | --- |
‘televisionpresenter’and‘newsreader’andthefourthwas
thresholdsforlabelscanimpactthelabelsoutputbyCLIP
‘Judge’. ThisisagainsimilartothebiasesfoundinGCV
andhowbiasesmanifest.
andpointstohistoricalgendereddifferences(Schwemmer
| Wecarriedoutthreeexperiments-wetestedforaccuracy |     |     |     |     |     |     | etal.,2020). |     |     |     |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- |
ongenderclassificationandwetestedforhowlabelswere
|                                                       |     |     |     |     |     |     | Interestingly, | whenweloweredthethresholdto0.5%for |     |     |
| ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------- | ---------------------------------- | --- | --- |
| differentiallydistributedacrosstwodifferentlabelsets. |     |     |     |     |     | For |                |                                    |     |     |
thissetoflabels,wefoundthatthelabelsdisproportionately
| ourfirstlabelset,weusedalabelsetof |     |     |     |     | 300occupationsand |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
describingmenalsoshiftedtoappearanceorientedwords
foroursecondlabelsetweusedacombinedsetoflabelsthat
|     |     |     |     |     |     |     | suchas‘suit’,‘tie’and‘necktie’(Figure18). |     |     | Manyoccupa- |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------------------- | --- | --- | ----------- |
GoogleCloudVision,AmazonRekognitionandMicrosoft
tionorientedwordssuchas‘militaryperson’and‘executive’
AzureComputerVisionreturnedforalltheimages.
-whichwerenotusedtodescribeimagesofwomenatthe
Wefirstsimplylookedintogenderpredictionperformance higher4%threshold-wereusedforbothmenandwomen
of the model on the images of Members of Congress, in atthelower0.5%threshold,whichcouldhavecausedthe
order to check to see if the model correctly recognized changeinlabelsformen. Thereversewasnottrue. Descrip-
men as men and women as women given the image of a tivewordsusedtodescribewomenwerestilluncommon
| personwhoappearedtobeinanofficialsetting/positionof |     |     |     |     |     |     | amongstmen. |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- |
power. Wefoundthatthemodelgot100%accuracyonthe
Designdecisionsateverystageofbuildingamodelimpact
images. Thisisslightlybetterperformancethanthemodel’s
|                                  |         |          |     |          |                   |        | how biases                   | manifest and this | is especially            | true for CLIP |
| -------------------------------- | ------- | -------- | --- | -------- | ----------------- | ------ | ---------------------------- | ----------------- | ------------------------ | ------------- |
| performanceontheFairFacedataset. |         |          |     |          | Wehypothesizethat |        |                              |                   |                          |               |
|                                  |         |          |     |          |                   |        | giventheflexibilityitoffers. |                   | Inadditiontochoicesabout |               |
| one of the                       | reasons | for this | is  | that all | the images        | in the |                              |                   |                          |               |
trainingdataandmodelarchitecture,decisionsaboutthings
MembersofCongressdatasetwerehigh-qualityandclear,
likeclassdesignsandthresholdingvaluescanalterthelabels
withthepeopleclearlycentered,unlikethoseintheFairFace
amodeloutputsandasaresultheightenorlowercertain
dataset.
kindsofharm,suchasthosedescribedbyCrawford(2017).
Inordertostudyhowthebiasesinreturnedlabelsdependon PeopledesigninganddevelopingmodelsandAIsystems
thethresholdssetforlabelprobability,wedidanexperiment haveconsiderablepower. Decisionsaboutthingslikeclass
in which we set threshold values at 0.5% and 4.0%. We designareakeydeterminernotonlyofmodelperformance,
foundthatthelowerthresholdledtolowerqualityoflabels. butalsoofhowandinwhatcontextsmodelbiasesmanifest.
| However,                            | even | the differing | distributions |     | of          | labels under |                   |         |                |             |
| ----------------------------------- | ---- | ------------- | ------------- | --- | ----------- | ------------ | ----------------- | ------- | -------------- | ----------- |
|                                     |      |               |               |     |             |              | These experiments | are not | comprehensive. | They illus- |
| thisthresholdcanholdsignalsforbias. |      |               |               |     | Forexample, | we           |                   |         |                |             |
tratepotentialissuesstemmingfromclassdesignandother
| find that | under | the 0.5% | threshold | labels | such | as ‘nanny’ |     |     |     |     |
| --------- | ----- | -------- | --------- | ------ | ---- | ---------- | --- | --- | --- | --- |
sourcesofbias,andareintendedtosparkinquiry.
and‘housekeeper’startappearingforwomenwhereaslabels
| such as | ‘prisoner’ | and ‘mobster’ |     | start | appearing | for men. |     |     |     |     |
| ------- | ---------- | ------------- | --- | ----- | --------- | -------- | --- | --- | --- | --- |
7.2.Surveillance
| This points | to  | gendered | associations | similar |     | to those that |     |     |     |     |
| ----------- | --- | -------- | ------------ | ------- | --- | ------------- | --- | --- | --- | --- |
havepreviouslybeenfoundforoccupations(Schwemmer We next sought to characterize model performance in re-
etal.,2020)(Noseketal.,2002)(Bolukbasietal.,2016).
lationtoadownstreamtaskforwhichthereissignificant
|     |     |     |     |     |     |     | societalsensitivity:surveillance. |     | Ouranalysisaimstobetter |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------- | --- | ----------------------- | --- |
Atthehigher4%threshold,thelabelswiththehighestprob-
abilityacrossbothgendersinclude“lawmaker”,“legislator” embodythecharacterizationapproachdescribedaboveand
tohelporienttheresearchcommunitytowardsthepotential
| and“congressman”. |     | However,thepresenceofthesebiases |     |     |     |     |                |                 |         |                  |
| ----------------- | --- | -------------------------------- | --- | --- | --- | --- | -------------- | --------------- | ------- | ---------------- |
|                   |     |                                  |     |     |     |     | future impacts | of increasingly | general | purpose computer |
amongstlowerprobabilitylabelsnonethelesspointtolarger
visionmodelsandaidthedevelopmentofnormsandchecks
questionsaboutwhat‘sufficiently’safebehaviourmaylook

24
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|                 |            | Top labels,     |     |        |                   |       | Top labels,   |        |
| --------------- | ---------- | --------------- | --- | ------ | ----------------- | ----- | ------------- | ------ |
|                 |            | images of women |     |        |                   |       | images of men |        |
|                 | woman      |                 |     |        |                   | man   |               |        |
|                 | lady       |                 |     |        |                   | male  |               |        |
|                 | female     |                 |     |        |                   | face  |               |        |
|                 | looking    |                 |     |        | player            |       |               |        |
| senior citizen  |            |                 |     |        |                   | black |               |        |
| public speaking |            |                 |     |        |                   | head  |               |        |
|                 | blonde     |                 |     |        | facial expression |       |               |        |
| spokesperson    |            |                 |     |        |                   | suit  |               |        |
|                 | blazer     |                 |     |        | photo             |       |               |        |
|                 | laughing   |                 |     |        | military officer  |       |               |        |
|                 | hot        |                 |     |        | walking           |       |               |        |
|                 | magenta    |                 |     |        | photograph        |       |               |        |
|                 | bob cut    |                 |     |        |                   | elder |               |        |
|                 | black hair |                 |     |        | display           |       |               |        |
|                 | pixie cut  |                 |     |        |                   | tie   |               |        |
|                 | pink       |                 |     |        | shoulder          |       |               |        |
|                 | bangs      |                 |     |        | frown             |       |               |        |
|                 | newsreader |                 |     |        |                   | kid   |               |        |
|                 | purple     |                 |     |        | necktie           |       |               |        |
|                 |            |                 |     | Women  |                   |       |               | Women  |
|                 | blouse     |                 |     | Men    | yellow            |       |               | Men    |
|                 | 0          | 20 40           | 60  | 80 100 |                   | 0 20  | 40 60         | 80 100 |
|                 |            | Frequency (%)   |     |        |                   |       | Frequency (%) |        |
Figure18.CLIPperformanceonMemberofCongressimageswhengiventhecombinedreturnedlabelsetfortheimagesfromGoogle
CloudVision,AmazonRekognitionandMicrosoftAzureComputerVision. The20mostgenderedlabelsformenandwomenwere
identifiedwithχ2testswiththethresholdat0.5%.Labelsaresortedbyabsolutefrequencies.Barsdenotethepercentageofimagesfora
certainlabelbygender.
aroundsuchsystems. Ourinclusionofsurveillanceisnot themodeltochoosefrom. Additionally,wecarriedouta
intendedtoindicateenthusiasmforthisdomain-rather,we ‘stresstest’wheretheclasssetincludedatleastonemore
think surveillance is an important domain to try to make caption for something that was ‘close’ to the image (for
predictions about given its societal implications (Zuboff, example,‘parkinglotwithwhitecar’vs. ‘parkinglotwith
2015;Browne,2015). red car’). We found that the model had a top-1 accuracy
|     |     |     |     |     | of 91.8% | on the CCTV | images for | the initial evaluation. |
| --- | --- | --- | --- | --- | -------- | ----------- | ---------- | ----------------------- |
Wemeasurethemodel’sperformanceonclassificationof
Theaccuracydroppedsignificantlyto51.1%forthesecond
imagesfromCCTVcamerasandzero-shotcelebrityidentifi-
evaluation,withthemodelincorrectlychoosingthe‘close’
cation. Wefirsttestedmodelperformanceonlow-resolution
answer40.7%ofthetime.
| images captured | from surveillance | cameras | (e.g. | CCTV |     |     |     |     |
| --------------- | ----------------- | ------- | ----- | ---- | --- | --- | --- | --- |
cameras). WeusedtheVIRATdataset(Ohetal.,2011)and Forfine-graineddetection,thezero-shotmodelperformed
datacapturedbyVaradarajan&Odobez(2009),whichboth poorly,withresultsnearrandom. Notethatthisexperiment
consistofrealworldoutdoorsceneswithnon-actors. wastargetedonlytowardsdetectingthepresenceorabsence
ofsmallobjectsinimagesequences.
| Given CLIP’s | flexible class | construction, | we  | tested 515 |     |     |     |     |
| ------------ | -------------- | ------------- | --- | ---------- | --- | --- | --- | --- |
surveillance images captured from 12 different video se- We also tested CLIP’s zero-shot performance for ‘in the
quencesonself-constructedgeneralclassesforcoarseand wild’identitydetectionusingtheCelebAdataset8. Wedid
finegrainedclassification. Coarseclassificationrequiredthe thistoevaluatethemodel’sperformanceforidentitydetec-
modeltocorrectlyidentifythemainsubjectoftheimage(i.e. tionusingjustthepubliclyavailabledataitwaspre-trained
determineiftheimagewasapictureofanemptyparking on.Whilewetestedthisonadatasetofcelebritieswhohave
lot,schoolcampus,etc.). Forfine-grainedclassification,the alargernumberofimagesontheinternet,wehypothesize
model had to choose between two options constructed to thatthenumberofimagesinthepre-trainingdataneeded
determineifthemodelcouldidentifythepresence/absence forthemodeltoassociatefaceswithnameswillkeepde-
ofsmallerfeaturesintheimagesuchasapersonstanding creasingasmodelsgetmorepowerful(seeTable8),which
inthecorner. hassignificantsocietalimplications(Garvie,2019). This
Forcoarseclassification,weconstructedtheclassesbyhand-
8Note:TheCelebAdatasetismorerepresentativeoffaceswith
captioning the images ourselves to describe the contents lighterskintones. Duetothenatureofthedataset,wewerenot
oftheimageandtherewerealwaysatleast6optionsfor abletocontrolforrace,gender,age,etc.

25
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     |     |     |     |     |     |     | We hope | that this work | motivates | future | research | on the |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------------- | --------- | ------ | -------- | ------ |
characterizationofthecapabilities,shortcomings,andbiases
| Model    |     | 100Classes |      | 1kClasses |     | 2kClasses |                 |     |        |         |           |          |
| -------- | --- | ---------- | ---- | --------- | --- | --------- | --------------- | --- | ------ | ------- | --------- | -------- |
|          |     |            |      |           |     |           | of such models, | and | we are | excited | to engage | with the |
| CLIPL/14 |     |            | 59.2 | 43.3      |     | 42.2      |                 |     |        |         |           |          |
researchcommunityonsuchquestions.
| CLIPRN50x64 |     |     | 56.4 | 39.5 |     | 38.4 |     |     |     |     |     |     |
| ----------- | --- | --- | ---- | ---- | --- | ---- | --- | --- | --- | --- | --- | --- |
CLIPRN50x16 52.7 37.4 36.3 Webelieveonegoodstepforwardiscommunityexploration
CLIPRN50x4 52.8 38.1 37.3 tofurthercharacterizethecapabilitiesofmodelslikeCLIP
and-crucially-identifyapplicationareaswheretheyhave
|     |     |     |     |     |     |     | promising | performance | and areas | where | they | may have |
| --- | --- | --- | --- | --- | --- | --- | --------- | ----------- | --------- | ----- | ---- | -------- |
Table8.CelebAZero-ShotTop-1IdentityRecognitionAccuracy
|     |     |     |     |     |     |     | reducedperformance9. |          | Thisprocessofcharacterizationcan |            |        |          |
| --- | --- | --- | --- | --- | --- | --- | -------------------- | -------- | -------------------------------- | ---------- | ------ | -------- |
|     |     |     |     |     |     |     | help researchers     | increase | the                              | likelihood | models | are used |
mirrorsrecentdevelopmentsinnaturallanguageprocessing,
beneficiallyby:
inwhichrecentlargelanguagemodelstrainedonInternet
| data often | exhibit | a surprising |     | ability to | provide | informa- |     |     |     |     |     |     |
| ---------- | ------- | ------------ | --- | ---------- | ------- | -------- | --- | --- | --- | --- | --- | --- |
• Identifyingpotentiallybeneficialdownstreamusesof
tionrelatedtorelativelyminorpublicfigures(Brownetal.,
|     |     |     |     |     |     |     | models | early in | the research | process, | enabling | other |
| --- | --- | --- | --- | --- | --- | --- | ------ | -------- | ------------ | -------- | -------- | ----- |
2020).
researcherstothinkaboutapplications.
| We found | that | the model | had | 59.2% top-1 | accuracy | out |     |     |     |     |     |     |
| -------- | ---- | --------- | --- | ----------- | -------- | --- | --- | --- | --- | --- | --- | --- |
• Surfacingtaskswithsignificantsensitivityandalarge
| of 100 possible |     | classes | for ‘in | the wild’ | 8k celebrity | im- |     |     |     |     |     |     |
| --------------- | --- | ------- | ------- | --------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
ages. However,thisperformancedroppedto43.3%when setofsocietalstakeholders,whichmaycallforinter-
ventionbypolicymakers.
| we increased | our | class | sizes to | 1k celebrity | names. | This |     |     |     |     |     |     |
| ------------ | --- | ----- | -------- | ------------ | ------ | ---- | --- | --- | --- | --- | --- | --- |
performanceisnotcompetitivewhencomparedtoproduc-
• Bettercharacterizingbiasesinmodels,alertingother
tionlevelmodelssuchasGoogle’sCelebrityRecognition
researcherstoareasofconcernandareasforinterven-
| (Google). | However,whatmakestheseresultsnoteworthyis |     |     |     |     |     |     |     |     |     |     |     |
| --------- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tions.
thatthisanalysiswasdoneusingonlyzero-shotidentifica-
tioncapabilitiesbasedonnamesinferredfrompre-training • CreatingsuitesofteststoevaluatesystemslikeCLIP
data-wedidn’tuseanyadditionaltask-specificdataset,and
|     |     |     |     |     |     |     | on, so | we can | better characterize |     | model | capabilities |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------ | ------------------- | --- | ----- | ------------ |
sothe(relatively)strongresultsfurtherindicatethatbefore
earlierinthedevelopmentcycle.
deployingmultimodalmodels,peoplewillneedtocarefully
studythemforbehaviorsinagivencontextanddomain. • Identifyingpotentialfailuremodesandareasforfurther
work.
CLIPofferssignificantbenefitfortasksthathaverelatively
| littledatagivenitszero-shotcapabilities. |     |     |     |     | However,large |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
datasetsandhighperformingsupervisedmodelsexistfor Weplantocontributetothiswork,andhopethisanalysis
manyin-demandsurveillancetaskssuchasfacialrecogni- providessomemotivatingexamplesforsubsequentresearch.
tion. Asaresult,CLIP’scomparativeappealforsuchuses
| is low. | Additionally, | CLIP | is  | not designed | for | common | 8.RelatedWork |     |     |     |     |     |
| ------- | ------------- | ---- | --- | ------------ | --- | ------ | ------------- | --- | --- | --- | --- | --- |
surveillance-relevanttaskslikeobjectdetectionandseman-
ticsegmentation. Thismeansithaslimiteduseforcertain Any model that leverages written, spoken, signed or any
surveillancetaskswhenmodelsthataredesignedwiththese otherformofhumanlanguageaspartofitstrainingsignal
usesinmindsuchasDetectron2(Wuetal.,2019)arewidely isarguablyusingnaturallanguageasasourceofsupervi-
sion. Thisisanadmittedlyextremelybroadareaandcovers
available.
mostworkinthefieldofdistributionalsemanticsincluding
| However, | CLIP | does unlock |     | a certain aspect |     | of usability |     |     |     |     |     |     |
| -------- | ---- | ----------- | --- | ---------------- | --- | ------------ | --- | --- | --- | --- | --- | --- |
topicmodels(Bleietal.,2003),word,sentence,andpara-
| givenhowitremovestheneedfortrainingdata. |     |     |     |     |     | Thus,CLIP |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- | --- |
graphvectors(Mikolovetal.,2013;Kirosetal.,2015;Le&
andsimilarmodelscouldenablebespoke,nichesurveillance
Mikolov,2014),andlanguagemodels(Bengioetal.,2003).
usecasesforwhichnowell-tailoredmodelsordatasetsexist,
ItalsoincludesmuchofthebroaderfieldofNLPthatdeals
andcouldlowertheskillrequirementstobuildsuchappli-
withpredictingormodelingsequencesofnaturallanguage
cations. Asourexperimentsshow,ZSCLIPdisplaysnon-
|     |     |     |     |     |     |     | insomeway. | WorkinNLPintentionallyleveragingnatural |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --------------------------------------- | --- | --- | --- | --- |
trivial,butnotexceptional,performanceonafewsurveil- languagesupervisionintheformofexplanations,feedback,
lancerelevanttaskstoday.
instructions,andadvicefortaskssuchasclassification(as
opposedtothecommonlyusedrepresentationofsupervision
7.3.FutureWork
asasetofarbitrarilyencodeddiscretecategorylabels)has
Thispreliminaryanalysisisintendedtoillustratesomeof 9Amodelcouldbeunfitforuseduetoinadequateperformance
thechallengesthatgeneralpurposecomputervisionmodels
orduetotheinappropriatenessofAIuseintheapplicationarea
| pose and | to give | a glimpse | into | their biases | and | impacts. |     |     |     |     |     |     |
| -------- | ------- | --------- | ---- | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- |
itself.

LearningTransferableVisualModelsFromNaturalLanguageSupervision 26
beenexploredinmanycreativeandadvancedways. Dialog largescalerepresentationlearningbytrainingasystemto
basedlearning(Weston,2016;Lietal.,2016;Hancocketal., pairdescriptivetextwithvideosinsteadofimages. Several
2019)developstechniquestolearnfrominteractivenatural workshaveexploredusingdensespokennaturallanguage
languagefeedbackindialog. Severalpapershaveleveraged supervisionforvideos(Miechetal.,2019;2020b). When
semanticparsingtoconvertnaturallanguageexplanations considered together with CLIP, these works suggest that
intofeatures(Srivastavaetal.,2017)oradditionaltraining largescalenaturallanguagesupervisionisapromisingway
labels (Hancock et al., 2018). More recently, ExpBERT tolearnhighqualityperceptualsystemsformanydomains.
(Murtyetal.,2020)usesfeaturerepresentationsproduced Alayracetal.(2020)extendedthislineofworktoanaddi-
byconditioningadeepcontextuallanguagemodelonnat- tionalmodalitybyaddingrawaudioasanadditionalsuper-
urallanguageexplanationsanddescriptionsofrelationsto visionsourceanddemonstratedbenefitsfromcombiningall
improveperformanceonthetaskofrelationextraction. threesourcesofsupervision.
CLIPisanexampleofusingnaturallanguageasatraining AspartofourworkonCLIPwealsoconstructanewdataset
signalforlearningaboutadomainotherthanlanguage. In ofimage-textpairs. Modernworkonimage-textretrieval
thiscontext,theearliestuseofthetermnaturallanguage has relied on a set of crowd-sourced sentence level im-
supervisionthatweareawareofistheworkofRamanathan agecaptionevaluationdatasetslikePascal1K(Rashtchian
etal.(2013)whichshowedthatnaturallanguagedescrip- etal.,2010),Flickr8K(Hodoshetal.,2013),andFlickr30K
tionscouldbeusedalongsideothersourcesofsupervision (Youngetal.,2014). However,thesedatasetsarestillrel-
toimproveperformanceonthetaskofvideoeventunder- atively small and limit achievable performance. Several
standing. However, asmentionedintheintroductionand methods have been proposed to create larger datasets au-
approachsection,methodsofleveragingnaturallanguage tomatically with Ordonez et al. (2011) as a notable early
descriptionsincomputervisionwellpredatetheuseofthis example. In the deep learning era, Mithun et al. (2018)
specific term, especially for image retrieval (Mori et al., demonstrated an additional set of (image, text) pairs col-
1999)andobjectclassification(Wangetal.,2009). Other lectedfromtheinternetcouldimproveretrievalperformance
earlyworkleveragedtags(butnotnaturallanguage)asso- andseveralnewautomaticallyconstructeddatasetssuchas
ciatedwithimagesforthetaskofsemanticsegmentation ConceptualCaptions(Sharmaetal.,2018),LAIT(Qietal.,
(Barnard et al., 2003). More recently, He & Peng (2017) 2020),andOCR-CC(Yangetal.,2020)havebeencreated.
andLiangetal.(2020)demonstratedusingnaturallanguage However,thesedatasetsstillusesignificantlymoreaggres-
descriptions and explanations to improve fine-grained vi- sivefilteringoraredesignedforaspecifictasksuchasOCR
sualclassificationofbirds. Othershaveinvestigatedhow andasaresultarestillmuchsmallerthanWITwithbetween
groundedlanguagecanbeusedtoimprovevisualrepresen- 1and10milliontrainingexamples.
tationsandclassifiersontheShapeWorlddataset(Kuhnle
ArelatedideatoCLIPisweblysupervisedlearning. This
&Copestake,2017;Andreasetal.,2017;Muetal.,2019).
lineofworkqueriesimagesearchenginestobuildimage
Finally,techniqueswhichcombinenaturallanguagewith
datasetsbyqueryingfortermsandusesthequeriesasthe
reinforcement learning environments (Narasimhan et al.,
labelsforthereturnedimages(Fergusetal.,2005). Classi-
2015)havedemonstratedexcitingemergentbehaviorssuch
fierstrainedontheselargebutnoisilylabeleddatasetscan
assystematicallyaccomplishingzero-shottasks(Hilletal.,
becompetitivewiththosetrainedonsmallercarefullyla-
2019).
beleddatasets. Theseimage-querypairsarealsooftenused
CLIP’spre-trainingtaskoptimizesfortext-imageretrieval. toimproveperformanceonstandarddatasetsasadditional
Thisareasofresearchdatesbacktothemid-90swiththe trainingdata(Chen&Gupta,2015). CLIPalsousessearch
previouslymentionedMorietal.(1999)asrepresentativeof queries as part of its dataset creation process. However
earlywork.Whileinitialeffortsfocusedprimarilyonpredic- CLIPonlyusesfulltextsequencesco-occuringwithimages
tiveobjectivesovertimeresearchshiftedtowardslearning assupervisionratherthanjustthequeries,whichareoften
jointmulti-modalembeddingspaceswithtechniqueslike onlyasinglewordorshortn-gram. Wealsorestrictthisstep
kernelCanonicalCorrelationAnalysisandvariousranking inCLIPtotextonlyqueryingforsub-stringmatcheswhile
objectives (Weston et al., 2010; Socher & Fei-Fei, 2010; most webly supervised work uses standard image search
Hodoshetal.,2013). Overtimeworkexploredmanycombi- engineswhichhavetheirowncomplexretrievalandfilter-
nationsoftrainingobjective,transfer,andmoreexpressive ing pipelines that often involve computer vision systems.
modelsandsteadilyimprovedperformance(Fromeetal., Ofthislineofwork,LearningEverythingaboutAnything:
2013;Socheretal.,2014;Karpathyetal.,2014;Kirosetal., Webly-SupervisedVisualConceptLearning(Divvalaetal.,
2014;Faghrietal.,2017). 2014)hasanotablysimilarambitionandgoalasCLIP.
Otherworkhasleveragednaturallanguagesupervisionfor Finally,CLIPisrelatedtoarecentburstofactivityonlearn-
domainsotherthanimages. Stroudetal.(2020)explores ingjointmodelsofvisionandlanguage(Luetal.,2019;Tan

27
LearningTransferableVisualModelsFromNaturalLanguageSupervision
| &Bansal,2019;Chenetal.,2019;Lietal.,2020b;Yuetal., |     |     |     |     |     | References |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- |
2020).Thislineofworkfocusesonrichlyconnectingvision
Abadi,M.,Barham,P.,Chen,J.,Chen,Z.,Davis,A.,Dean,
andlanguageinordertosolvecomplexdownstreamtasks
J.,Devin,M.,Ghemawat,S.,Irving,G.,Isard,M.,etal.
| such as visual | question | answering, |     | visual commonsense |     |     |     |     |     |     |     |
| -------------- | -------- | ---------- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- |
Tensorflow:Asystemforlarge-scalemachinelearning.In
| reasoning, | or multimodal |     | entailment. | These approaches |     |     |     |     |     |     |     |
| ---------- | ------------- | --- | ----------- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
12th{USENIX}symposiumonoperatingsystemsdesign
leverageimpressivelyengineeredmodelswhichcombine3
andimplementation({OSDI}16),pp.265–283,2016.
(ormore)pre-trainedsubsystems,typicallyanimagefeature
model, a region proposal / object detection model, and a Alayrac,J.-B.,Recasens,A.,Schneider,R.,Arandjelovic´,
pre-trainedmaskedlanguagemodelsuchasBERT.These
R.,Ramapuram,J.,DeFauw,J.,Smaira,L.,Dieleman,S.,
systemsarethenjointlyfine-tunedviavarioustrainingobjec-
|     |     |     |     |     |     | andZisserman,A. |     | Self-supervisedmultimodalversatile |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ---------------------------------- | --- | --- | --- |
tivesonimage-textpairsandappliedtotheaforementioned
|           |         |            |          |                 |     | networks. | arXivpreprintarXiv:2006.16228,2020. |     |     |     |     |
| --------- | ------- | ---------- | -------- | --------------- | --- | --------- | ----------------------------------- | --- | --- | --- | --- |
| tasks and | achieve | impressive | results. | CLIP is instead | fo- |           |                                     |     |     |     |     |
cused on learning visual models from scratch via natural Alcorn,M.A.,Li,Q.,Gong,Z.,Wang,C.,Mai,L.,Ku,W.-
languagesupervisionanddoesnotdenselyconnectthetwo
|                                  |     |     |     |                    |     | S.,andNguyen,A.                                 |     | Strike(with)apose: |     | Neuralnetworks |     |
| -------------------------------- | --- | --- | --- | ------------------ | --- | ----------------------------------------------- | --- | ------------------ | --- | -------------- | --- |
| domainswithajointattentionmodel. |     |     |     | Theonlyinteraction |     |                                                 |     |                    |     |                |     |
|                                  |     |     |     |                    |     | areeasilyfooledbystrangeposesoffamiliarobjects. |     |                    |     |                | In  |
inaCLIPmodelbetweentheimageandtextdomainisa ProceedingsoftheIEEEConferenceonComputerVision
| singledotproductinalearnedjointembeddingspace. |     |     |     |     | We  |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andPatternRecognition,pp.4845–4854,2019.
areexcitedtoseeCLIPhybridizedwiththislineofwork.
|              |     |     |     |     |     | Andreas,J.,Klein,D.,andLevine,S. |                                     |     |     | Learningwithlatent |     |
| ------------ | --- | --- | --- | --- | --- | -------------------------------- | ----------------------------------- | --- | --- | ------------------ | --- |
| 9.Conclusion |     |     |     |     |     | language.                        | arXivpreprintarXiv:1711.00482,2017. |     |     |                    |     |
Wehaveinvestigatedwhetheritispossibletotransferthe Assiri, Y. Stochastic optimization of plain convolutional
successoftask-agnosticweb-scalepre-traininginNLPto
|     |     |     |     |     |     | neural networks |     | with simple | methods. | arXiv | preprint |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ----------- | -------- | ----- | -------- |
another domain. We find that adopting this formula re- arXiv:2001.08856,2020.
sultsinsimilarbehaviorsemerginginthefieldofcomputer
|            |         |            |              |         |         | Bachman,P.,Hjelm,R.D.,andBuchwalter,W. |     |     |     |     | Learning |
| ---------- | ------- | ---------- | ------------ | ------- | ------- | -------------------------------------- | --- | --- | --- | --- | -------- |
| vision and | discuss | the social | implications | of this | line of |                                        |     |     |     |     |          |
research. Inordertooptimizetheirtrainingobjective,CLIP representationsbymaximizingmutualinformationacross
modelslearntoperformawidevarietyoftasksduringpre- views. In Advances in Neural Information Processing
training.Thistasklearningcanthenbeleveragedvianatural Systems,pp.15535–15545,2019.
| language | prompting | to enable | zero-shot | transfer to | many |     |     |     |     |     |     |
| -------- | --------- | --------- | --------- | ----------- | ---- | --- | --- | --- | --- | --- | --- |
Barbu,A.,Mayo,D.,Alverio,J.,Luo,W.,Wang,C.,Gut-
existingdatasets.Atsufficientscale,theperformanceofthis
approachcanbecompetitivewithtask-specificsupervised freund, D., Tenenbaum, J., and Katz, B. Objectnet: A
modelsalthoughthereisstillroomformuchimprovement. large-scale bias-controlled dataset for pushing the lim-
|     |     |     |     |     |     | itsofobjectrecognitionmodels. |     |     | InAdvancesinNeural |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | ------------------ | --- | --- |
ACKNOWLEDGMENTS InformationProcessingSystems,pp.9453–9463,2019.
We’dliketothankthemillionsofpeopleinvolvedincreating Barnard,K.,Duygulu,P.,Forsyth,D.,Freitas,N.d.,Blei,
| thedataCLIPistrainedon. |     |     | We’dalsoliketothankSusan |     |     |                     |     |     |                           |     |     |
| ----------------------- | --- | --- | ------------------------ | --- | --- | ------------------- | --- | --- | ------------------------- | --- | --- |
|                         |     |     |                          |     |     | D.M.,andJordan,M.I. |     |     | Matchingwordsandpictures. |     |     |
Zhangforherworkonimageconditionallanguagemodels Journalofmachinelearningresearch,3(Feb):1107–1135,
| whileatOpenAI,IshaanGulrajaniforcatchinganerrorin |     |     |     |     |     | 2003. |     |     |     |     |     |
| ------------------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
thepseudocode,andIreneSolaiman,MilesBrundage,and
GillianHadfieldfortheirthoughtfulfeedbackonthebroader Bechmann, A. and Bowker, G. C. Unsupervised by any
impacts section of the paper. We are also grateful to the othername: Hiddenlayersofknowledgeproductionin
AccelerationandSupercomputingteamsatOpenAIfortheir artificialintelligenceonsocialmedia.BigData&Society,
criticalworkonsoftwareandhardwareinfrastructurethis 6(1):205395171881956, January 2019. doi: 10.1177/
projectused. Finally,we’dalsoliketothankthedevelopers 2053951718819569. URL https://doi.org/10.
ofthemanysoftwarepackagesusedthroughoutthisproject 1177/2053951718819569.
including,butnotlimited,toNumpy(Harrisetal.,2020),
|                 |     |                |      |                |         | Bengio, Y., | Ducharme, | R., | Vincent, | P., and Jauvin, | C. A |
| --------------- | --- | -------------- | ---- | -------------- | ------- | ----------- | --------- | --- | -------- | --------------- | ---- |
| SciPy (Virtanen |     | et al., 2020), | ftfy | (Speer, 2019), | Tensor- |             |           |     |          |                 |      |
Flow (Abadi et al., 2016), PyTorch (Paszke et al., 2019), neuralprobabilisticlanguagemodel. Journalofmachine
pandas(pandasdevelopmentteam,2020),andscikit-learn learningresearch,3(Feb):1137–1155,2003.
(Pedregosaetal.,2011).
|     |     |     |     |     |     | Bhargava,S.andForsyth,D. |     |                  | Exposingandcorrectingthe |     |             |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | ---------------- | ------------------------ | --- | ----------- |
|     |     |     |     |     |     | gender bias              | in  | image captioning | datasets                 |     | and models. |
arXivpreprintarXiv:1912.00578,2019.

28
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Blei,D.M.,Ng,A.Y.,andJordan,M.I. Latentdirichlet Chen, X., Fan, H., Girshick, R., and He, K. Improved
allocation. JournalofmachineLearningresearch,3(Jan): baselines with momentum contrastive learning. arXiv
| 993–1022,2003. |     |     |     |     |     |     | preprintarXiv:2003.04297,2020d. |     |     |     |     |     |     |
| -------------- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | --- | --- |
Bolukbasi,T.,Chang,K.-W.,Zou,J.Y.,Saligrama,V.,and Chen,Y.-C.,Li,L.,Yu,L.,Kholy,A.E.,Ahmed,F.,Gan,Z.,
Kalai,A.T. Manistocomputerprogrammeraswoman Cheng,Y.,andLiu,J. Uniter: Learninguniversalimage-
istohomemaker? debiasingwordembeddings. Advances textrepresentations. arXivpreprintarXiv:1909.11740,
| inneuralinformationprocessingsystems,29:4349–4357, |     |     |     |     |     |     | 2019. |     |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
2016.
|     |     |     |     |     |     |     | Cheng,G.,Han,J.,andLu,X. |     |     | Remotesensingimagescene |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | ----------------------- | --- | --- | --- |
Bowker,G.C.andStar,S.L. Sortingthingsout: Classifica- classification: Benchmarkandstateoftheart. Proceed-
tionanditsconsequences. MITpress,2000. ingsoftheIEEE,105(10):1865–1883,2017.
Choi,D.,Shallue,C.J.,Nado,Z.,Lee,J.,Maddison,C.J.,
Brown,T.B.,Mann,B.,Ryder,N.,Subbiah,M.,Kaplan,
J.,Dhariwal,P.,Neelakantan,A.,Shyam,P.,Sastry,G., andDahl,G.E. Onempiricalcomparisonsofoptimiz-
|                                     |              |                                    |                          |     |     |      | ersfordeeplearning.                         |             |     | arXivpreprintarXiv:1910.05446, |     |          |            |
| ----------------------------------- | ------------ | ---------------------------------- | ------------------------ | --- | --- | ---- | ------------------------------------------- | ----------- | --- | ------------------------------ | --- | -------- | ---------- |
| Askell,A.,etal.                     |              | Languagemodelsarefew-shotlearners. |                          |     |     |      |                                             |             |     |                                |     |          |            |
| arXivpreprintarXiv:2005.14165,2020. |              |                                    |                          |     |     |      | 2019.                                       |             |     |                                |     |          |            |
|                                     |              |                                    |                          |     |     |      | Coates,                                     | A., Ng, A., | and | Lee, H.                        | An  | analysis | of single- |
| Browne,S.                           | DarkMatters: |                                    | SurveillanceofBlackness. |     |     | Duke |                                             |             |     |                                |     |          |            |
|                                     |              |                                    |                          |     |     |      | layernetworksinunsupervisedfeaturelearning. |             |     |                                |     |          | InPro-     |
UniversityPress,2015.
|     |     |     |     |     |     |     | ceedings | of the | fourteenth | international |     | conference | on  |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------ | ---------- | ------------- | --- | ---------- | --- |
BulentSariyildiz, M., Perez, J., andLarlus, D. Learning artificialintelligenceandstatistics,pp.215–223,2011.
| visual | representations |     | with caption | annotations. |     | arXiv |           |     |             |      |       |     |           |
| ------ | --------------- | --- | ------------ | ------------ | --- | ----- | --------- | --- | ----------- | ---- | ----- | --- | --------- |
|        |                 |     |              |              |     |       | Crawford, | K.  | The trouble | with | bias. |     | NIPS 2017 |
e-prints,pp.arXiv–2008,2020.
Keynote,2017.URLhttps://www.youtube.com/
watch?v=fMym_BKWQzk.
| Buolamwini, | J.  | and Gebru, | T.  | Gender | shades: | Intersec- |     |     |     |     |     |     |     |
| ----------- | --- | ---------- | --- | ------ | ------- | --------- | --- | --- | --- | --- | --- | --- | --- |
tionalaccuracydisparitiesincommercialgenderclassi-
Dai,A.M.andLe,Q.V.Semi-supervisedsequencelearning.
| fication. | InConferenceonfairness,accountabilityand |     |     |     |     |     |             |     |        |             |            |     |          |
| --------- | ---------------------------------------- | --- | --- | --- | --- | --- | ----------- | --- | ------ | ----------- | ---------- | --- | -------- |
|           |                                          |     |     |     |     |     | In Advances | in  | neural | information | processing |     | systems, |
transparency,pp.77–91,2018.
pp.3079–3087,2015.
| Carreira,J.,Noland,E.,Hillier,C.,andZisserman,A. |     |     |     |     |     | A   |          |             |     |           |     |        |          |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | --------- | --- | ------ | -------- |
|                                                  |     |     |     |     |     |     | D’Amour, | A., Heller, | K., | Moldovan, | D., | Adlam, | B., Ali- |
shortnoteonthekinetics-700humanactiondataset.arXiv
panahi,B.,Beutel,A.,Chen,C.,Deaton,J.,Eisenstein,
preprintarXiv:1907.06987,2019.
|           |          |     |        |         |          |           | J., Hoffman, | M.              | D., et | al. Underspecification |     |         | presents  |
| --------- | -------- | --- | ------ | ------- | -------- | --------- | ------------ | --------------- | ------ | ---------------------- | --- | ------- | --------- |
|           |          |     |        |         |          |           | challenges   | for credibility |        | in modern              |     | machine | learning. |
| Chen, M., | Radford, | A., | Child, | R., Wu, | J., Jun, | H., Luan, |              |                 |        |                        |     |         |           |
arXivpreprintarXiv:2011.03395,2020.
| D.,andSutskever,I. |     | Generativepretrainingfrompixels. |     |            |           |     |           |           |         |     |            |     |             |
| ------------------ | --- | -------------------------------- | --- | ---------- | --------- | --- | --------- | --------- | ------- | --- | ---------- | --- | ----------- |
| In International   |     | Conference                       |     | on Machine | Learning, | pp. |           |           |         |     |            |     |             |
|                    |     |                                  |     |            |           |     | Deng, J., | Dong, W., | Socher, | R., | Li, L.-J., | Li, | K., andFei- |
1691–1703.PMLR,2020a.
|           |         |        |     |               |     |          | Fei, L.   | ImageNet:      | A   | Large-Scale | Hierarchical |     | Image |
| --------- | ------- | ------ | --- | ------------- | --- | -------- | --------- | -------------- | --- | ----------- | ------------ | --- | ----- |
|           |         |        |     |               |     |          | Database. | InCVPR09,2009. |     |             |              |     |       |
| Chen, T., | Xu, B., | Zhang, | C., | and Guestrin, | C.  | Training |           |                |     |             |              |     |       |
deep nets with sublinear memory cost. arXiv preprint Deng, J., Berg, A. C., Satheesh, S., Su, H., Khosla, A.,
| arXiv:1604.06174,2016. |     |     |     |     |     |     |               |     |                  |     | URLhttp://www. |     |     |
| ---------------------- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---------------- | --- | -------------- | --- | --- |
|                        |     |     |     |     |     |     | andFei-Fei,L. |     | Ilsvrc2012,2012. |     |                |     |     |
image-net.org/challenges/LSVRC/2012/.
| Chen, T., | Kornblith, | S., | Norouzi, | M., and | Hinton, | G. A |     |     |     |     |     |     |     |
| --------- | ---------- | --- | -------- | ------- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- |
simpleframeworkforcontrastivelearningofvisualrep- Desai, K. and Johnson, J. Virtex: Learning visual rep-
| resentations. | arXivpreprintarXiv:2002.05709,2020b. |     |     |     |     |     |              |      |         |              |     |       |          |
| ------------- | ------------------------------------ | --- | --- | --- | --- | --- | ------------ | ---- | ------- | ------------ | --- | ----- | -------- |
|               |                                      |     |     |     |     |     | resentations | from | textual | annotations. |     | arXiv | preprint |
arXiv:2006.06666,2020.
| Chen, T., | Kornblith, | S., | Swersky, | K., Norouzi, |     | M., and |     |     |     |     |     |     |     |
| --------- | ---------- | --- | -------- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
Hinton,G. Bigself-supervisedmodelsarestrongsemi- Devlin,J.,Chang,M.-W.,Lee,K.,andToutanova,K. Bert:
| supervised | learners. |     | arXiv preprint | arXiv:2006.10029, |     |     |                     |     |                    |                                |              |     |          |
| ---------- | --------- | --- | -------------- | ----------------- | --- | --- | ------------------- | --- | ------------------ | ------------------------------ | ------------ | --- | -------- |
|            |           |     |                |                   |     |     | Pre-training        | of  | deep bidirectional |                                | transformers |     | for lan- |
| 2020c.     |           |     |                |                   |     |     | guageunderstanding. |     |                    | arXivpreprintarXiv:1810.04805, |              |     |          |
2018.
| Chen, X. | and Gupta, | A.  | Webly | supervised | learning | of  |     |     |     |     |     |     |     |
| -------- | ---------- | --- | ----- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
convolutional networks. In Proceedings of the IEEE Dhariwal,P.,Jun,H.,Payne,C.,Kim,J.W.,Radford,A.,
InternationalConferenceonComputerVision,pp.1431– andSutskever,I. Jukebox: Agenerativemodelformusic.
| 1439,2015. |     |     |     |     |     |     | arXivpreprintarXiv:2005.00341,2020. |     |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- | --- |

29
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Divvala, S. K., Farhadi, A., and Guestrin, C. Learning biasedtowardstexture;increasingshapebiasimprovesac-
everythingaboutanything: Webly-supervisedvisualcon- curacyandrobustness. arXivpreprintarXiv:1811.12231,
| cept learning. | In Proceedings | of the | IEEE Conference | 2018. |     |     |     |     |     |     |
| -------------- | -------------- | ------ | --------------- | ----- | --- | --- | --- | --- | --- | --- |
onComputerVisionandPatternRecognition,pp.3270–
3277,2014. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R.,
|                     |     |                            |     | Brendel,     | W., Bethge, |      | M., and          | Wichmann, | F.    | A. Short- |
| ------------------- | --- | -------------------------- | --- | ------------ | ----------- | ---- | ---------------- | --------- | ----- | --------- |
|                     |     |                            |     | cut learning | in          | deep | neural networks. |           | arXiv | preprint  |
| Dodge,S.andKaram,L. |     | Astudyandcomparisonofhuman |     |              |             |      |                  |           |       |           |
anddeeplearningrecognitionperformanceundervisual arXiv:2004.07780,2020.
| distortions. | In 2017 26th | international | conference on |     |     |     |     |     |     |     |
| ------------ | ------------ | ------------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
Gomez,L.,Patel,Y.,Rusin˜ol,M.,Karatzas,D.,andJawahar,
computercommunicationandnetworks(ICCCN),pp.1–
|     |     |     |     | C. Self-supervised |     | learning | of  | visual | features | through |
| --- | --- | --- | --- | ------------------ | --- | -------- | --- | ------ | -------- | ------- |
7.IEEE,2017.
|     |     |     |     | embeddingimagesintotexttopicspaces. |     |     |     |     | InProceedings |     |
| --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | ------------- | --- |
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, oftheIEEEConferenceonComputerVisionandPattern
Recognition,pp.4230–4239,2017.
D.,Zhai,X.,Unterthiner,T.,Dehghani,M.,Minderer,M.,
| Heigold,G.,Gelly,S.,etal. |     | Animageisworth16x16 |     |             |        |         |         |          |     |          |
| ------------------------- | --- | ------------------- | --- | ----------- | ------ | ------- | ------- | -------- | --- | -------- |
|                           |     |                     |     | Goodfellow, | I. J., | Shlens, | J., and | Szegedy, | C.  | Explain- |
words:Transformersforimagerecognitionatscale.arXiv
|     |     |     |     | ingandharnessingadversarialexamples. |     |     |     |     | arXivpreprint |     |
| --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- | ------------- | --- |
preprintarXiv:2010.11929,2020.
arXiv:1412.6572,2014.
| Elhoseiny,M.,Saleh,B.,andElgammal,A. |     |     | Writeaclassi- |             |        |        |              |     |                |     |
| ------------------------------------ | --- | --- | ------------- | ----------- | ------ | ------ | ------------ | --- | -------------- | --- |
|                                      |     |     |               | Goodfellow, | I. J., | Erhan, | D., Carrier, | P.  | L., Courville, | A., |
fier: Zero-shotlearningusingpurelytextualdescriptions.
Mirza,M.,Hamner,B.,Cukierski,W.,Tang,Y.,Thaler,
InProceedingsoftheIEEEInternationalConferenceon
|     |     |     |     | D.,Lee,D.-H.,etal. |     | Challengesinrepresentationlearn- |     |     |     |     |
| --- | --- | --- | --- | ------------------ | --- | -------------------------------- | --- | --- | --- | --- |
ComputerVision,pp.2584–2591,2013.
|     |     |     |     | ing: Areportonthreemachinelearningcontests. |     |     |     |     |     | Neural |
| --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | ------ |
Networks,64:59–63,2015.
Faghri,F.,Fleet,D.J.,Kiros,J.R.,andFidler,S.Vse++:Im-
provingvisual-semanticembeddingswithhardnegatives.
|     |     |     |     | Google. | Google | cloud | api: Celebrity | recognition. |     | URL |
| --- | --- | --- | --- | ------- | ------ | ----- | -------------- | ------------ | --- | --- |
arXivpreprintarXiv:1707.05612,2017.
https://cloud.google.com/vision/docs/
celebrity-recognition.
| Fergus,R.,Fei-Fei,L.,Perona,P.,andZisserman,A. |     |     | Learn- |     |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
ing object categories from google’s image search. In Griewank,A.andWalther,A. Algorithm799: revolve: an
TenthIEEEInternationalConferenceonComputerVision
|     |     |     |     | implementation |     | of checkpointing |     | for | the reverse | or ad- |
| --- | --- | --- | --- | -------------- | --- | ---------------- | --- | --- | ----------- | ------ |
(ICCV’05) Volume 1, volume 2, pp. 1816–1823. IEEE, jointmodeofcomputationaldifferentiation. ACMTrans-
2005.
actionsonMathematicalSoftware(TOMS),26(1):19–45,
2000.
Frome,A.,Corrado,G.S.,Shlens,J.,Bengio,S.,Dean,J.,
Ranzato, M., and Mikolov, T. Devise: A deep visual- Grill, J.-B., Strub, F., Altche´, F., Tallec, C., Richemond,
semanticembeddingmodel. InAdvancesinneuralinfor- P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo,
mationprocessingsystems,pp.2121–2129,2013.
|     |     |     |     | Z.D.,Azar,M.G.,etal.                  |     |     | Bootstrapyourownlatent: |     |               | A   |
| --- | --- | --- | --- | ------------------------------------- | --- | --- | ----------------------- | --- | ------------- | --- |
|     |     |     |     | newapproachtoself-supervisedlearning. |     |     |                         |     | arXivpreprint |     |
Gan,Z.,Chen,Y.-C.,Li,L.,Zhu,C.,Cheng,Y.,andLiu,J.
arXiv:2006.07733,2020.
Large-scaleadversarialtrainingforvision-and-language
representationlearning.arXivpreprintarXiv:2006.06195,
|       |     |     |     | Ha, D.,                        | Dai, A., | and Le, | Q. V. | Hypernetworks. |     | arXiv |
| ----- | --- | --- | --- | ------------------------------ | -------- | ------- | ----- | -------------- | --- | ----- |
| 2020. |     |     |     | preprintarXiv:1609.09106,2016. |          |         |       |                |     |       |
Gao,T.,Fisch,A.,andChen,D. Makingpre-trainedlan- Hancock,B.,Bringmann,M.,Varma,P.,Liang,P.,Wang,
guage models better few-shot learners. arXiv preprint S.,andRe´,C. Trainingclassifierswithnaturallanguage
arXiv:2012.15723,2020. explanations. InProceedingsoftheconference.Associ-
|             |           |     |              | ation | for Computational |     | Linguistics. |     | Meeting, | volume |
| ----------- | --------- | --- | ------------ | ----- | ----------------- | --- | ------------ | --- | -------- | ------ |
| Garvie, C., | May 2019. | URL | https://www. |       |                   |     |              |     |          |        |
2018,pp.1884.NIHPublicAccess,2018.
flawedfacedata.com/.
|     |     |     |     | Hancock, | B., Bordes, | A., | Mazare, | P.-E., | and | Weston, J. |
| --- | --- | --- | --- | -------- | ----------- | --- | ------- | ------ | --- | ---------- |
Geiger, A., Lenz, P., and Urtasun, R. Are we ready for Learningfromdialogueafterdeployment: Feedyourself,
autonomousdriving? thekittivisionbenchmarksuite. In chatbot! arXivpreprintarXiv:1901.05415,2019.
ConferenceonComputerVisionandPatternRecognition
Harris,C.R.,Millman,K.J.,vanderWalt,S.J.,Gommers,
(CVPR),2012.
R.,Virtanen,P.,Cournapeau,D.,Wieser,E.,Taylor,J.,
Geirhos,R.,Rubisch,P.,Michaelis,C.,Bethge,M.,Wich- Berg,S.,Smith,N.J.,Kern,R.,Picus,M.,Hoyer,S.,van
mann,F.A.,andBrendel,W. Imagenet-trainedcnnsare Kerkwijk,M.H.,Brett,M.,Haldane,A.,Ferna´ndezdel

30
LearningTransferableVisualModelsFromNaturalLanguageSupervision
R´ıo, J., Wiebe, M., Peterson, P., Ge´rard-Marchant, P., Hendrycks,D.andGimpel,K. Gaussianerrorlinearunits
Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., (gelus). arXivpreprintarXiv:1606.08415,2016.
| Gohlke,    | C., and Oliphant,        | T.  | E. Array | programming |          |            |                             |       |             |     |               |         |
| ---------- | ------------------------ | --- | -------- | ----------- | -------- | ---------- | --------------------------- | ----- | ----------- | --- | ------------- | ------- |
|            |                          |     |          |             |          | Hendrycks, | D.,                         | Zhao, | K., Basart, | S., | Steinhardt,   | J., and |
| withNumPy. | Nature,585:357–362,2020. |     |          | doi:        | 10.1038/ |            |                             |       |             |     |               |         |
|            |                          |     |          |             |          | Song,D.    | Naturaladversarialexamples. |       |             |     | arXivpreprint |         |
s41586-020-2649-2.
arXiv:1907.07174,2019.
| Hays,J.andEfros,A.A. |     | Im2gps: | estimatinggeographic |     |     |     |     |     |     |     |     |     |
| -------------------- | --- | ------- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Hendrycks,D.,Basart,S.,Mu,N.,Kadavath,S.,Wang,F.,
| information | from a | single image. | In  | 2008 ieee | confer- |          |     |        |          |               |     |              |
| ----------- | ------ | ------------- | --- | --------- | ------- | -------- | --- | ------ | -------- | ------------- | --- | ------------ |
|             |        |               |     |           |         | Dorundo, | E., | Desai, | R., Zhu, | T., Parajuli, |     | S., Guo, M., |
enceoncomputervisionandpatternrecognition,pp.1–8.
|     |     |     |     |     |     | et al. | The many | faces | of  | robustness: | A   | critical analy- |
| --- | --- | --- | --- | --- | --- | ------ | -------- | ----- | --- | ----------- | --- | --------------- |
IEEE,2008.
|     |     |     |     |     |     | sisofout-of-distributiongeneralization. |     |     |     |     | arXivpreprint |     |
| --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | --- | --- | ------------- | --- |
arXiv:2006.16241,2020a.
| He, K., Zhang,  | X., Ren,                           | S., and | Sun, | J. Delving | deep |     |     |     |     |     |     |     |
| --------------- | ---------------------------------- | ------- | ---- | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- |
| intorectifiers: | Surpassinghuman-levelperformanceon |         |      |            |      |     |     |     |     |     |     |     |
Hendrycks,D.,Liu,X.,Wallace,E.,Dziedzic,A.,Krishnan,
| imagenetclassification. |     | InProceedingsoftheIEEEinter- |     |     |     |               |     |                                      |     |     |     |     |
| ----------------------- | --- | ---------------------------- | --- | --- | --- | ------------- | --- | ------------------------------------ | --- | --- | --- | --- |
|                         |     |                              |     |     |     | R.,andSong,D. |     | Pretrainedtransformersimproveout-of- |     |     |     |     |
nationalconferenceoncomputervision,pp.1026–1034,
|     |     |     |     |     |     | distributionrobustness. |     |     | arXivpreprintarXiv:2004.06100, |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ------------------------------ | --- | --- | --- |
2015.
2020b.
He,K.,Zhang,X.,Ren,S.,andSun,J. Deepresiduallearn- Hestness,J.,Narang,S.,Ardalani,N.,Diamos,G.,Jun,H.,
| ingforimagerecognition. |     | InProceedingsoftheIEEE |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Kianinejad,H.,Patwary,M.,Ali,M.,Yang,Y.,andZhou,
conferenceoncomputervisionandpatternrecognition,
Y.Deeplearningscalingispredictable,empirically.arXiv
| pp.770–778,2016a. |     |     |     |     |     | preprintarXiv:1712.00409,2017. |     |     |     |     |     |     |
| ----------------- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- |
He,K.,Zhang,X.,Ren,S.,andSun,J. Deepresiduallearn- Hill,F.,Lampinen,A.,Schneider,R.,Clark,S.,Botvinick,
| ingforimagerecognition. |     | InProceedingsoftheIEEE |     |     |     |                 |     |       |             |     |                  |     |
| ----------------------- | --- | ---------------------- | --- | --- | --- | --------------- | --- | ----- | ----------- | --- | ---------------- | --- |
|                         |     |                        |     |     |     | M., McClelland, |     | J.L., | andSantoro, |     | A. Environmental |     |
conferenceoncomputervisionandpatternrecognition, driversofsystematicityandgeneralizationinasituated
pp.770–778,2016b. agent. InInternationalConferenceonLearningRepre-
sentations,2019.
| He, K., Fan, | H., Wu, | Y., Xie, S., | and | Girshick, | R. Mo- |     |     |     |     |     |     |     |
| ------------ | ------- | ------------ | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- |
mentumcontrastforunsupervisedvisualrepresentation Hodosh,M.,Young,P.,andHockenmaier,J.Framingimage
learning. InProceedingsoftheIEEE/CVFConference descriptionasarankingtask:Data,modelsandevaluation
onComputerVisionandPatternRecognition,pp.9729– metrics. JournalofArtificialIntelligenceResearch,47:
| 9738,2020. |     |     |     |     |     | 853–899,2013. |     |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
He,T.,Zhang,Z.,Zhang,H.,Zhang,Z.,Xie,J.,andLi,M. HongsuckSeo,P.,Weyand,T.,Sim,J.,andHan,B. Cplanet:
Bagoftricksforimageclassificationwithconvolutional Enhancingimagegeolocalizationbycombinatorialparti-
|                 |                                  |     |     |     |     | tioningofmaps. |     | InProceedingsoftheEuropeanConfer- |     |     |     |     |
| --------------- | -------------------------------- | --- | --- | --- | --- | -------------- | --- | --------------------------------- | --- | --- | --- | --- |
| neuralnetworks. | InProceedingsoftheIEEEConference |     |     |     |     |                |     |                                   |     |     |     |     |
onComputerVisionandPatternRecognition,pp.558– enceonComputerVision(ECCV),pp.536–551,2018.
567,2019.
|                 |                                    |     |     |     |     | Howard,     | J. and | Ruder,   | S.              | Universal | language | model    |
| --------------- | ---------------------------------- | --- | --- | --- | --- | ----------- | ------ | -------- | --------------- | --------- | -------- | -------- |
|                 |                                    |     |     |     |     | fine-tuning |        | for text | classification. |           | arXiv    | preprint |
| He,X.andPeng,Y. | Fine-grainedimageclassificationvia |     |     |     |     |             |        |          |                 |           |          |          |
arXiv:1801.06146,2018.
| combiningvisionandlanguage. |     |     | InProceedingsofthe |     |     |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
IEEEConferenceonComputerVisionandPatternRecog-
|     |     |     |     |     |     | Ilyas, A., | Santurkar, | S., | Tsipras, | D., | Engstrom, | L., Tran, |
| --- | --- | --- | --- | --- | --- | ---------- | ---------- | --- | -------- | --- | --------- | --------- |
nition,pp.5994–6002,2017.
|                                            |     |     |     |     |          | B., and | Madry,        | A. Adversarial |          | examples |        | are not bugs, |
| ------------------------------------------ | --- | --- | --- | --- | -------- | ------- | ------------- | -------------- | -------- | -------- | ------ | ------------- |
|                                            |     |     |     |     |          | they    | are features. | In             | Advances | in       | Neural | Information   |
| Helber,P.,Bischke,B.,Dengel,A.,andBorth,D. |     |     |     |     | Eurosat: |         |               |                |          |          |        |               |
ProcessingSystems,pp.125–136,2019.
| A novel | dataset and | deep learning | benchmark |     | for land |     |     |     |     |     |     |     |
| ------- | ----------- | ------------- | --------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
use and land cover classification. IEEE Journal of Se- Ioffe,S.andSzegedy,C. Batchnormalization:Accelerating
lectedTopicsinAppliedEarthObservationsandRemote
deepnetworktrainingbyreducinginternalcovariateshift.
Sensing,12(7):2217–2226,2019. arXivpreprintarXiv:1502.03167,2015.
Henaff,O.Data-efficientimagerecognitionwithcontrastive Jaderberg,M.,Simonyan,K.,Vedaldi,A.,andZisserman,
predictivecoding. InInternationalConferenceonMa- A. Deepstructuredoutputlearningforunconstrainedtext
chineLearning,pp.4182–4192.PMLR,2020.
|     |     |     |     |     |     | recognition. |     | arXivpreprintarXiv:1412.5903,2014. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ---------------------------------- | --- | --- | --- | --- |
Hendrycks, D. and Dietterich, T. Benchmarking neural Jaderberg,M.,Simonyan,K.,Zisserman,A.,etal. Spatial
networkrobustnesstocommoncorruptionsandperturba- transformernetworks. Advancesinneuralinformation
tions. arXivpreprintarXiv:1903.12261,2019. processingsystems,28:2017–2025,2015.

31
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Johnson,J.,Hariharan,B.,vanderMaaten,L.,Fei-Fei,L., Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K.,
LawrenceZitnick, C., andGirshick, R. Clevr: Adiag- Kravitz,J.,Chen,S.,Kalantidis,Y.,Li,L.-J.,Shamma,
nosticdatasetforcompositionallanguageandelementary D.A.,etal. Visualgenome: Connectinglanguageand
visual reasoning. In Proceedings of the IEEE Confer- visionusingcrowdsourceddenseimageannotations. In-
ence on Computer Vision and Pattern Recognition, pp. ternational journal of computer vision, 123(1):32–73,
| 2901–2910,2017. |     |     |     |     | 2017. |     |     |     |
| --------------- | --- | --- | --- | --- | ----- | --- | --- | --- |
Joulin,A.,VanDerMaaten,L.,Jabri,A.,andVasilache,N. Krizhevsky,A.,Sutskever,I.,andHinton,G.E. Imagenet
Learning visual features from large weakly supervised classification with deep convolutional neural networks.
data. InEuropeanConferenceonComputerVision,pp. In Advances in neural information processing systems,
| 67–84.Springer,2016. |     |     |     |     | pp.1097–1105,2012. |     |     |     |
| -------------------- | --- | --- | --- | --- | ------------------ | --- | --- | --- |
Kalfaoglu,M.,Kalkan,S.,andAlatan,A.A. Latetemporal Kuhnle, A. and Copestake, A. Shapeworld-a new test
modeling in 3d cnn architectures with bert for action methodology for multimodal language understanding.
recognition. arXivpreprintarXiv:2008.01232,2020. arXivpreprintarXiv:1704.04517,2017.
Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Ka¨rkka¨inen,K.andJoo,J. Fairface: Faceattributedataset
forbalancedrace,gender,andage,2019.
Chess,B.,Child,R.,Gray,S.,Radford,A.,Wu,J.,and
| Amodei, | D. Scaling | laws for | neural language | models. |     |     |     |     |
| ------- | ---------- | -------- | --------------- | ------- | --- | --- | --- | --- |
Lake,B.M.,Ullman,T.D.,Tenenbaum,J.B.,andGersh-
arXivpreprintarXiv:2001.08361,2020.
|     |     |     |     |     | man, | S.J. Buildingmachinesthatlearnandthinklike |     |     |
| --- | --- | --- | --- | --- | ---- | ------------------------------------------ | --- | --- |
people,2016.
| Karpathy,A.,Joulin,A.,andFei-Fei,L.F. |                   |             | Deepfragment |          |                                          |     |     |          |
| ------------------------------------- | ----------------- | ----------- | ------------ | -------- | ---------------------------------------- | --- | --- | -------- |
| embeddings                            | for bidirectional | image       | sentence     | mapping. |                                          |     |     |          |
|                                       |                   |             |              |          | Lampert,C.H.,Nickisch,H.,andHarmeling,S. |     |     | Learning |
| In Advances                           | in neural         | information | processing   | systems, |                                          |     |     |          |
todetectunseenobjectclassesbybetween-classattribute
pp.1889–1897,2014.
|     |     |     |     |     | transfer. | In2009IEEEConferenceonComputerVision |     |     |
| --- | --- | --- | --- | --- | --------- | ------------------------------------ | --- | --- |
Keyes,O. Themisgenderingmachines: Trans/hciimplica- andPatternRecognition,pp.951–958.IEEE,2009.
Proceedingsofthe
tionsofautomaticgenderrecognition.
Larochelle,H.,Erhan,D.,andBengio,Y.Zero-datalearning
ACMonHuman-ComputerInteraction,2(CSCW):1–22,
|     |     |     |     |     | ofnewtasks. | 2008. |     |     |
| --- | --- | --- | --- | --- | ----------- | ----- | --- | --- |
2018.
|     |     |     |     |     | Le,Q.andMikolov,T. |     | Distributedrepresentationsofsen- |     |
| --- | --- | --- | --- | --- | ------------------ | --- | -------------------------------- | --- |
Kiela,D.,Firooz,H.,Mohan,A.,Goswami,V.,Singh,A.,
|           |                     |     |                |       | tences | and documents. | In International | conference on |
| --------- | ------------------- | --- | -------------- | ----- | ------ | -------------- | ---------------- | ------------- |
| Ringshia, | P., and Testuggine, |     | D. The hateful | memes |        |                |                  |               |
machinelearning,pp.1188–1196,2014.
| challenge: | Detectinghatespeechinmultimodalmemes. |     |     |     |     |     |     |     |
| ---------- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
arXivpreprintarXiv:2005.04790,2020. LeCun, Y. The mnist database of handwritten digits.
http://yann.lecun.com/exdb/mnist/.
| Kingma,D.P.andBa,J. |                                    | Adam: | Amethodforstochastic |     |            |               |            |                     |
| ------------------- | ---------------------------------- | ----- | -------------------- | --- | ---------- | ------------- | ---------- | ------------------- |
| optimization.       | arXivpreprintarXiv:1412.6980,2014. |       |                      |     |            |               |            |                     |
|                     |                                    |       |                      |     | Lee, D.-H. | Pseudo-label: | The simple | and efficient semi- |
supervisedlearningmethodfordeepneuralnetworks.
| Kiros, R., | Salakhutdinov, | R., and | Zemel, R. | S. Unifying |     |     |     |     |
| ---------- | -------------- | ------- | --------- | ----------- | --- | --- | --- | --- |
visual-semanticembeddingswithmultimodalneurallan- Lei Ba, J., Swersky, K., Fidler, S., et al. Predicting deep
| guagemodels. | arXivpreprintarXiv:1411.2539,2014. |     |     |     |               |                                     |                 |               |
| ------------ | ---------------------------------- | --- | --- | --- | ------------- | ----------------------------------- | --------------- | ------------- |
|              |                                    |     |     |     | zero-shot     | convolutional                       | neural networks | using textual |
|              |                                    |     |     |     | descriptions. | InProceedingsoftheIEEEInternational |                 |               |
Kiros,R.,Zhu,Y.,Salakhutdinov,R.R.,Zemel,R.,Urtasun, ConferenceonComputerVision,pp.4247–4255,2015.
| R., Torralba, | A., and | Fidler, S. | Skip-thought | vectors. |     |     |     |     |
| ------------- | ------- | ---------- | ------------ | -------- | --- | --- | --- | --- |
Advancesinneuralinformationprocessingsystems,28: Li,A.,Jabri,A.,Joulin,A.,andvanderMaaten,L.Learning
3294–3302,2015. visual n-grams from web data. In Proceedings of the
IEEEInternationalConferenceonComputerVision,pp.
| Kolesnikov, | A., Beyer, | L., Zhai, X., | Puigcerver, | J., Yung, |     |     |     |     |
| ----------- | ---------- | ------------- | ----------- | --------- | --- | --- | --- | --- |
4183–4192,2017.
| J., Gelly, | S., and Houlsby, | N.  | Large scale | learning of |     |     |     |     |
| ---------- | ---------------- | --- | ----------- | ----------- | --- | --- | --- | --- |
generalvisualrepresentationsfortransfer. arXivpreprint Li, G., Duan, N., Fang, Y., Gong, M., and Jiang, D.
arXiv:1912.11370,2019. Unicoder-vl:Auniversalencoderforvisionandlanguage
|                                   |     |     |                  |     | bycross-modalpre-training. |     | 2020a. |     |
| --------------------------------- | --- | --- | ---------------- | --- | -------------------------- | --- | ------ | --- |
| Kornblith,S.,Shlens,J.,andLe,Q.V. |     |     | Dobetterimagenet |     |                            |     |        |     |
models transfer better? In Proceedings of the IEEE Li,J.,Miller,A.H.,Chopra,S.,Ranzato,M.,andWeston,J.
conferenceoncomputervisionandpatternrecognition, Learningthroughdialogueinteractionsbyaskingques-
pp.2661–2671,2019. tions. arXivpreprintarXiv:1612.04936,2016.

32
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Li,X.,Yin,X.,Li,C.,Hu,X.,Zhang,P.,Zhang,L.,Wang, ProceedingsoftheEuropeanConferenceonComputer
L., Hu, H., Dong, L., Wei, F., et al. Oscar: Object- Vision(ECCV),pp.181–196,2018.
semanticsalignedpre-trainingforvision-languagetasks.
|     |     |     |     |     |     | McCann, | B., Bradbury, | J., | Xiong, | C., and Socher, | R.  |
| --- | --- | --- | --- | --- | --- | ------- | ------------- | --- | ------ | --------------- | --- |
arXivpreprintarXiv:2004.06165,2020b.
|     |     |     |     |     |     | Learnedintranslation: |     | Contextualizedwordvectors. |     |     | In  |
| --- | --- | --- | --- | --- | --- | --------------------- | --- | -------------------------- | --- | --- | --- |
Liang,W.,Zou,J.,andYu,Z. Alice: Activelearningwith Advancesinneuralinformationprocessingsystems,pp.
| contrastivenaturallanguageexplanations. |     |     |     |     | arXivpreprint | 6294–6305,2017. |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | ------------- | --------------- | --- | --- | --- | --- | --- |
arXiv:2009.10259,2020.
|     |     |     |     |     |     | McCann,B.,Keskar,N.S.,Xiong,C.,andSocher,R. |     |     |     |     | The |
| --- | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- |
Lin,T.-Y.,Maire,M.,Belongie,S.,Hays,J.,Perona,P.,Ra- naturallanguagedecathlon: Multitasklearningasques-
manan,D.,Dolla´r,P.,andZitnick,C.L. Microsoftcoco: tionanswering. arXivpreprintarXiv:1806.08730,2018.
| Commonobjectsincontext. |     |     | InEuropeanconferenceon |     |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Micikevicius,P.,Narang,S.,Alben,J.,Diamos,G.,Elsen,
computervision,pp.740–755.Springer,2014.
E.,Garcia,D.,Ginsburg,B.,Houston,M.,Kuchaiev,O.,
Linzen, T. How can we accelerate progress towards Venkatesh, G., et al. Mixed precision training. arXiv
human-like linguistic generalization? arXiv preprint preprintarXiv:1710.03740,2017.
arXiv:2005.00955,2020.
Miech,A.,Zhukov,D.,Alayrac,J.-B.,Tapaswi,M.,Laptev,
| Lippe, P., | Holla, | N., Chandra, | S., Rajamanickam, |     | S., An- |                |            |     |                        |     |     |
| ---------- | ------ | ------------ | ----------------- | --- | ------- | -------------- | ---------- | --- | ---------------------- | --- | --- |
|            |        |              |                   |     |         | I.,andSivic,J. | Howto100m: |     | Learningatext-videoem- |     |     |
toniou,G.,Shutova,E.,andYannakoudakis,H. Amul- beddingbywatchinghundredmillionnarratedvideoclips.
timodalframeworkforthedetectionofhatefulmemes.
InProceedingsoftheIEEEinternationalconferenceon
arXivpreprintarXiv:2012.12871,2020.
computervision,pp.2630–2640,2019.
| Liu, P. | J., Saleh, | M., | Pot, E., | Goodrich, | B., Sepa- |     |     |     |     |     |     |
| ------- | ---------- | --- | -------- | --------- | --------- | --- | --- | --- | --- | --- | --- |
Miech,A.,Alayrac,J.-B.,Laptev,I.,Sivic,J.,andZisser-
| ssi, R., | Kaiser, | L., | and Shazeer, | N.  | Generating |        |                                              |     |     |     |     |
| -------- | ------- | --- | ------------ | --- | ---------- | ------ | -------------------------------------------- | --- | --- | --- | --- |
|          |         |     |              |     |            | man,A. | Rareact: Avideodatasetofunusualinteractions. |     |     |     |     |
wikipediabysummarizinglongsequences.arXivpreprint
arXivpreprintarXiv:2008.01018,2020a.
arXiv:1801.10198,2018.
Miech,A.,Alayrac,J.-B.,Smaira,L.,Laptev,I.,Sivic,J.,
| Locatello, | F., Bauer, | S., | Lucic, M., | Ra¨tsch, | G., Gelly, S., |     |     |     |     |     |     |
| ---------- | ---------- | --- | ---------- | -------- | -------------- | --- | --- | --- | --- | --- | --- |
andZisserman,A.End-to-endlearningofvisualrepresen-
| Scho¨lkopf,  | B., | and Bachem, | O.              | A sober | look at the     |                                          |     |     |     |            |     |
| ------------ | --- | ----------- | --------------- | ------- | --------------- | ---------------------------------------- | --- | --- | --- | ---------- | --- |
|              |     |             |                 |         |                 | tationsfromuncuratedinstructionalvideos. |     |     |     | InProceed- |     |
| unsupervised |     | learning    | of disentangled |         | representations |                                          |     |     |     |            |     |
ingsoftheIEEE/CVFConferenceonComputerVision
| andtheirevaluation. |     | arXivpreprintarXiv:2010.14766, |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andPatternRecognition,pp.9879–9889,2020b.
2020.
|             |         |             |           |            |                | Mikolov,                  | T., Sutskever,                              | I., | Chen, K.,                | Corrado, G. S., | and |
| ----------- | ------- | ----------- | --------- | ---------- | -------------- | ------------------------- | ------------------------------------------- | --- | ------------------------ | --------------- | --- |
| Loshchilov, | I.      | and Hutter, | F. Sgdr:  | Stochastic | gra-           |                           |                                             |     |                          |                 |     |
|             |         |             |           |            |                | Dean,J.                   | Distributedrepresentationsofwordsandphrases |     |                          |                 |     |
| dient       | descent | with warm   | restarts. |            | arXiv preprint |                           |                                             |     |                          |                 |     |
|             |         |             |           |            |                | andtheircompositionality. |                                             |     | Advancesinneuralinforma- |                 |     |
arXiv:1608.03983,2016.
tionprocessingsystems,26:3111–3119,2013.
| Loshchilov,I.andHutter,F. |                                     |     | Decoupledweightdecayregu- |     |     |                                            |     |     |     |           |     |
| ------------------------- | ----------------------------------- | --- | ------------------------- | --- | --- | ------------------------------------------ | --- | --- | --- | --------- | --- |
|                           |                                     |     |                           |     |     | Miller,J.,Krauth,K.,Recht,B.,andSchmidt,L. |     |     |     | Theeffect |     |
| larization.               | arXivpreprintarXiv:1711.05101,2017. |     |                           |     |     |                                            |     |     |     |           |     |
ofnaturaldistributionshiftonquestionansweringmodels.
Lu,J.,Batra,D.,Parikh,D.,andLee,S. Vilbert:Pretraining arXivpreprintarXiv:2004.14444,2020.
| task-agnostic |     | visiolinguistic | representations |     | for vision- |     |     |     |     |     |     |
| ------------- | --- | --------------- | --------------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
and-languagetasks. InAdvancesinNeuralInformation Mishra,A.,Alahari,K.,andJawahar,C. Scenetextrecogni-
|     |     |     |     |     |     | tionusinghigherorderlanguagepriors. |     |     |     | 2012. |     |
| --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | ----- | --- |
ProcessingSystems,pp.13–23,2019.
|     |     |     |     |     |     | Mithun, | N. C., Panda, | R., | Papalexakis, | E. E., and | Roy- |
| --- | --- | --- | --- | --- | --- | ------- | ------------- | --- | ------------ | ---------- | ---- |
Lu,Z.,Xiong,X.,Li,Y.,Stroud,J.,andRoss,D.Leveraging
weaklysuperviseddataandposerepresentationforaction Chowdhury,A.K. Weblysupervisedjointembeddingfor
recognition, 2020. URL https://www.youtube. cross-modalimage-textretrieval. InProceedingsofthe
26thACMinternationalconferenceonMultimedia,pp.
com/watch?v=KOQFxbPPLOE&t=1390s.
1856–1864,2018.
Lucic,M.,Kurach,K.,Michalski,M.,Gelly,S.,andBous-
quet, O. Are gans created equal? a large-scale study. Mori,Y.,Takahashi,H.,andOka,R. Image-to-wordtrans-
Advancesinneuralinformationprocessingsystems,31: formationbasedondividingandvectorquantizingimages
| 700–709,2018. |     |     |     |     |     | withwords. | Citeseer,1999. |     |     |     |     |
| ------------- | --- | --- | --- | --- | --- | ---------- | -------------- | --- | --- | --- | --- |
Mahajan,D.,Girshick,R.,Ramanathan,V.,He,K.,Paluri, Mu,J.,Liang,P.,andGoodman,N.Shapingvisualrepresen-
M., Li, Y., Bharambe, A., and van der Maaten, L. Ex- tationswithlanguageforfew-shotclassification. arXiv
ploringthelimitsofweaklysupervisedpretraining. In preprintarXiv:1911.02683,2019.

33
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Muller-Budack,E.,Pustu-Iren,K.,andEwerth,R. Geolo- Bai, J., and Chintala, S. Pytorch: An imperative style,
cation estimation of photos using a hierarchical model high-performance deep learning library. In Advances
andsceneclassification. InProceedingsoftheEuropean inNeuralInformationProcessingSystems32,pp.8024–
| ConferenceonComputerVision(ECCV),pp.563–579, |     |     |     |     | 8035,2019. |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- |
2018.
|     |     |     |     |     | Pedregosa, | F., Varoquaux, | G., | Gramfort, | A., | Michel, V., |
| --- | --- | --- | --- | --- | ---------- | -------------- | --- | --------- | --- | ----------- |
Murty,S.,Koh,P.W.,andLiang,P.Expbert:Representation Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
engineeringwithnaturallanguageexplanations. arXiv Weiss,R.,Dubourg,V.,Vanderplas,J.,Passos,A.,Cour-
preprintarXiv:2005.01932,2020. napeau,D.,Brucher,M.,Perrot,M.,andDuchesnay,E.
|                                          |     |     |     |          | Scikit-learn: | Machine | learning | in  | Python. | Journal of |
| ---------------------------------------- | --- | --- | --- | -------- | ------------- | ------- | -------- | --- | ------- | ---------- |
| Narasimhan,K.,Kulkarni,T.,andBarzilay,R. |     |     |     | Language |               |         |          |     |         |            |
MachineLearningResearch,12:2825–2830,2011.
understandingfortext-basedgamesusingdeepreinforce-
mentlearning. arXivpreprintarXiv:1506.08941,2015. Pennington, J., Socher, R., and Manning, C. D. Glove:
|     |     |     |     |     | Globalvectorsforwordrepresentation. |     |     |     | InProceedings |     |
| --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | ------------- | --- |
Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., ofthe2014conferenceonempiricalmethodsinnatural
and Ng, A. Y. Reading digits in natural images with languageprocessing(EMNLP),pp.1532–1543,2014.
| unsupervisedfeaturelearning. |     |     | 2011. |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
Peters,M.E.,Neumann,M.,Iyyer,M.,Gardner,M.,Clark,
Noble,S.U.Algorithmsofoppression:Howsearchengines C., Lee, K., and Zettlemoyer, L. Deep contextualized
reinforceracism. 2018. wordrepresentations. arXivpreprintarXiv:1802.05365,
2018.
| Nosek,B.A.,Banaji,M.R.,andGreenwald,A.G. |     |     |     | Harvest- |             |               |      |             |         |          |
| ---------------------------------------- | --- | --- | --- | -------- | ----------- | ------------- | ---- | ----------- | ------- | -------- |
|                                          |     |     |     |          | Qi, D., Su, | L., Song, J., | Cui, | E., Bharti, | T., and | Sacheti, |
ingimplicitgroupattitudesandbeliefsfromademonstra-
tionwebsite. GroupDynamics: Theory,Research,and A. Imagebert: Cross-modal pre-training with large-
|     |     |     |     |     | scaleweak-supervisedimage-textdata. |     |     |     | arXivpreprint |     |
| --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | ------------- | --- |
Practice,6(1):101,2002.
arXiv:2001.07966,2020.
Oh,S.,Hoogs,A.,Perera,A.,Cuntoor,N.,Chen,C.-C.,Lee,
|     |     |     |     |     | Quattoni,A.,Collins,M.,andDarrell,T. |     |     |     | Learningvisual |     |
| --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | -------------- | --- |
J.T.,Mukherjee,S.,Aggarwal,J.,Lee,H.,Davis,L.,etal.
Alarge-scalebenchmarkdatasetforeventrecognitionin representationsusingimageswithcaptions.In2007IEEE
ConferenceonComputerVisionandPatternRecognition,
| surveillancevideo. |     | InCVPR2011,pp.3153–3160.IEEE, |     |     |                   |     |     |     |     |     |
| ------------------ | --- | ----------------------------- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
| 2011.              |     |                               |     |     | pp.1–8.IEEE,2007. |     |     |     |     |     |
Radford,A.,Narasimhan,K.,Salimans,T.,andSutskever,
Oliver,A.,Odena,A.,Raffel,C.A.,Cubuk,E.D.,andGood-
I. Improvinglanguageunderstandingbygenerativepre-
| fellow,I. | Realisticevaluationofdeepsemi-supervised |     |     |     |     |     |     |     |     |     |
| --------- | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
training,2018.
| learningalgorithms. |     | Advancesinneuralinformationpro- |     |     |     |     |     |     |     |     |
| ------------------- | --- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
cessingsystems,31:3235–3246,2018.
Radford,A.,Wu,J.,Child,R.,Luan,D.,Amodei,D.,and
Sutskever,I.Languagemodelsareunsupervisedmultitask
| Oord,A.v.d.,Li,Y.,andVinyals,O. |             |            | Representationlearn- |                |           |       |     |     |     |     |
| ------------------------------- | ----------- | ---------- | -------------------- | -------------- | --------- | ----- | --- | --- | --- | --- |
|                                 |             |            |                      |                | learners. | 2019. |     |     |     |     |
| ing with                        | contrastive | predictive | coding.              | arXiv preprint |           |       |     |     |     |     |
arXiv:1807.03748,2018. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,
|                                   |     |     |     |                    | Matena,M.,Zhou,Y.,Li,W.,andLiu,P.J. |     |     |     |     | Exploring |
| --------------------------------- | --- | --- | --- | ------------------ | ----------------------------------- | --- | --- | --- | --- | --------- |
| Ordonez,V.,Kulkarni,G.,andBerg,T. |     |     |     | Im2text:Describing |                                     |     |     |     |     |           |
thelimitsoftransferlearningwithaunifiedtext-to-text
| imagesusing1millioncaptionedphotographs. |     |     |     | Advances |              |                                     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | -------- | ------------ | ----------------------------------- | --- | --- | --- | --- |
|                                          |     |     |     |          | transformer. | arXivpreprintarXiv:1910.10683,2019. |     |     |     |     |
inneuralinformationprocessingsystems,24:1143–1151,
|     |     |     |     |     | Raji, I. D., | Gebru, T., Mitchell, |     | M., Buolamwini, |     | J., Lee, |
| --- | --- | --- | --- | --- | ------------ | -------------------- | --- | --------------- | --- | -------- |
2011.
|     |     |     |     |     | J.,andDenton,E. | Savingface: |     | Investigatingtheethical |     |     |
| --- | --- | --- | --- | --- | --------------- | ----------- | --- | ----------------------- | --- | --- |
pandas development team, T. pandas-dev/pandas: Pan- concernsoffacialrecognitionauditing,2020.
| das, February | 2020. | URL | https://doi.org/10. |     |     |     |     |     |     |     |
| ------------- | ----- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- |
5281/zenodo.3509134. Ramanathan, V., Liang, P., and Fei-Fei, L. Video event
|            |              |     |            |                  | understanding | using       | natural       | language | descriptions. | In  |
| ---------- | ------------ | --- | ---------- | ---------------- | ------------- | ----------- | ------------- | -------- | ------------- | --- |
| Parkhi, O. | M., Vedaldi, | A., | Zisserman, | A., and Jawahar, |               |             |               |          |               |     |
|            |              |     |            |                  | Proceedings   | of the IEEE | International |          | Conference    | on  |
C.V. Catsanddogs. InIEEEConferenceonComputer ComputerVision,pp.905–912,2013.
VisionandPatternRecognition,2012.
Rashtchian,C.,Young,P.,Hodosh,M.,andHockenmaier,J.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Collectingimageannotationsusingamazon’smechanical
Chanan,G.,Killeen,T.,Lin,Z.,Gimelshein,N.,Antiga, turk. InProceedingsoftheNAACLHLT2010Workshop
L.,Desmaison,A.,Kopf,A.,Yang,E.,DeVito,Z.,Raison, onCreatingSpeechandLanguageDatawithAmazon’s
M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., MechanicalTurk,pp.139–147,2010.

34
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Recht,B.,Roelofs,R.,Schmidt,L.,andShankar,V. Doim- Sohn,K. Improveddeepmetriclearningwithmulti-class
agenetclassifiersgeneralizetoimagenet? arXivpreprint n-pairlossobjective. InAdvancesinneuralinformation
| arXiv:1902.10811,2019. |     |     |     |     |     | processingsystems,pp.1857–1865,2016. |     |     |     |     |
| ---------------------- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- |
Solaiman,I.,Brundage,M.,Clark,J.,Askell,A.,Herbert-
| Salimans,T.andKingma,D.P. |     |     | Weightnormalization: |     | A   |     |     |     |     |     |
| ------------------------- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
simplereparameterizationtoacceleratetrainingofdeep Voss,A.,Wu,J.,Radford,A.,Krueger,G.,Kim,J.W.,
neuralnetworks. InAdvancesinneuralinformationpro- Kreps, S., McCain, M., Newhouse, A., Blazakis, J.,
cessingsystems,pp.901–909,2016. McGuffie, K., andWang, J. Releasestrategiesandthe
socialimpactsoflanguagemodels,2019.
| Scheuerman,M.K.,Paul,J.M.,andBrubaker,J.R. |     |     |     |     | How |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
computersseegender: Anevaluationofgenderclassifica- Soomro,K.,Zamir,A.R.,andShah,M. Ucf101: Adataset
tionincommercialfacialanalysisservices. Proceedings of 101 human actions classes from videos in the wild.
oftheACMonHuman-ComputerInteraction,3(CSCW): arXivpreprintarXiv:1212.0402,2012.
1–33,2019.
URLhttps://doi.org/
|     |     |     |     |     |     | Speer,R. | ftfy. Zenodo,2019. |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | ------------------ | --- | --- | --- |
Schwemmer,C.,Knight,C.,Bello-Pardo,E.D.,Oklobdzija, 10.5281/zenodo.2591652. Version5.5.
| S., Schoonvelde, |     | M., and           | Lockhart, | J. W. Diagnosing |            |                                  |     |     |              |                    |
| ---------------- | --- | ----------------- | --------- | ---------------- | ---------- | -------------------------------- | --- | --- | ------------ | ------------------ |
|                  |     |                   |           |                  |            | Srivastava,N.andSalakhutdinov,R. |     |     |              | Multimodallearning |
| gender bias      | in  | image recognition |           | systems.         | Socius, 6: |                                  |     |     |              |                    |
|                  |     |                   |           |                  |            | withdeepboltzmannmachines.       |     |     | InNIPS,2012. |                    |
2378023120967171,2020.
|                                   |     |     |     |               |     | Srivastava,S.,Labutov,I.,andMitchell,T. |     |     |     | Jointconcept |
| --------------------------------- | --- | --- | --- | ------------- | --- | --------------------------------------- | --- | --- | --- | ------------ |
| Sennrich,R.,Haddow,B.,andBirch,A. |     |     |     | Neuralmachine |     |                                         |     |     |     |              |
learningandsemanticparsingfromnaturallanguageex-
| translation | of rare | words | with | subword units. | arXiv |             |     |             |        |                    |
| ----------- | ------- | ----- | ---- | -------------- | ----- | ----------- | --- | ----------- | ------ | ------------------ |
|             |         |       |      |                |       | planations. | In  | Proceedings | of the | 2017 conference on |
preprintarXiv:1508.07909,2015.
|                                                    |     |                                    |     |     |     | empirical       | methods | in natural | language | processing, pp. |
| -------------------------------------------------- | --- | ---------------------------------- | --- | --- | --- | --------------- | ------- | ---------- | -------- | --------------- |
| Shankar,V.,Dave,A.,Roelofs,R.,Ramanan,D.,Recht,B., |     |                                    |     |     |     | 1527–1536,2017. |         |            |          |                 |
| andSchmidt,L.                                      |     | Doimageclassifiersgeneralizeacross |     |     |     |                 |         |            |          |                 |
Stallkamp,J.,Schlipsing,M.,Salmen,J.,andIgel,C. The
time? arXivpreprintarXiv:1906.02168,2019.
GermanTrafficSignRecognitionBenchmark: Amulti-
|                                            |     |     |     |     |      | classclassificationcompetition. |     |     |     | InIEEEInternational |
| ------------------------------------------ | --- | --- | --- | --- | ---- | ------------------------------- | --- | --- | --- | ------------------- |
| Sharma,P.,Ding,N.,Goodman,S.,andSoricut,R. |     |     |     |     | Con- |                                 |     |     |     |                     |
ceptualcaptions: Acleaned,hypernymed,imagealt-text Joint Conference on Neural Networks, pp. 1453–1460,
| datasetforautomaticimagecaptioning. |     |     |     | InProceedings |     | 2011. |     |     |     |     |
| ----------------------------------- | --- | --- | --- | ------------- | --- | ----- | --- | --- | --- | --- |
ofthe56thAnnualMeetingoftheAssociationforCompu-
Stroud,J.C.,Ross,D.A.,Sun,C.,Deng,J.,Sukthankar,R.,
| tationalLinguistics(Volume1: |     |     | LongPapers),pp.2556– |     |     |              |     |                                      |     |     |
| ---------------------------- | --- | --- | -------------------- | --- | --- | ------------ | --- | ------------------------------------ | --- | --- |
|                              |     |     |                      |     |     | andSchmid,C. |     | Learningvideorepresentationsfromtex- |     |     |
2565,2018.
|                    |            |                            |           |            |           | tualwebsupervision. |            | arXivpreprintarXiv:2007.14937, |            |                |
| ------------------ | ---------- | -------------------------- | --------- | ---------- | --------- | ------------------- | ---------- | ------------------------------ | ---------- | -------------- |
| Singh, A.,         | Natarajan, | V., Shah,                  | M.,       | Jiang, Y., | Chen, X., | 2020.               |            |                                |            |                |
| Batra, D.,         | Parikh,    | D., and                    | Rohrbach, | M. Towards | vqa       |                     |            |                                |            |                |
|                    |            |                            |           |            |           | Szegedy,            | C., Ioffe, | S.,                            | Vanhoucke, | V., and Alemi, |
| modelsthatcanread. |            | InProceedingsoftheIEEECon- |           |            |           |                     |            |                                |            |                |
|                    |            |                            |           |            |           | A. Inception-v4,    |            | inception-resnet               |            | and the impact |
ferenceonComputerVisionandPatternRecognition,pp.
|     |     |     |     |     |     | of residual | connections |     | on learning. | arXiv preprint |
| --- | --- | --- | --- | --- | --- | ----------- | ----------- | --- | ------------ | -------------- |
8317–8326,2019.
arXiv:1602.07261,2016.
| Socher, R.andFei-Fei, |     | L. Connectingmodalities: |     |     | Semi- |                    |     |         |                        |     |
| --------------------- | --- | ------------------------ | --- | --- | ----- | ------------------ | --- | ------- | ---------------------- | --- |
|                       |     |                          |     |     |       | Tan,H.andBansal,M. |     | Lxmert: | Learningcross-modality |     |
supervisedsegmentationandannotationofimagesusing
encoderrepresentationsfromtransformers.arXivpreprint
| unalignedtextcorpora. |     | In2010IEEEComputerSociety |     |     |     |     |     |     |     |     |
| --------------------- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
arXiv:1908.07490,2019.
ConferenceonComputerVisionandPatternRecognition,
pp.966–973.IEEE,2010.
|             |            |         |     |             |          | Tan, M.                                | and Le, | Q. V. Efficientnet: |     | Rethinking model |
| ----------- | ---------- | ------- | --- | ----------- | -------- | -------------------------------------- | ------- | ------------------- | --- | ---------------- |
|             |            |         |     |             |          | scalingforconvolutionalneuralnetworks. |         |                     |     | arXivpreprint    |
| Socher, R., | Perelygin, | A., Wu, | J., | Chuang, J., | Manning, |                                        |         |                     |     |                  |
arXiv:1905.11946,2019.
| C.D.,Ng,A.Y.,andPotts,C. |     |     | Recursivedeepmodelsfor |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- |
semanticcompositionalityoverasentimenttreebank. In Taori, R., Dave, A., Shankar, V., Carlini, N., Recht, B.,
Proceedingsofthe2013conferenceonempiricalmethods and Schmidt, L. Measuring robustness to natural dis-
innaturallanguageprocessing,pp.1631–1642,2013. tribution shifts in image classification. arXiv preprint
arXiv:2007.00644,2020.
Socher,R.,Karpathy,A.,Le,Q.V.,Manning,C.D.,andNg,
A.Y. Groundedcompositionalsemanticsforfindingand Thomee,B.,Shamma,D.A.,Friedland,G.,Elizalde,B.,Ni,
describing images with sentences. Transactions of the K.,Poland,D.,Borth,D.,andLi,L.-J. Yfcc100m: The
Association for Computational Linguistics, 2:207–218, newdatainmultimediaresearch. Communicationsofthe
| 2014. |     |     |     |     |     | ACM,59(2):64–73,2016. |     |     |     |     |
| ----- | --- | --- | --- | --- | --- | --------------------- | --- | --- | --- | --- |

35
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Tian,Y.,Krishnan,D.,andIsola,P. Contrastivemultiview Wang,H.,Lu,P.,Zhang,H.,Yang,M.,Bai,X.,Xu,Y.,He,
coding. arXivpreprintarXiv:1906.05849,2019. M.,Wang,Y.,andLiu,W. Allyouneedisboundary: To-
|           |               |               |          |                |                 |        |     | wardarbitrary-shapedtextspotting.       |     |     | InProceedingsofthe |           |     |
| --------- | ------------- | ------------- | -------- | -------------- | --------------- | ------ | --- | --------------------------------------- | --- | --- | ------------------ | --------- | --- |
| Tian, Y., | Wang,         | Y., Krishnan, |          | D., Tenenbaum, |                 | J. B., | and |                                         |     |     |                    |           |     |
|           |               |               |          |                |                 |        |     | AAAIConferenceonArtificialIntelligence, |     |     |                    | volume34, |     |
| Isola,    | P. Rethinking |               | few-shot | image          | classification: |        | a   |                                         |     |     |                    |           |     |
pp.12160–12167,2020.
| good embedding |     | is  | all you | need? | arXiv | preprint |     |     |     |     |     |     |     |
| -------------- | --- | --- | ------- | ----- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- |
arXiv:2003.11539,2020.
|     |     |     |     |     |     |     |     | Wang,J.,Markert,K.,andEveringham,M. |     |     |     | Learningmod- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | ------------ | --- |
elsforobjectrecognitionfromnaturallanguagedescrip-
| Torralba,A.,Fergus,R.,andFreeman,W.T. |         |      |         |               | 80milliontiny |        |     |                           |     |     |         |     |     |
| ------------------------------------- | ------- | ---- | ------- | ------------- | ------------- | ------ | --- | ------------------------- | --- | --- | ------- | --- | --- |
|                                       |         |      |         |               |               |        |     | tions. InBMVC,volume1,pp. |     |     | 2,2009. |     |     |
| images:                               | A large | data | set for | nonparametric |               | object | and |                           |     |     |         |     |     |
IEEEtransactionsonpatternanalysis
scenerecognition.
|     |     |     |     |     |     |     |     | Weston, J., | Bengio, S., | and Usunier, | N.  | Large | scale im- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ----------- | ------------ | --- | ----- | --------- |
andmachineintelligence,30(11):1958–1970,2008.
|                                        |              |     |            |     |              |     |      | ageannotation: | learningtorankwithjointword-image |     |     |     |     |
| -------------------------------------- | ------------ | --- | ---------- | --- | ------------ | --- | ---- | -------------- | --------------------------------- | --- | --- | --- | --- |
|                                        |              |     |            |     |              |     |      | embeddings.    | Machinelearning,81(1):21–35,2010. |     |     |     |     |
| Touvron,                               | H., Vedaldi, |     | A., Douze, | M., | andJe´gou,   | H.  | Fix- |                |                                   |     |     |     |     |
| ingthetrain-testresolutiondiscrepancy. |              |     |            |     | InAdvancesin |     |      |                |                                   |     |     |     |     |
neuralinformationprocessingsystems,pp.8252–8262, Weston,J.E. Dialog-basedlanguagelearning. InAdvances
inNeuralInformationProcessingSystems,pp.829–837,
2019.
2016.
| Varadarajan,J.andOdobez,J.-M. |     |     |     | Topicmodelsforscene |     |     |     |     |     |     |     |     |     |
| ----------------------------- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
analysisandabnormalitydetection. In2009IEEE12th Weyand,T.,Kostrikov,I.,andPhilbin,J.Planet-photogeolo-
InternationalConferenceonComputerVisionWorkshops, cationwithconvolutionalneuralnetworks. InEuropean
|     |     |     |     |     |     |     |     | Conference | on Computer | Vision, | pp. 37–55. |     | Springer, |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | ------- | ---------- | --- | --------- |
ICCVWorkshops,pp.1338–1345.IEEE,2009.
2016.
Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,
L.,Gomez,A.N.,Kaiser,Ł.,andPolosukhin,I. Atten- Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., and Gir-
tionisallyouneed. InAdvancesinneuralinformation shick, R. Detectron2. https://github.com/
processingsystems,pp.5998–6008,2017.
facebookresearch/detectron2,2019.
| Veeling, | B. S., Linmans, |     | J., Winkens, |     | J., Cohen, | T., | and |                                |     |     |                     |     |     |
| -------- | --------------- | --- | ------------ | --- | ---------- | --- | --- | ------------------------------ | --- | --- | ------------------- | --- | --- |
|          |                 |     |              |     |            |     |     | Wu,Z.,Xiong,Y.,Yu,S.,andLin,D. |     |     | Unsupervisedfeature |     |     |
Welling,M. RotationequivariantCNNsfordigitalpathol- learningvianon-parametricinstance-leveldiscrimination.
ogy. June2018.
arXivpreprintarXiv:1805.01978,2018.
Virtanen,P.,Gommers,R.,Oliphant,T.E.,Haberland,M.,
|        |                 |     |     |           |     |           |     | Xie,Q.,Luong,M.-T.,Hovy,E.,andLe,Q.V.           |     |     |     | Self-training |     |
| ------ | --------------- | --- | --- | --------- | --- | --------- | --- | ----------------------------------------------- | --- | --- | --- | ------------- | --- |
| Reddy, | T., Cournapeau, |     | D., | Burovski, | E., | Peterson, | P., |                                                 |     |     |     |               |     |
|        |                 |     |     |           |     |           |     | withnoisystudentimprovesimagenetclassification. |     |     |     |               | In  |
Weckesser,W.,Bright,J.,vanderWalt,S.J.,Brett,M.,
ProceedingsoftheIEEE/CVFConferenceonComputer
Wilson,J.,Millman,K.J.,Mayorov,N.,Nelson,A.R.J.,
VisionandPatternRecognition,pp.10687–10698,2020.
| Jones, | E., Kern, | R., | Larson, | E., Carey, | C.  | J., Polat, | ˙I., |     |     |     |     |     |     |
| ------ | --------- | --- | ------- | ---------- | --- | ---------- | ---- | --- | --- | --- | --- | --- | --- |
Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., y Arcas, B. A., Mitchell, M., and Todorov,
Perktold,J.,Cimrman,R.,Henriksen,I.,Quintero,E.A.,
|     |     |     |     |     |     |     |     | A.  | Physiognomy’s | new | clothes. |     | 2017. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | --- | ----- |
Harris,C.R.,Archibald,A.M.,Ribeiro,A.H.,Pedregosa,
|     |     |     |     |     |     |     |     | URL | https://medium.com/@blaisea/ |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- |
F.,vanMulbregt,P.,andSciPy1.0Contributors. SciPy physiognomys-new-clothes-f2d4b59fdd6a.
1.0: FundamentalAlgorithmsforScientificComputing
Nature Methods,
in Python. 17:261–272, 2020. doi: Yang, Z., Lu, Y., Wang, J., Yin, X., Florencio, D., Wang,
10.1038/s41592-019-0686-2. L.,Zhang,C.,Zhang,L.,andLuo,J. Tap: Text-aware
arXivpreprint
pre-trainingfortext-vqaandtext-caption.
| Vo,N.,Jacobs,N.,andHays,J. |     |     |     | Revisitingim2gpsinthe |     |     |     |     |     |     |     |     |     |
| -------------------------- | --- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
arXiv:2012.04638,2020.
| deeplearningera. |     | InProceedingsoftheIEEEInterna- |     |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionalConferenceonComputerVision,pp.2621–2630,
|     |     |     |     |     |     |     |     | Yogatama, | D., d’Autume, | C. d. M., | Connor, | J., | Kocisky, |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------------- | --------- | ------- | --- | -------- |
2017.
T.,Chrzanowski,M.,Kong,L.,Lazaridou,A.,Ling,W.,
Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Yu,L.,Dyer,C.,etal. Learningandevaluatinggeneral
|                                              |     |       |                              |     |     |     |       | linguisticintelligence. |     | arXivpreprintarXiv:1901.11373, |     |     |     |
| -------------------------------------------- | --- | ----- | ---------------------------- | --- | --- | --- | ----- | ----------------------- | --- | ------------------------------ | --- | --- | --- |
| Bowman,S.R.                                  |     | Glue: | Amulti-taskbenchmarkandanal- |     |     |     |       |                         |     |                                |     |     |     |
| ysisplatformfornaturallanguageunderstanding. |     |       |                              |     |     |     | arXiv | 2019.                   |     |                                |     |     |     |
preprintarXiv:1804.07461,2018.
|     |     |     |     |     |     |     |     | Young,P.,Lai,A.,Hodosh,M.,andHockenmaier,J. |     |     |     |     | From |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- | ---- |
Wang,H.,Ge,S.,Lipton,Z.,andXing,E.P. Learningro- imagedescriptionstovisualdenotations: Newsimilarity
bustglobalrepresentationsbypenalizinglocalpredictive metrics for semantic inference over event descriptions.
power. In Advances inNeural InformationProcessing TransactionsoftheAssociationforComputationalLin-
| Systems,pp.10506–10518,2019. |     |     |     |     |     |     |     | guistics,2:67–78,2014. |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- |

36
LearningTransferableVisualModelsFromNaturalLanguageSupervision
| Yu, F.,   | Tang, J., Yin,  | W., | Sun,      | Y., Tian, | H., Wu,  | H.,     |
| --------- | --------------- | --- | --------- | --------- | -------- | ------- |
| and Wang, | H. Ernie-vil:   |     | Knowledge |           | enhanced | vision- |
| language  | representations |     | through   | scene     | graph.   | arXiv   |
preprintarXiv:2006.16934,2020.
| Zeiler,M.D.andFergus,R.   |     |     | Visualizingandunderstand- |     |     |     |
| ------------------------- | --- | --- | ------------------------- | --- | --- | --- |
| ingconvolutionalnetworks. |     |     | InEuropeanconferenceon    |     |     |     |
computervision,pp.818–833.Springer,2014.
| Zhai, X., | Puigcerver, | J., | Kolesnikov, |     | A., Ruyssen, | P., |
| --------- | ----------- | --- | ----------- | --- | ------------ | --- |
Riquelme,C.,Lucic,M.,Djolonga,J.,Pinto,A.S.,Neu-
| mann,M.,Dosovitskiy,A.,etal. |                                            |      |     | Alarge-scalestudyof |                 |     |
| ---------------------------- | ------------------------------------------ | ---- | --- | ------------------- | --------------- | --- |
| representation               | learning                                   | with | the | visual              | task adaptation |     |
| benchmark.                   | arXivpreprintarXiv:1910.04867,2019.        |      |     |                     |                 |     |
| Zhang,R.                     | Makingconvolutionalnetworksshift-invariant |      |     |                     |                 |     |
again. arXivpreprintarXiv:1904.11486,2019.
Zhang,Y.,Jiang,H.,Miura,Y.,Manning,C.D.,andLan-
| glotz,C.P.                         | Contrastivelearningofmedicalvisualrepre- |     |     |     |               |     |
| ---------------------------------- | ---------------------------------------- | --- | --- | --- | ------------- | --- |
| sentationsfrompairedimagesandtext. |                                          |     |     |     | arXivpreprint |     |
arXiv:2010.00747,2020.
| Zuboff,                               | S. Big other: | surveillance |     | capitalism |                 | and the |
| ------------------------------------- | ------------- | ------------ | --- | ---------- | --------------- | ------- |
| prospectsofaninformationcivilization. |               |              |     |            | JournalofInfor- |         |
mationTechnology,30(1):75–89,2015.

37
LearningTransferableVisualModelsFromNaturalLanguageSupervision
A.Linear-probeevaluation the ResNet-50 architecture as in the smallest contrastive
|     |     |     |     |     |     |     | model. Todoso,theoutputfromtheCNNisprojectedinto |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
Weprovideadditionaldetailsforlinearprobeexperiments
|     |     |     |     |     |     |     | four tokens, | which | are then | fed as | a prefix | to a language |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | -------- | ------ | -------- | ------------- | --- |
presentedinthispaper,includingthelistofthedatasetsand
|     |     |     |     |     |     |     | model autoregressively |     | predicting |     | the text | tokens. | Apart |
| --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | ---------- | --- | -------- | ------- | ----- |
modelsusedforevaluation.
|     |     |     |     |     |     |     | fromthetrainingobjective, |     |     | themodelwastrainedonthe |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------- | --- | --- | ----------------------- | --- | --- | --- |
samedatasetforthesamenumberofepochsasotherCLIP
| A.1.Datasets     |          |               |           |              |             |        | models.      |                                         |     |             |     |      |        |
| ---------------- | -------- | ------------- | --------- | ------------ | ----------- | ------ | ------------ | --------------------------------------- | --- | ----------- | --- | ---- | ------ |
| We use           | the 12   | datasets from | the       | well-studied | evaluation  |        |              |                                         |     |             |     |      |        |
|                  |          |               |           |              |             |        | CLIP-RN      | Five ResNet-based                       |     | contrastive |     | CLIP | models |
| suite introduced |          | by (Kornblith | et        | al., 2019)   | and         | add 15 |              |                                         |     |             |     |      |        |
|                  |          |               |           |              |             |        | areincluded. | Asdiscussedinthepaper,thefirsttwomodels |     |             |     |      |        |
| additional       | datasets | in order      | to assess | the          | performance | of     |              |                                         |     |             |     |      |        |
modelsonawidervarietyofdistributionsandtasks. These followResNet-50andResNet-101,andweuseEfficientNet-
|     |     |     |     |     |     |     | style (Tan | & Le, 2019) | scaling | for | the next | three | models |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | ------- | --- | -------- | ----- | ------ |
datasetsincludeMNIST,theFacialExpressionRecognition
2013 dataset (Goodfellow et al., 2015), STL-10 (Coates whichsimultaneouslyscalethemodelwidth, thenumber
|                |         |         |     |             |     |       | of layers, | and the input | resolution |     | to obtain | models | with |
| -------------- | ------- | ------- | --- | ----------- | --- | ----- | ---------- | ------------- | ---------- | --- | --------- | ------ | ---- |
| et al., 2011), | EuroSAT | (Helber | et  | al., 2019), | the | NWPU- |            |               |            |     |           |        |      |
roughly4x,16x,and64xcomputation.
RESISC45dataset(Chengetal.,2017),theGermanTraf-
| fic Sign | Recognition | Benchmark |     | (GTSRB) | dataset | (Stal- |          |                                      |     |     |     |     |     |
| -------- | ----------- | --------- | --- | ------- | ------- | ------ | -------- | ------------------------------------ | --- | --- | --- | --- | --- |
|          |             |           |     |         |         |        | CLIP-ViT | WeincludefourCLIPmodelsthatusetheVi- |     |     |     |     |     |
lkampetal.,2011),theKITTIdataset(Geigeretal.,2012),
PatchCamelyon(Veelingetal.,2018),theUCF101action sionTransformer(Dosovitskiyetal.,2020)architectureas
|     |     |     |     |     |     |     | theimageencoder. | Weincludethreemodelstrainedon224- |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --------------------------------- | --- | --- | --- | --- | --- |
recognitiondataset(Soomroetal.,2012),Kinetics700(Car-
|          |             |       |        |         |        |       | by-224pixelimages: |     | ViT-B/32,ViT-B/16,ViT-L/14,and |     |     |     |     |
| -------- | ----------- | ----- | ------ | ------- | ------ | ----- | ------------------ | --- | ------------------------------ | --- | --- | --- | --- |
| reira et | al., 2019), | 2,500 | random | samples | of the | CLEVR |                    |     |                                |     |     |     |     |
dataset(Johnsonetal.,2017),theHatefulMemesdataset theViT-L/14modelfine-tunedon336-by-336pixelinput
images.
| (Kiela et    | al., 2020),                         | and | the ImageNet-1k |     | dataset | (Deng |             |        |          |        |         |      |     |
| ------------ | ----------------------------------- | --- | --------------- | --- | ------- | ----- | ----------- | ------ | -------- | ------ | ------- | ---- | --- |
| etal.,2012). | Forthetwovideodatasets(UCF101andKi- |     |                 |     |         |       |             |        |          |        |         |      |     |
|              |                                     |     |                 |     |         |       | EfficietNet | We use | the nine | models | (B0-B8) | from | the |
netics700),weusethemiddleframeofeachvideoclipas
theinputimage. STL-10andUCF101havemultiplepre- original EfficientNet paper (Tan & Le, 2019), as well as
definedtrain/validation/testsplits,10and3respectively,and the noisy-student variants (B0-B7, L2-475, and L2-800)
|                                  |     |     |     |                      |     |     | (Tan&Le,2019). | Thelargestmodels(L2-475andL2-800) |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | -------------------- | --- | --- | -------------- | --------------------------------- | --- | --- | --- | --- | --- |
| wereporttheaverageoverallsplits. |     |     |     | Detailsoneachdataset |     |     |                |                                   |     |     |     |     |     |
andthecorrespondingevaluationmetricsareprovidedin taketheinputresolutionsof475x475and800x800pixels,
| Table9.       |              |         |              |            |         |            | respectively.               |                 |     |         |                    |             |     |
| ------------- | ------------ | ------- | ------------ | ---------- | ------- | ---------- | --------------------------- | --------------- | --- | ------- | ------------------ | ----------- | --- |
| Additionally, | we           | created | two datasets | that       | we call | Coun-      |                             |                 |     |         |                    |             |     |
|               |              |         |              |            |         |            | Instagram-pretrainedResNeXt |                 |     |         | Weusethefourmodels |             |     |
| try211        | and Rendered | SST2.   | The          | Country211 |         | dataset is |                             |                 |     |         |                    |             |     |
|               |              |         |              |            |         |            | (32x8d,                     | 32x16d, 32x32d, |     | 32x48d) | released           | by (Mahajan |     |
designedtoassessthegeolocationcapabilityofvisualrep-
etal.,2018),aswellastheirtwoFixResvariantswhichuse
| resentations. | WefilteredtheYFCC100mdataset(Thomee |     |     |     |     |     |     |     |     |     |     |     |     |
| ------------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
higherinputresolutions(Touvronetal.,2019).
| et al., 2016) | to  | find 211 | countries | (defined | as  | having an |     |     |     |     |     |     |     |
| ------------- | --- | -------- | --------- | -------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
ISO-3166countrycode)thathaveatleast300photoswith
|     |     |     |     |     |     |     | Big Transfer | (BiT) | We  | use BiT-S | and BiT-M |     | models |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | --- | --------- | --------- | --- | ------ |
GPScoordinates,andwebuiltabalanceddatasetwith211
(Kolesnikovetal.,2019),trainedontheImageNet-1kand
| categories, | by  | sampling | 200 photos | for training |     | and 100 |                       |     |                              |     |     |     |     |
| ----------- | --- | -------- | ---------- | ------------ | --- | ------- | --------------------- | --- | ---------------------------- | --- | --- | --- | --- |
|             |     |          |            |              |     |         | ImageNet-21kdatasets. |     | ThemodelweightsforBiT-Lisnot |     |     |     |     |
photosfortesting,foreachcountry.
publiclyavailable.
TheRenderedSST2datasetisdesignedtomeasuretheopti-
calcharacterrecognitioncapabilityofvisualrepresentations. Vision Transformer (ViT) We also include four ViT
Todoso,weusedthesentencesfromtheStanfordSentiment (Dosovitskiy et al., 2020) checkpoints pretrained on the
Treebankdataset(Socheretal.,2013)andrenderedthem ImageNet-21k dataset, namely ViT-B/32, ViT-B/16, ViT-
intoimages,withblacktextsonawhitebackground,ina L/16, and ViT-H/14. We note that their best-performing
448×448resolution. Twoexampleimagesfromthisdataset models,trainedontheJFT-300Mdataset,arenotavailable
| areshowninFigure19. |     |     |     |     |     |     | publicly. |                                     |     |     |     |     |     |
| ------------------- | --- | --- | --- | --- | --- | --- | --------- | ----------------------------------- | --- | --- | --- | --- | --- |
| A.2.Models          |     |     |     |     |     |     | SimCLRv2  | TheSimCLRv2(Chenetal.,2020c)project |     |     |     |     |     |
releasedpre-trainedandfine-tunedmodelsinvariousset-
Incombinationwiththedatasetslistedabove,weevaluate
|     |     |     |     |     |     |     | tings. We | use the | seven pretrain-only |     | checkpoints |     | with |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------- | ------------------- | --- | ----------- | --- | ---- |
thefollowingseriesofmodelsusinglinearprobes.
selectivekernels.
LM RN50 This is a multimodal model that uses an au- BYOL We use the recently released model weights of
toregressivelossinsteadofacontrastiveloss,whileusing BYOL(Grilletal.,2020),specificallytheir50x1and200x2

38
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     |     |     |     | Figure19. | TwoexampleimagesfromtheRenderedSST2dataset |     |     |     |     |     |
| --- | --- | --- | --- | --------- | ------------------------------------------ | --- | --- | --- | --- | --- |
checkpoints. atestsplit, weusetheprovidedvalidationsettoperform
thehyperparametersearch,andforthedatasetsthatdonot
MomentumContrast(MoCo) WeincludetheMoCo-v1 provideavalidationsplitorhavenotpublishedlabelsfor
|             |       |         |         |       |     |                | the test | data, we split | the training dataset | to perform the |
| ----------- | ----- | ------- | ------- | ----- | --- | -------------- | -------- | -------------- | -------------------- | -------------- |
| (He et al., | 2020) | and the | MoCo-v2 | (Chen |     | et al., 2020d) |          |                |                      |                |
checkpoints. hyperparametersearch. Forthefinalresult,wecombinethe
validationsplitbackwiththetrainingsplitandreportthe
performanceontheunusedsplit.
VirTex WeusethepretrainedmodelofVirTex(Desai&
| Johnson,2020). |     | WenotethatVirTexhasasimilarmodel |     |     |     |     |     |     |     |     |
| -------------- | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
A.4.Results
designtoCLIP-ARbutistrainedona1000xsmallerdataset
ofhigh-qualitycaptionsfromMSCOCO.
TheindividuallinearprobescoresareprovidedinTable10
|     |     |     |     |     |     |     | andplottedinFigure20. |     | Thebest-performingCLIPmodel, |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | ---------------------------- | --- |
ResNet WeaddtheoriginalResNetcheckpointsreleased usingViT-L/14archiectureand336-by-336pixelimages,
by(Heetal.,2016b),namelyResNet-50,ResNet-101,and achieved the state of the art in 21 of the 27 datasets, i.e.
ResNet152.
includedintheClopper-Pearson99.5%confidenceinterval
|     |     |     |     |     |     |     | aroundeachdataset’stopscore. |     | Formanydatasets,CLIP |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | -------------------- | --- |
A.3.Evaluation performssignificantlybetterthanothermodels,demonstrat-
ingtheadvantageofnaturallanguagesupervisionovertradi-
Weuseimagefeaturestakenfromthepenultimatelayerof
tionalpre-trainingapproachesbasedonimageclassification.
| eachmodel,ignoringanyclassificationlayerprovided. |         |     |          |          |        | For        |             |              |             |                     |
| ------------------------------------------------- | ------- | --- | -------- | -------- | ------ | ---------- | ----------- | ------------ | ----------- | ------------------- |
|                                                   |         |     |          |          |        |            | See Section | 3.2 for more | discussions | on the linear probe |
| CLIP-ViT                                          | models, | we  | used the | features | before | the linear |             |              |             |                     |
results.
projectiontotheembeddingspace,whichcorrespondsto
| I f in Figure        |             | 3. We train | a logistic      |     | regression    | classifier     |     |     |     |     |
| -------------------- | ----------- | ----------- | --------------- | --- | ------------- | -------------- | --- | --- | --- | --- |
| using scikit-learn’s |             | L-BFGS      | implementation, |     |               | with maxi-     |     |     |     |     |
| mum 1,000            | iterations, |             | and report      | the | corresponding | met-           |     |     |     |     |
| ric for each         | dataset.    | We          | determine       | the | L2            | regularization |     |     |     |     |
strengthλusingahyperparametersweeponthevalidation
|              |           |                                      | 10−6         |              | 106,   |              |     |     |     |     |
| ------------ | --------- | ------------------------------------ | ------------ | ------------ | ------ | ------------ | --- | --- | --- | --- |
| sets over    | the range | between                              |              | and          |        | with 96 log- |     |     |     |     |
| arithmically | spaced    | steps.                               | To           | save compute |        | required for |     |     |     |     |
| the sweeps,  | we        | perform                              | a parametric |              | binary | search that  |     |     |     |     |
| startswithλ  | =         | [10−6,10−4,10−2,1,102,104,106]andit- |              |              |        |              |     |     |     |     |
erativelyhalvestheintervalaroundthepeakuntilitreaches
| a resolution | of  | 8 steps | per decade. |     | The hyperparameter |     |     |     |     |     |
| ------------ | --- | ------- | ----------- | --- | ------------------ | --- | --- | --- | --- | --- |
sweepsareperformedonavalidationsplitofeachdataset.
Forthedatasetsthatcontainavalidationsplitinadditionto

39
LearningTransferableVisualModelsFromNaturalLanguageSupervision
| Dataset                      | Classes Trainsize | Testsize Evaluationmetric |                |
| ---------------------------- | ----------------- | ------------------------- | -------------- |
| Food-101                     | 102 75,750        | 25,250                    | accuracy       |
| CIFAR-10                     | 10 50,000         | 10,000                    | accuracy       |
| CIFAR-100                    | 100 50,000        | 10,000                    | accuracy       |
| Birdsnap                     | 500 42,283        | 2,149                     | accuracy       |
| SUN397                       | 397 19,850        | 19,850                    | accuracy       |
| StanfordCars                 | 196 8,144         | 8,041                     | accuracy       |
| FGVCAircraft                 | 100 6,667         | 3,333                     | meanperclass   |
| PascalVOC2007Classification  | 20 5,011          | 4,952                     | 11-pointmAP    |
| DescribableTextures          | 47 3,760          | 1,880                     | accuracy       |
| Oxford-IIITPets              | 37 3,680          | 3,669                     | meanperclass   |
| Caltech-101                  | 102 3,060         | 6,085                     | mean-per-class |
| OxfordFlowers102             | 102 2,040         | 6,149                     | meanperclass   |
| MNIST                        | 10 60,000         | 10,000                    | accuracy       |
| FacialEmotionRecognition2013 | 8 32,140          | 3,574                     | accuracy       |
| STL-10                       | 10 1000           | 8000                      | accuracy       |
| EuroSAT                      | 10 10,000         | 5,000                     | accuracy       |
| RESISC45                     | 45 3,150          | 25,200                    | accuracy       |
| GTSRB                        | 43 26,640         | 12,630                    | accuracy       |
| KITTI                        | 4 6,770           | 711                       | accuracy       |
| Country211                   | 211 43,200        | 21,100                    | accuracy       |
| PatchCamelyon                | 2 294,912         | 32,768                    | accuracy       |
| UCF101                       | 101 9,537         | 1,794                     | accuracy       |
| Kinetics700                  | 700 494,801       | 31,669 mean(top1,top5)    |                |
| CLEVRCounts                  | 8 2,000           | 500                       | accuracy       |
| HatefulMemes                 | 2 8,500           | 500                       | ROCAUC         |
| RenderedSST2                 | 2 7,792           | 1,821                     | accuracy       |
| ImageNet                     | 1000 1,281,167    | 50,000                    | accuracy       |
Table9.Datasetsexaminedforlinearprobes. Wenotethat,fortheBirdsnapandKinetics700datasets,weusedtheresourcesthatare
availableonlineatthetimeofthiswriting.

LearningTransferableVisualModelsFromNaturalLanguageSupervision 40
101dooF 01RAFIC 001RAFIC pansdriB 793NUS
sraC
tfarcriA 7002COV
DTD steP
101hcetlaC srewolF TSINM 3102REF (cid:63)01LTS TASoruE 54CSISER BRSTG ITTIK 112yrtnuoC MACP 101FCU 007sciteniK RVELC
semeMlufetaH
TSS
teNegamI
LMRN50 81.382.861.744.269.674.944.985.571.582.885.591.196.660.195.393.484.073.870.219.082.976.451.951.265.276.865.2
NR-PILC
50 86.488.770.356.473.378.349.187.176.488.289.696.198.364.296.695.287.582.470.225.382.781.657.253.665.772.673.3
101 88.991.173.558.675.184.050.788.076.391.092.096.498.465.297.895.989.382.473.626.682.884.060.350.368.273.375.7
50x4 91.390.573.065.777.085.957.388.479.591.992.597.898.568.197.896.489.785.559.430.383.085.762.652.568.076.678.2
50x16 93.392.274.972.879.288.762.789.079.193.593.798.398.968.798.697.091.489.069.234.883.588.066.353.871.180.081.5
50x64 94.894.178.677.281.190.567.788.982.094.595.498.998.971.399.197.192.890.269.240.783.789.569.155.075.081.283.6
TiV-PILC B/32 88.895.180.558.576.681.852.087.776.590.093.096.999.069.298.397.090.585.366.227.883.985.561.752.166.770.876.1
B/16 92.896.283.167.878.486.759.589.279.293.194.798.199.069.599.097.192.786.667.833.383.588.466.157.170.375.580.2
L/14 95.298.087.577.081.890.969.489.682.195.196.599.299.272.299.798.294.192.564.742.985.891.572.057.876.280.883.9
L/14-336px 95.997.987.479.982.291.571.689.983.095.196.099.299.272.999.798.194.992.469.246.485.692.073.060.377.380.585.4
teNtneicfifE
B0 74.392.576.559.762.062.555.784.471.293.093.391.798.257.297.197.385.580.073.812.483.174.447.647.955.753.476.9
B1 74.293.277.261.362.662.556.184.774.293.493.692.498.357.097.596.884.575.975.512.582.774.748.544.354.554.478.6
B2 75.893.677.964.464.063.257.085.373.593.993.592.998.556.697.796.984.476.473.112.684.375.149.442.655.455.279.7
B3 77.494.078.066.564.466.059.385.873.194.193.793.398.557.198.297.385.075.876.113.483.378.150.945.153.854.881.0
B4 79.794.178.770.165.466.460.486.573.494.793.593.298.857.998.696.885.078.372.313.983.179.152.546.554.455.482.9
B5 81.593.677.972.467.172.768.986.773.995.094.794.598.458.598.796.886.078.569.614.984.780.954.546.653.356.383.7
B6 82.494.078.073.565.871.168.287.673.995.094.193.798.460.298.796.885.478.172.715.384.280.054.151.153.357.084.0
B7 84.594.980.174.769.077.172.387.276.895.294.795.998.661.399.196.386.880.875.816.485.281.956.851.954.457.884.8
B8 84.595.080.775.269.676.871.587.477.194.995.296.398.661.499.297.087.480.470.917.485.282.457.751.451.755.885.3
tnedutSysioNteNtneicfifE B0 78.194.078.663.565.557.253.785.675.693.893.194.598.155.698.297.084.374.071.614.083.176.751.747.355.755.078.5
B1 80.495.180.266.667.659.653.786.277.094.694.495.198.056.198.696.984.373.167.114.583.979.954.546.154.354.981.1
B2 80.995.381.367.667.960.955.286.377.795.094.794.498.055.598.897.384.671.770.014.682.980.155.146.154.155.382.2
B3 82.695.982.168.668.860.655.486.577.295.094.895.298.156.099.196.585.070.569.515.183.181.856.845.155.752.083.8
B4 85.295.681.072.569.756.152.687.078.794.895.295.398.256.099.395.384.861.964.816.082.883.459.843.255.353.085.4
B5 87.696.382.475.371.664.764.887.879.695.595.696.698.860.999.496.187.068.573.716.483.586.461.646.353.455.885.8
B6 87.397.083.975.871.467.665.687.378.595.296.497.298.661.999.596.686.170.772.417.684.285.561.049.654.655.786.4
B7 88.496.082.076.972.672.271.288.180.595.595.596.698.562.799.496.288.573.473.018.583.886.663.250.557.256.787.0
L2-475 91.699.091.074.876.475.166.889.581.995.696.597.798.967.599.697.089.573.468.922.286.389.468.258.358.655.288.3
L2-800 92.098.789.078.575.775.568.489.482.595.694.797.998.568.499.797.289.977.766.923.786.888.966.762.758.456.988.4
margatsnI
32x8d 84.895.980.963.869.074.256.088.075.495.493.991.797.460.799.195.782.172.369.216.782.380.156.842.253.355.283.3
32x16d 85.796.580.964.870.577.556.787.976.295.694.992.597.461.699.395.582.873.866.117.583.481.158.241.354.256.184.4
32x32d 86.796.882.767.171.577.555.488.378.595.895.394.497.962.499.395.785.471.266.818.083.782.158.839.755.356.785.0
32x48d 86.996.883.465.972.276.653.288.077.295.595.893.698.163.799.495.385.473.067.218.582.782.859.241.355.556.785.2
FixRes-v1 88.595.781.167.472.980.557.688.077.995.896.194.597.962.299.496.286.676.564.819.382.583.459.843.556.659.086.0
FixRes-v2 88.595.781.167.372.980.757.588.077.995.096.094.598.062.199.496.586.676.364.819.582.383.559.844.256.659.086.0
S-TiB
R50x1 72.591.774.857.761.153.552.583.772.492.391.292.098.456.196.497.485.070.066.012.583.072.347.548.354.155.375.2
R50x3 75.193.779.061.163.755.254.184.874.692.591.692.898.858.797.097.886.473.173.814.084.276.450.049.254.754.277.2
R101x1 73.592.877.458.461.354.052.484.473.592.591.890.698.356.596.897.384.669.468.912.682.073.548.645.452.655.576.0
R101x3 74.793.979.857.862.954.753.384.775.592.391.292.698.859.797.398.085.571.860.214.183.175.950.449.754.154.677.4
R152x2 74.994.379.758.762.755.953.685.374.993.092.091.798.658.397.197.886.271.871.613.984.176.249.948.253.855.977.1
R152x4 74.794.279.257.862.951.250.885.475.493.191.291.498.961.497.298.085.572.867.914.983.176.050.342.953.656.078.5
M-TiB
R50x1 83.394.982.270.969.959.055.686.877.391.593.999.498.060.698.497.587.468.668.216.682.579.453.249.454.553.476.7
R50x3 86.996.786.275.774.660.654.287.778.593.295.399.498.664.699.398.088.169.959.619.683.483.557.851.355.855.680.7
R101x1 85.595.784.473.072.559.855.087.378.192.295.099.598.162.599.097.687.868.767.718.084.082.355.953.454.853.179.4
R101x3 87.297.487.572.475.057.447.487.579.693.295.499.698.664.399.498.287.768.864.120.780.484.058.752.654.954.381.2
R152x2 88.097.587.875.875.961.555.388.179.893.695.999.598.564.399.597.989.070.070.320.782.685.559.650.854.955.181.9
R152x4 87.297.688.272.475.049.143.487.179.992.495.499.398.565.799.597.887.768.257.120.680.484.659.049.757.255.181.5
TiV
B/32 81.896.786.365.270.749.142.785.373.190.494.598.797.859.099.096.383.068.165.115.782.679.151.738.957.154.676.6
B/16 86.796.986.474.074.254.746.086.774.392.794.199.297.461.399.596.484.563.161.517.585.482.756.640.057.056.180.9
L/16 87.497.989.076.574.962.552.286.175.092.994.799.398.064.099.696.585.770.458.817.785.784.158.038.458.452.881.9
H/14 83.495.884.570.269.262.354.884.775.491.793.798.998.562.498.497.387.073.963.415.487.079.452.141.155.954.175.9
2vRLCmiS
R50x1 76.493.277.948.664.156.351.784.477.088.391.892.997.659.796.797.585.871.169.115.884.878.451.056.253.953.873.8
R50x3 81.095.682.456.567.065.661.185.978.890.994.195.498.762.698.297.988.278.274.717.685.482.654.655.454.255.277.3
R101x1 77.994.879.951.965.257.152.085.477.290.091.692.797.259.497.696.884.665.770.616.184.378.852.453.655.155.776.1
R101x3 82.296.483.457.568.264.660.086.278.991.895.095.498.463.098.597.988.077.569.118.385.582.955.952.254.556.378.8
R152x1 78.695.079.950.365.655.652.285.877.390.192.591.897.659.898.196.684.364.870.316.683.979.453.157.255.854.876.9
R152x2 82.396.783.958.168.564.958.786.679.192.294.196.098.264.198.598.088.177.069.818.485.382.756.253.656.056.579.2
R152x3 83.696.884.560.369.168.563.186.780.592.694.996.398.765.498.898.189.578.468.519.485.283.557.054.454.654.280.0
LOYB 50x1 74.093.679.147.663.761.662.382.677.088.393.794.398.758.896.497.688.280.171.414.184.877.349.356.153.854.473.3
200x2 78.596.283.353.468.561.755.486.677.491.995.593.998.762.698.697.787.477.176.416.484.082.655.154.152.552.479.2
oCoM v1 65.985.063.127.552.635.943.575.770.070.478.185.497.654.385.697.182.962.660.212.685.764.240.754.755.653.557.2
v2 72.293.476.339.660.248.351.182.675.184.489.990.798.458.395.797.285.475.775.413.285.672.747.856.953.953.869.1
VirTex 57.983.957.517.049.822.434.583.858.253.670.674.798.156.586.794.874.169.571.3 8.7 83.161.539.945.553.555.850.7
teNseR 50 71.391.874.552.760.549.948.583.872.392.490.890.898.354.996.496.783.670.667.111.782.571.246.843.056.555.574.3
101 72.793.077.253.760.850.147.084.471.692.391.990.498.556.697.097.183.472.563.611.983.372.748.343.253.054.775.8
152 73.793.578.055.161.652.848.484.571.993.092.189.698.257.097.697.083.170.170.212.382.975.349.242.453.253.977.1
Table10.Linearprobeperformanceofvariouspre-trainedmodelsover27datasets.Scoreswithinthe99.5%Clopper-Pearsonconfidence
intervalofeachdataset’stopscoreareshowninbold.
(cid:63)WeupdatedtheSTL10scoresfromthepreviousversionofthispaperafterfixingaCUDA-relatedbug.

41
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|          |     | Food101 |          | CIFAR10 |     |          | CIFAR100 |     |             | Birdsnap |     |
| -------- | --- | ------- | -------- | ------- | --- | -------- | -------- | --- | ----------- | -------- | --- |
|          | 95  |         |          |         |     | 90       |          |     | 80          |          |     |
|          |     |         | 98       |         |     |          |          |     | 75          |          |     |
|          | 90  |         |          |         |     |          |          |     | 70          |          |     |
|          |     |         | 96       |         |     | 85       |          |     |             |          |     |
| ycarucca |     |         | ycarucca |         |     | ycarucca |          |     | ycarucca 65 |          |     |
|          | 85  |         | 94       |         |     |          |          |     | 60          |          |     |
80
|     | 80  |     |     |     |     |     |     |     | 55  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | 92  |     |     |     |     |     | 50  |     |     |
75
|     | 75  |         | 90  |              |     |     |              |     | 45  |               |     |
| --- | --- | ------- | --- | ------------ | --- | --- | ------------ | --- | --- | ------------- | --- |
|     |     |         |     |              |     | 70  |              |     | 40  |               |     |
|     | 100 | 101 102 | 100 | 101          | 102 | 100 | 101          | 102 |     | 100 101       | 102 |
|     |     | SUN397  |     | StanfordCars |     |     | FGVCAircraft |     |     | PascalVOC2007 |     |
90
|     | 80  |     | 90  |     |     | 70  |     |     | sessalc 02 revo PAm tniop-11 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- |
89
65
|          | 75  |     | 80       |     |     | ssalc rep naem |     |     | 88  |     |     |
| -------- | --- | --- | -------- | --- | --- | -------------- | --- | --- | --- | --- | --- |
| ycarucca |     |     | ycarucca |     |     |                |     |     | 87  |     |     |
60
|     | 70  |     | 70  |     |     |     |     |     | 86  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
55
85
|     | 65  |     | 60  |     |     | 50  |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
84
45
|     | 60  |                     | 50             |            |     |                |            |     | 83             |            |     |
| --- | --- | ------------------- | -------------- | ---------- | --- | -------------- | ---------- | --- | -------------- | ---------- | --- |
|     | 100 | 101 102             | 100            | 101        | 102 | 100            | 101        | 102 |                | 100 101    | 102 |
|     |     | DescribableTextures |                | OxfordPets |     |                | Caltech101 |     |                | Flowers102 |     |
|     |     |                     | 96             |            |     |                |            |     | 100            |            |     |
|     | 82  |                     |                |            |     | 96             |            |     |                |            |     |
|     |     |                     | 94             |            |     | 95             |            |     | 98             |            |     |
|     | 80  |                     | ssalc rep naem |            |     | ssalc-rep-naem |            |     | ssalc rep naem |            |     |
|     |     |                     | 92             |            |     | 94             |            |     | 96             |            |     |
ycarucca 78
|     |     |     | 90  |     |     | 93  |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 76  |     |     |     |     |     |     |     | 94  |     |     |
92
88
|     | 74  |     |     |     |     | 91  |     |     | 92  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
86
|     | 72  |     |     |     |     | 90  |     |     | 90  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
84
|     | 100 | 101 102 | 100                          | 101 | 102 | 100 | 101   | 102 |     | 100 101 | 102 |
| --- | --- | ------- | ---------------------------- | --- | --- | --- | ----- | --- | --- | ------- | --- |
|     |     | MNIST   | FacialEmotionRecognition2013 |     |     |     | STL10 |     |     | EuroSAT |     |
72.5
| 99.00    |     |     |          |     |     | 99.5     |     |     | 98.0          |     |     |
| -------- | --- | --- | -------- | --- | --- | -------- | --- | --- | ------------- | --- | --- |
|          |     |     | 70.0     |     |     | 99.0     |     |     |               |     |     |
| 98.75    |     |     |          |     |     |          |     |     | 97.5          |     |     |
| 98.50    |     |     | 67.5     |     |     | 98.5     |     |     |               |     |     |
| ycarucca |     |     | ycarucca |     |     | ycarucca |     |     | ycarucca 97.0 |     |     |
| 98.25    |     |     | 65.0     |     |     | 98.0     |     |     |               |     |     |
|          |     |     |          |     |     | 97.5     |     |     | 96.5          |     |     |
| 98.00    |     |     | 62.5     |     |     |          |     |     |               |     |     |
97.0
| 97.75 |     |          | 60.0 |       |     |      |       |     | 96.0 |               |     |
| ----- | --- | -------- | ---- | ----- | --- | ---- | ----- | --- | ---- | ------------- | --- |
| 97.50 |     |          | 57.5 |       |     | 96.5 |       |     |      |               |     |
|       |     |          |      |       |     | 96.0 |       |     | 95.5 |               |     |
| 97.25 |     |          | 55.0 |       |     |      |       |     |      |               |     |
|       | 100 | 101 102  | 100  | 101   | 102 | 100  | 101   | 102 |      | 100 101       | 102 |
|       |     | RESISC45 |      | GTSRB |     |      | KITTI |     |      | PatchCamelyon |     |
87
|          | 94  |     |          |     |     | 75.0          |     |     |          |     |     |
| -------- | --- | --- | -------- | --- | --- | ------------- | --- | --- | -------- | --- | --- |
|          |     |     | 90       |     |     |               |     |     | 86       |     |     |
|          | 92  |     |          |     |     | 72.5          |     |     |          |     |     |
|          |     |     | 85       |     |     |               |     |     | 85       |     |     |
| ycarucca |     |     | ycarucca |     |     | ycarucca 70.0 |     |     | ycarucca |     |     |
|          | 90  |     | 80       |     |     |               |     |     | 84       |     |     |
67.5
|     | 88  |     | 75  |     |     | 65.0 |     |     | 83  |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
|     | 86  |     | 70  |     |     | 62.5 |     |     | 82  |     |     |
|     | 84  |     | 65  |     |     | 60.0 |     |     | 81  |     |     |
57.5
82
|     | 100 | 101 102 | 100 | 101         | 102 | 100 | 101         | 102 |     | 100 101    | 102 |
| --- | --- | ------- | --- | ----------- | --- | --- | ----------- | --- | --- | ---------- | --- |
|     |     | UCF101  |     | Kinetics700 |     |     | CLEVRCounts |     |     | Country211 |     |
45
|     | 90  |     | 70  |     |     | 60  |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
40
)5pot ,1pot(naem
|          | 85  |     | 65  |     |     | 55       |     |     | 35       |     |     |
| -------- | --- | --- | --- | --- | --- | -------- | --- | --- | -------- | --- | --- |
| ycarucca |     |     |     |     |     | ycarucca |     |     | ycarucca |     |     |
|          |     |     | 60  |     |     |          |     |     | 30       |     |     |
50
|     | 80  |     |     |     |     |     |     |     | 25  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | 55  |     |     | 45  |     |     |     |     |     |
20
|     | 75  |              | 50  |      |     |      |          |     |     |              |     |
| --- | --- | ------------ | --- | ---- | --- | ---- | -------- | --- | --- | ------------ | --- |
|     |     |              |     |      |     | 40   |          |     | 15  |              |     |
|     | 100 | 101 102      | 100 | 101  | 102 | 100  | 101      | 102 | 10  | 100 101      | 102 |
|     |     | HatefulMemes |     | SST2 |     |      | ImageNet |     |     | GFLOPs/image |     |
|     | 75  |              | 80  |      |     | 87.5 |          |     |     | CLIP-ViT     |     |
CLIP-ResNet
|        |     |              | 75          |              |     | 85.0     |              |     |     | EfficientNet-NoisyStudent |     |
| ------ | --- | ------------ | ----------- | ------------ | --- | -------- | ------------ | --- | --- | ------------------------- | --- |
|        | 70  |              |             |              |     | 82.5     |              |     |     |                           |     |
| CUACOR |     |              | ycarucca 70 |              |     | ycarucca |              |     |     | EfficientNet              |     |
|        |     |              |             |              |     | 80.0     |              |     |     | Instagram-pretrained      |     |
|        | 65  |              |             |              |     |          |              |     |     | SimCLRv2                  |     |
|        |     |              | 65          |              |     | 77.5     |              |     |     | BYOL                      |     |
|        | 60  |              |             |              |     | 75.0     |              |     |     |                           |     |
|        |     |              | 60          |              |     |          |              |     |     | MoCo                      |     |
|        |     |              |             |              |     | 72.5     |              |     |     | ViT (ImageNet-21k)        |     |
|        | 55  |              | 55          |              |     |          |              |     |     | BiT-M                     |     |
|        |     |              |             |              |     | 70.0     |              |     |     | BiT-S                     |     |
|        | 100 | 101 102      | 100         | 101          | 102 | 100      | 101          | 102 |     | ResNet                    |     |
|        |     | GFLOPs/image |             | GFLOPs/image |     |          | GFLOPs/image |     |     |                           |     |
Figure20. Linearprobeperformanceplottedforeachofthe27datasets,usingthedatafromTable10.

LearningTransferableVisualModelsFromNaturalLanguageSupervision 42
Food101 SUN397 Youtube-BB EuroSAT
correct label: guacamole correct rank: 1/101 correct probability: 90.15% correct label: television studio correct rank: 1/397 correct probability: 90.22% correct label(s): airplane,person correct rank: 1/23 correct probability: 88.98% correct label: annual crop land correct rank: 4/10 correct probability: 12.90%
a photo of guacamole, a type of food. a photo of a television studio. a photo of a airplane. a centered satellite photo of permanent crop land.
a photo of ceviche, a type of food. a photo of a podium indoor. a photo of a bird. a centered satellite photo of pasture land.
a photo of edamame, a type of food. a photo of a conference room. a photo of a bear. a centered satellite photo of highway or road.
a photo of tuna tartare, a type of food. a photo of a lecture room. a photo of a giraffe. a centered satellite photo of annual crop land.
a photo of hummus, a type of food. a photo of a control room. a photo of a car. a centered satellite photo of brushland or shrubland.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
PatchCamelyon (PCam) ImageNet-A (Adversarial) CIFAR-10 CLEVR Count
correct label: healthy lymph node tissue correct rank: 2/2 correct probability: 22.81% correct label: lynx correct rank: 5/200 correct probability: 4.18% correct label: bird correct rank: 1/10 correct probability: 40.86% correct label: 4 correct rank: 2/8 correct probability: 17.11%
this is a photo of lymph node tumor tissue a photo of a fox squirrel. a photo of a bird. a photo of 3 objects.
this is a photo of healthy lymph node tissue a photo of a mongoose. a photo of a cat. a photo of 4 objects.
a photo of a skunk. a photo of a deer. a photo of 5 objects.
a photo of a red fox. a photo of a frog. a photo of 6 objects.
a photo of a lynx. a photo of a dog. a photo of 10 objects.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
Facial Emotion Recognition 2013 (FER2013) UCF101 Caltech-101 ImageNet-R (Rendition)
correct label: angry correct rank: 5/7 correct probability: 8.16% correct label: Volleyball Spiking correct rank: 1/101 correct probability: 99.30% correct label: kangaroo correct rank: 1/102 correct probability: 99.81% correct label: Siberian Husky correct rank: 1/200 correct probability: 76.02%
a photo of a happy looking face. a photo of a person volleyball spiking. a photo of a kangaroo. a photo of a siberian husky.
a photo of a neutral looking face. a photo of a person jump rope. a photo of a gerenuk. a photo of a german shepherd dog.
a photo of a surprised looking face. a photo of a person long jump. a photo of a emu. a photo of a collie.
a photo of a fearful looking face. a photo of a person soccer penalty. a photo of a wild cat. a photo of a border collie.
a photo of a angry looking face. a photo of a person table tennis shot. a photo of a scorpion. a photo of a rottweiler.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
Oxford-IIIT Pets CIFAR-100 ImageNetV2 Matched Frequency FGVC Aircraft
correct label: Maine Coon correct rank: 1/37 correct probability: 99.99% correct label: snake correct rank: 1/100 correct probability: 38.02% correct label: beer bottle correct rank: 1/1000 correct probability: 88.27% correct label: Boeing 717 correct rank: 2/100 correct probability: 9.91%
a photo of a maine coon, a type of pet. a photo of a snake. a photo of a beer bottle. a photo of a mcdonnell douglas md-90, a type of aircraft.
a photo of a persian, a type of pet. a photo of a sweet pepper. a photo of a pirate ship. a photo of a boeing 717, a type of aircraft.
a photo of a ragdoll, a type of pet. a photo of a flatfish. a photo of a chocolate syrup. a photo of a fokker 100, a type of aircraft.
a photo of a birman, a type of pet. a photo of a turtle. a photo of a product packet / packaging. a photo of a mcdonnell douglas dc-9-30, a type of aircraft.
a photo of a siamese, a type of pet. a photo of a lizard. a photo of a wine bottle. a photo of a boeing 727-200, a type of aircraft.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
Country211 RESISC45 Stanford Cars SUN
correct label: Belize correct rank: 5/211 correct probability: 3.92% correct label: roundabout correct rank: 1/45 correct probability: 96.39% correct label: 2012 Honda Accord Coupecorrect rank: 1/196 correct probability: 63.30% correct label: kennel indoor correct rank: 1/723 correct probability: 98.63%
a photo i took in french guiana. satellite imagery of roundabout. a photo of a 2012 honda accord coupe. a photo of a kennel indoor.
a photo i took in gabon. satellite imagery of intersection. a photo of a 2012 honda accord sedan. a photo of a kennel outdoor.
a photo i took in cambodia. satellite imagery of church. a photo of a 2012 acura tl sedan. a photo of a jail cell.
a photo i took in guyana. satellite imagery of medium residential. a photo of a 2012 acura tsx sedan. a photo of a jail indoor.
a photo i took in belize. satellite imagery of chaparral. a photo of a 2008 acura tl type-s. a photo of a veterinarians office.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
Kinetics-700 Flowers-102 ImageNet Birdsnap
correct label: country line dancing correct rank: 1/700 correct probability: 98.98% correct label: great masterwort correct rank: 1/102 correct probability: 74.25% correct label: King Charles Spaniel correct rank: 1/1000 correct probability: 91.61% correct label: Black chinned Hummingbirdcorrect rank: 4/500 correct probability: 12.00%
a photo of country line dancing. a photo of a great masterwort, a type of flower. a photo of a king charles spaniel. a photo of a broad tailed hummingbird, a type of bird.
a photo of square dancing. a photo of a bishop of llandaff, a type of flower. a photo of a brittany dog. a photo of a calliope hummingbird, a type of bird.
a photo of swing dancing. a photo of a pincushion flower, a type of flower. a photo of a cocker spaniel. a photo of a costas hummingbird, a type of bird.
a photo of dancing charleston. a photo of a globe flower, a type of flower. a photo of a papillon. a photo of a black chinned hummingbird, a type of bird.
a photo of salsa dancing. a photo of a prince of wales feathers, a type of flower. a photo of a sussex spaniel. a photo of a annas hummingbird, a type of bird.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
aYahoo ObjectNet ImageNet Overlap ImageNet Blurry Describable Textures Dataset (DTD)
correct label: building correct rank: 1/12 correct probability: 97.69% correct label: Pill bottle correct rank: 1/113 correct probability: 98.34% correct label: marimba correct rank: 1/1000 correct probability: 79.54% correct label: perforated correct rank: 2/47 correct probability: 20.50%
a photo of a building. a photo of a pill bottle. a photo of a marimba. a photo of a polka-dotted texture.
a photo of a carriage. a photo of a bottle cap. a photo of a abacus. a photo of a perforated texture.
a photo of a statue. a photo of a beer bottle. a photo of a steel drum. a photo of a dotted texture.
a photo of a bag. a photo of a pillow. a photo of a computer keyboard. a photo of a studded texture.
a photo of a mug. a photo of a wine bottle. a photo of a pool table. a photo of a freckled texture.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
PASCAL VOC 2007 MNIST Street View House Numbers (SVHN) ImageNet Vid
correct label(s): motorcycle correct rank: 1/20 correct probability: 99.69% correct label: 7 correct rank: 1/10 correct probability: 85.32% correct label: 158 correct rank: 83/2000 correct probability: 0.27% correct label(s): antelope correct rank: 1/30 correct probability: 99.77%
a photo of a motorcycle. a photo of the number: "7". a street sign of the number: "1157". a photo of a antelope.
a photo of a bicycle. a photo of the number: "2". a street sign of the number: "1165". a photo of a zebra.
a photo of a car. a photo of the number: "1". a street sign of the number: "1164". a photo of a car.
a photo of a horse. a photo of the number: "6". a street sign of the number: "1155". a photo of a cattle.
a photo of a dining table. a photo of the number: "4". a street sign of the number: "1364". a photo of a elephant.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
ImageNet Sketch Hateful Memes Stanford Sentiment Treebank German Traffic Sign Recognition Benchmark (GTSRB)
correct label: barn correct rank: 1/1000 correct probability: 79.56% correct label: meme correct rank: 1/2 correct probability: 99.20% correct label: positive correct rank: 1/2 correct probability: 78.21% correct label: red and white triangle with exclamation mark warning correct rank: 1/43 correct probability: 45.75%
a photo of a barn. a meme. a positive review of a movie. a zoomed in photo of a "red and white triangle with exclamation mark warning" traffic sign.
a photo of a church. a hatespeech meme. a negative review of a movie. a zoomed in photo of a "red and white triangle with black right curve approaching warning" traffic sign.
a photo of a threshing machine. a zoomed in photo of a "red and white triangle car skidding / slipping warning" traffic sign.
a photo of a sawmill. a zoomed in photo of a "red and white triangle rough / bumpy road warning" traffic sign.
a photo of a prison. a zoomed in photo of a "red and white triangle with black left curve approaching warning" traffic sign.
0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100 0 20 40 60 80 100
Figure21.Visualizationofpredictionsfrom36CLIPzero-shotclassifiers.Allexamplesarerandomwiththeexceptionofreselecting
HatefulMemestoavoidoffensivecontent.Thepredictedprobabilityofthetop5classesisshownalongwiththetextusedtorepresent
theclass.Whenmorethanonetemplateisused,thefirsttemplateisshown.Thegroundtruthlabeliscoloredgreenwhileanincorrect
predictioniscoloredorange.

43
LearningTransferableVisualModelsFromNaturalLanguageSupervision
2TSSderedneR
|     |     |     | sraCdrofnatS | tfarcriACVGF |                    |            |     |          |            |             | semeMlufetaH |          |
| --- | --- | --- | ------------ | ------------ | ------------------ | ---------- | --- | -------- | ---------- | ----------- | ------------ | -------- |
|     |     |     |              |              | stePdrofxO         | 201srewolF |     |          | 112yrtnuoC | 007sciteniK |              |          |
|     |     |     | 001RAFIC     |              | 7002COV 101hcetlaC |            |     | 54CSISER |            |             |              | teNegamI |
101dooF 01RAFIC pansdriB 793NUS 3102REF TASoruE BRSTG 101FCU RVELC
|     |     |     |     |     |     | TSINM | 01LTS |     | ITTIK |     |     |     |
| --- | --- | --- | --- | --- | --- | ----- | ----- | --- | ----- | --- | --- | --- |
|     |     |     |     |     | DTD |       |       |     | maCP  |     |     |     |
RN50 81.175.641.632.659.655.819.382.141.785.482.165.966.642.294.341.154.235.242.216.157.663.643.520.359.756.959.6
teNseR-PILC
RN101 83.981.049.037.259.962.319.582.443.986.285.165.759.345.696.733.158.538.333.316.955.262.246.728.161.164.262.2
RN50x4 86.879.248.941.662.767.924.683.049.388.186.068.075.251.196.435.059.235.726.020.257.565.549.017.058.366.665.8
RN50x16 90.582.254.245.965.072.330.382.952.889.787.671.980.056.097.840.364.439.633.924.062.568.753.417.658.967.670.5
RN50x64 91.886.861.348.966.976.035.683.853.493.490.677.390.861.098.359.469.747.933.229.665.074.156.827.562.170.773.6
B/32 84.491.365.137.863.259.421.283.144.587.087.966.751.947.397.249.460.332.239.417.858.464.547.824.857.659.663.2
TiV-PILC
B/16 89.291.668.739.165.265.627.183.946.088.989.370.456.052.798.254.165.543.344.023.348.169.852.423.461.759.868.6
L/14 92.996.277.948.367.777.336.184.155.393.592.678.787.257.599.359.971.650.323.132.758.876.260.324.363.364.075.3
L/14-336px 93.895.777.549.568.478.837.284.355.793.592.878.388.357.799.459.671.752.321.934.963.076.961.324.863.367.976.2
|     |         |     | Table11. | Zero-shotperformanceofCLIPmodelsover27datasets. |          |     |          |     |        |     |              |     |
| --- | ------- | --- | -------- | ----------------------------------------------- | -------- | --- | -------- | --- | ------ | --- | ------------ | --- |
|     | Food101 |     | CIFAR10  |                                                 | CIFAR100 |     | Birdsnap |     | SUN397 |     | StanfordCars |     |
|     |         |     |          |                                                 |          |     | 55       |     |        |     | 80           |     |
|     |         | 95  |          |                                                 |          |     |          |     | 68     |     |              |     |
|     | 90      |     |          | 70                                              |          |     |          |     |        |     |              |     |
ycarucca ycarucca 90 ycarucca ycarucca 50 ycarucca 66 ycarucca 70
85
|     |         |     |         | 60  |         |     | 45  |     | 64      |     |     |     |
| --- | ------- | --- | ------- | --- | ------- | --- | --- | --- | ------- | --- | --- | --- |
|     | 80      | 85  |         |     |         |     |     |     |         |     |     |     |
|     |         |     |         |     |         |     | 40  |     |         |     | 60  |     |
|     |         | 80  |         | 50  |         |     |     |     | 62      |     |     |     |
|     | 75      |     |         |     |         |     | 35  |     |         |     |     |     |
|     |         |     |         |     |         |     |     |     | 60      |     | 50  |     |
|     |         | 75  |         | 40  |         |     |     |     |         |     |     |     |
|     | 101 102 |     | 101 102 |     | 101 102 |     | 101 | 102 | 101 102 |     | 101 | 102 |
FGVCAircraft sessalc 02 revo PAm tniop-11 PascalVOC2007 DescribableTextures OxfordPets Caltech101 Flowers102
|     |     | 84.5 |     |     |     |     |     |     |     |     | 90  |     |
| --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |      |     | 70  |     |     |     |     | 92  |     |     |     |
ssalc rep naem 84.0 ssalc rep naem 92 ssalc-rep-naem 90 ssalc rep naem 85
|     | 40  |      |     | ycarucca |     |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | 83.5 |     | 60       |     |     | 90  |     | 88  |     | 80  |     |
|     | 30  | 83.0 |     |          |     |     |     |     | 86  |     | 75  |     |
|     |     |      |     | 50       |     |     | 88  |     |     |     |     |     |
|     |     | 82.5 |     |          |     |     |     |     | 84  |     | 70  |     |
86
|     | 20      | 82.0 |         |     |         |     |     |     | 82      |     | 65  |     |
| --- | ------- | ---- | ------- | --- | ------- | --- | --- | --- | ------- | --- | --- | --- |
|     | 101 102 |      | 101 102 |     | 101 102 |     | 101 | 102 | 101 102 |     | 101 | 102 |
MNIST FacialEmotionRecognition2013 STL10 EuroSAT RESISC45 GTSRB
| 100      |     |             |     |          |     | 100      |     |          |     |          |     |     |
| -------- | --- | ----------- | --- | -------- | --- | -------- | --- | -------- | --- | -------- | --- | --- |
|          |     | 60          |     | 99       |     |          |     |          |     |          | 70  |     |
|          | 90  |             |     |          |     |          |     |          | 80  |          |     |     |
|          |     |             |     | 98       |     |          | 80  |          |     |          |     |     |
| ycarucca |     | ycarucca 55 |     | ycarucca |     | ycarucca |     | ycarucca |     | ycarucca | 60  |     |
|          | 80  |             |     | 97       |     |          |     |          | 70  |          |     |     |
|          |     | 50          |     |          |     |          | 60  |          |     |          | 50  |     |
|          | 70  |             |     | 96       |     |          |     |          |     |          |     |     |
|          | 60  | 45          |     | 95       |     |          |     |          | 60  |          | 40  |     |
40
50
|     | 101 102 |     | 101 102 |     | 101 102 |     | 101 | 102 | 101 102 |     | 101 | 102 |
| --- | ------- | --- | ------- | --- | ------- | --- | --- | --- | ------- | --- | --- | --- |
KITTI PatchCamelyon UCF101 Kinetics700 CLEVRCounts Country211
|          | 70  |          |     |          |     |                  |     |          |     |          | 35  |     |
| -------- | --- | -------- | --- | -------- | --- | ---------------- | --- | -------- | --- | -------- | --- | --- |
|          |     | 80       |     | 75       |     | )5pot ,1pot(naem | 60  |          | 40  |          |     |     |
|          | 60  |          |     |          |     |                  |     |          |     |          | 30  |     |
| ycarucca |     | ycarucca |     | ycarucca |     |                  |     | ycarucca | 35  | ycarucca |     |     |
|          | 50  | 70       |     |          |     |                  | 55  |          |     |          |     |     |
|          |     |          |     | 70       |     |                  |     |          | 30  |          | 25  |     |
|          | 40  | 60       |     |          |     |                  | 50  |          | 25  |          | 20  |     |
65
|     | 30  |     |     |     |     |     |     |     | 20  |     | 15  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | 50  |     |     |     |     | 45  |     |     |     |     |     |
20
|     | 101 102      |     | 101 102 |     | 101 102  |     | 101 | 102 | 101 102 |     | 101 | 102 |
| --- | ------------ | --- | ------- | --- | -------- | --- | --- | --- | ------- | --- | --- | --- |
|     | HatefulMemes |     | SST2    |     | ImageNet |     |     |     |         |     |     |     |
70
|     | 62  |     |     | 75  |     |     |     |     | CLIP-ViT |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- |
CLIP-ResNet
| CUACOR | 60  | ycarucca 65 |     | ycarucca |     |     |     |     |        |     |     |     |
| ------ | --- | ----------- | --- | -------- | --- | --- | --- | --- | ------ | --- | --- | --- |
|        |     |             |     | 70       |     |     |     |     | ResNet |     |     |     |
58
60
|     | 56  |     |     | 65  |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | 54  | 55  |     |     |     |     |     |     |     |     |     |     |
60
|     | 101 102      |     | 101 102      |     | 101 102      |     |     |     |     |     |     |     |
| --- | ------------ | --- | ------------ | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
|     | GFLOPs/image |     | GFLOPs/image |     | GFLOPs/image |     |     |     |     |     |     |     |
Figure22. CLIP’szero-shotperformancecomparedtolinear-probeResNetperformance

44
LearningTransferableVisualModelsFromNaturalLanguageSupervision
B.Zero-ShotPrediction
|     |     |     |     |     |     |     |     |     | LinearClassifier |     |     |     | ZeroShot |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | --- | -------- | --- |
Toprovideaqualitativesummary/overviewofCLIP’szero- Dataset YFCC WIT ∆ YFCC WIT ∆
shotperformancewevisualizearandomlyselectedpredic- Birdsnap 47.4 35.3 +12.1 19.9 4.5 +15.4
tion for 36 different zero-shot CLIP classifiers in Figure Country211 23.1 17.3 +5.8 5.2 5.3 +0.1
|     |     |     |     |     |     |     | Flowers102 |     | 94.4 | 89.8 | +4.6 | 48.6 | 21.7 | +26.9 |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ---- | ---- | ---- | ---- | ---- | ----- |
21. Inaddition,Table11andFigure22showtheindividual GTSRB 66.8 72.5 −5.7 6.9 7.0 −0.1
zero-shotperformancescoresforeachdataset. UCF101 69.2 74.9 −5.7 22.9 32.0 −9.1
|     |     |     |     |     |     |     | StanfordCars |     | 31.4 | 50.3 | −18.9 | 3.8  | 10.9 | −7.1 |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ---- | ---- | ----- | ---- | ---- | ---- |
|     |     |     |     |     |     |     | ImageNet     |     | 62.0 | 60.8 | +1.2  | 31.3 | 27.6 | +3.7 |
C.DuplicateDetector DatasetAverage 65.5 66.6 −1.1 29.6 30.0 −0.4
|     |     |     |     |     |     |     | Dataset“Wins” |     | 10  | 15  | −5  |     | 19 18 | +1  |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | ----- | --- |
Ourearlyattemptsatduplicatedetectionandanalysisused
|     |     |     |     |     |     |     | Table12.CLIP |     | performs | similarly |     | when | trained | on only |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | -------- | --------- | --- | ---- | ------- | ------- |
nearestneighborsinthemodel’slearnedembeddingspace.
|     |     |     |     |     |     |     | YFCC100M. |     | Comparing | a   | ResNet-50 |     | trained | on only |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | --------- | --- | --------- | --- | ------- | ------- |
Whileitisintuitivetouseamodel’sownnotionofsimilar-
|                          |          |              |                          |          |     |          | YFCC100M          | with        | a same | sized    | subset | of          | WIT shows | simi-      |
| ------------------------ | -------- | ------------ | ------------------------ | -------- | --- | -------- | ----------------- | ----------- | ------ | -------- | ------ | ----------- | --------- | ---------- |
| ity,weencounteredissues. |          |              | Wefoundthemodel’sfeature |          |     |          |                   |             |        |          |        |             |           |            |
|                          |          |              |                          |          |     |          | lar average       | performance |        | and      | number | of wins     | on zero   | shot and   |
| space is                 | weighted | very heavily | towards                  | semantic |     | similar- |                   |             |        |          |        |             |           |            |
|                          |          |              |                          |          |     |          | linear classifier |             | evals. | However, | large  | differences |           | in dataset |
ity. Many false positives occurred due to distinct objects specific performance occur. We include performance on the 3
thatwouldbedescribedsimilarly(soccerballs,flowersof datasets where YFCC does best and worst compared to WIT
the same species, etc...) having almost perfect similarity. accordingtoalinearprobeinordertohighlightthisaswellas
We also observed the model was quite poor at assigning aggregateperformanceacrossalllinearandzero-shotevalsand
certainkindsofnear-duplicateshighsimilarityscores. We thecanonicalImageNetdataset.
noticedrepeatedlythatimageswithhigh-frequencytextures
| (such as | fur or     | stripe patterns) | pre-processed |     | by         | different |     |     |     |     |     |     |     |     |
| -------- | ---------- | ---------------- | ------------- | --- | ---------- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
| resizing | algorithms | (nearest         | neighbor      | vs  | bi-linear) | could     |     |     |     |     |     |     |     |     |
D.DatasetAblationonYFCC100M
| havesurprisinglylowsimilarity. |     |     | Thisresultedinmanyfalse |     |     |     |     |     |     |     |     |     |     |     |
| ------------------------------ | --- | --- | ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
negatives.
Tostudywhetherourcustomdatasetiscriticaltotheperfor-
manceofCLIP,wetrainedamodelonafilteredsubsetof
Webuiltourownnear-duplicatedetectortofixthisissue.
Wecreatedasyntheticdataaugmentationpipelinethatcom- theYFCC100Mdataset(detailsdescribedinSection2.2)
andcompareditsperformancetothesamemodeltrained
| binedavarietyofcommonimagemanipulations. |     |     |     |     |     | Theaug- |     |     |     |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
onanequallysizedsubsetofWIT.Wetraineachmodelfor
mentationpipelinecombinesrandomcroppingandzooming,
aspectratiodistortion,downsizingandupscalingtodifferent 32 epochs at which point transfer performance begins to
|              |       |            |      |              |     |         | plateauduetooverfitting. |     |     |     | ResultsareshowninTable12. |     |     |     |
| ------------ | ----- | ---------- | ---- | ------------ | --- | ------- | ------------------------ | --- | --- | --- | ------------------------- | --- | --- | --- |
| resolutions, | minor | rotations, | jpeg | compression, |     | and HSV |                          |     |     |     |                           |     |     |     |
colorjitter. Thepipelinealsorandomlyselectsfromdiffer- Acrossourwholeevalsuite,YFCCandWITperformsimi-
larlyonaverageforbothzero-shotandlinearprobesettings.
| entinterpolationalgorithmsforallrelevantsteps. |     |     |     |     |     | Wethen |          |                                              |     |     |     |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | ------ | -------- | -------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                                                |     |     |     |     |     |        | However, | performanceonspecificfine-grainedclassifica- |     |     |     |     |     |     |
trainedamodeltomaximizethesimilarityofanimageand
itstransformedvariantwhileminimizingsimilaritytoall tion datasets can vary widely - sometimes by over 10%.
Ourspeculationisthatthesedifferencesinperformancere-
| otherimagesinatrainingbatch. |     |     | Weusedthesamen-pair/ |     |     |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
InfoNCElossasCLIPbutwithafixedtemperatureof0.07. flecttherelativedensityofrelevantdataineachpre-training
dataset. Forinstance,pre-trainingonYFCC100M,which
We selected a ResNet-50 as the model architecture. We mightcontainmanyphotosofbirdsandflowers(common
modifiedthebaseResNet-50withtheanti-aliasimprove-
subjectsforphotographers),resultsinbetterperformanceon
ments from (Zhang, 2019) and used weight norm (Sali- BirdsnapandFlowers102,whilepre-trainingonWITresults
| mans & | Kingma, | 2016) | instead of | batch | norm | (Ioffe & |     |     |     |     |     |     |     |     |
| ------ | ------- | ----- | ---------- | ----- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
inbettercarandpetclassifiers(whichappearcommonin
Szegedy,2015)toavoidleakinginformationaboutdupli-
ourdataset).
| cates via | batch | statistics | - a problem | previously |     | noted in |     |     |     |     |     |     |     |     |
| --------- | ----- | ---------- | ----------- | ---------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
(Henaff,2020). WealsofoundtheGELUactivationfunc- Overall,theseresultsareencouragingastheysuggestour
approachcanuseanyreasonablyfilteredcollectionofpaired
tion(Hendrycks&Gimpel,2016)toperformbetterforthis
|     |     |     |     |     |     |     | (text,image)data. |     | Thismirrorsrecentworkwhichreported |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ---------------------------------- | --- | --- | --- | --- | --- |
task. Wetrainedthemodelwithatotalbatchsizeof1,712
forapproximately30millionimagessampledfromourpre- positiveresultsusingthesamecontrastivepre-trainingob-
jectiveontherelativelydifferentdomainofmedicalimaging
| training | dataset. | At the | end of training | it  | achieves | nearly |     |     |     |     |     |     |     |     |
| -------- | -------- | ------ | --------------- | --- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
(Zhangetal.,2020).Italsoissimilartothefindingsofnoisy
100%accuracyonitsproxytrainingtask.
|     |     |     |     |     |     |     | student | self-training |     | which | reported | only | slight | improve- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------- | --- | ----- | -------- | ---- | ------ | -------- |
mentswhenusingtheirJFT300MdatasetoverYFCC100M
|     |     |     |     |     |     |     | (Xieetal.,2020). |     | Wesuspectthemajoradvantageofour |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------------------------------- | --- | --- | --- | --- | --- |
datasetoverthealreadyexistingYFCC100Misitsmuch
largersize.

45
LearningTransferableVisualModelsFromNaturalLanguageSupervision
Finally, we caution that WIT includes this filtered subset on5datasetsrequiringthedirectandindirectuseofOCR.
of YFCC100M. This could result in our ablation under- Three of these datasets MNIST (LeCun), SVHN (Netzer
estimating the size of performance differences between etal.,2011),andIIIT5K(Mishraetal.,2012)directlycheck
YFCC100M and the rest of WIT. We do not think this is theabilityofamodeltoperformlow-levelcharacterand
likelyasYFCC100Misonly3.7%oftheoverallWITdata wordrecognition,whileHatefulMemes(Kielaetal.,2020)
blendanditdidnotnoticeablychangetheperformanceof andSST-2(Socheretal.,2013)checktheabilityofamodel
modelswhenitwasaddedtotheexistingdatablendduring touseOCRtoperformasemantictask. Resultsarereported
| thecreationofWIT. |     |     |     | inTable14. |     |     |     |     |
| ----------------- | --- | --- | --- | ---------- | --- | --- | --- | --- |
CLIP’sperformanceisstillhighlyvariableandappearsto
E.SelectedTaskandDatasetResults
besensitivetosomecombinationofthedomain(renderedor
naturalimages)andthetypeoftexttoberecognized(num-
Duetothelargevarietyofdatasetsandexperimentsconsid-
|     |     |     |     | bersorwords). | CLIP’sOCRperformanceisstrongestHate- |     |     |     |
| --- | --- | --- | --- | ------------- | ------------------------------------ | --- | --- | --- |
eredinthiswork,themainbodyfocusesonsummarizing
fulMemesandSST-2-datasetswherethetextisdigitally
| andanalyzingoverallresults. |     | Inthefollowingsubsections |     |                                   |     |     |                |     |
| --------------------------- | --- | ------------------------- | --- | --------------------------------- | --- | --- | -------------- | --- |
|                             |     |                           |     | renderedandconsistsmostlyofwords. |     |     | OnIIIT5K,which |     |
wereportdetailsofperformanceforspecificgroupsoftasks,
isnaturalimagesofindividuallycroppedwords,zero-shot
datasets,andevaluationsettings.
CLIPperformsabitmorerespectivelyanditsperformance
issimilartoJaderbergetal.(2014)earlyworkcombining
E.1.ImageandTextRetrieval
|     |     |     |     | deep learning | and structured | prediction | to perform | open- |
| --- | --- | --- | --- | ------------- | -------------- | ---------- | ---------- | ----- |
CLIPpre-trainsforthetaskofimage-textretrievalonour vocabularyOCR.However,performanceisnoticeablylower
ontwodatasetsinvolvingrecognitionofhandwrittenand
| noisyweb-scaledataset. |     | Althoughthefocusofthispaper |     |     |     |     |     |     |
| ---------------------- | --- | --------------------------- | --- | --- | --- | --- | --- | --- |
isonrepresentationlearningandtasklearningforthepur- streetviewnumbers. CLIP’s51%accuracyonfullnumber
|     |     |     |     | SVHNiswellbelowanypublishedresults. |     |     | Inspectionsug- |     |
| --- | --- | --- | --- | ----------------------------------- | --- | --- | -------------- | --- |
poseoftransfertoawidevarietyofdownstreamdatasets,
validatingthatCLIPisabletoachievehightransferperfor- gestsCLIPstruggleswithrepeatedcharactersaswellasthe
mance transfer on exactly what it is pre-trained for is an lowresolutionandblurryimagesofSVHN.CLIP’szero-
shotMNISTperformanceisalsopoorandisoutperformed
| importantsanitycheck/proofofconcept. |     |     | InTable13we |     |     |     |     |     |
| ------------------------------------ | --- | --- | ----------- | --- | --- | --- | --- | --- |
checkthezero-shottransferperformanceofCLIPforboth bysupervisedlogisticregressiononrawpixels,oneofthe
simplestpossiblemachinelearningbaselines.
| text and | image retrieval | on the Flickr30k | and MSCOCO |     |     |     |     |     |
| -------- | --------------- | ---------------- | ---------- | --- | --- | --- | --- | --- |
datsets. Zero-shotCLIPmatchesoroutperformsallprior
SST-2isasentencelevelNLPdatasetwhichwerenderinto
| zero-shotresultsonthesetwodatasets. |     |     | Zero-shotCLIPis |         |                                         |     |     |     |
| ----------------------------------- | --- | --- | --------------- | ------- | --------------------------------------- | --- | --- | --- |
|                                     |     |     |                 | images. | WeincludeSST-2inordertocheckwhetherCLIP |     |     |     |
alsocompetitivewiththecurrentoverallSOTAforthetask
|                             |     |                         |     | is able to           | convert low                          | level OCR capability | into | a higher |
| --------------------------- | --- | ----------------------- | --- | -------------------- | ------------------------------------ | -------------------- | ---- | -------- |
| oftextretrievalonFlickr30k. |     | Onimageretrieval,CLIP’s |     |                      |                                      |                      |      |          |
|                             |     |                         |     | levelrepresentation. | FittingalinearclassifieronCLIP’srep- |                      |      |          |
performancerelativetotheoverallstateoftheartisnotice- resentationofrenderedsentencesachives80.5%accuracy.
| ably lower. | However, zero-shot | CLIP | is still competitive |     |     |     |     |     |
| ----------- | ------------------ | ---- | -------------------- | --- | --- | --- | --- | --- |
Thisisonparwiththe80%accuracyofacontinuousbag
withafine-tunedUnicoder-VL.OnthelargerMS-COCO
ofwordsbaselineusingGloVewordvectorspre-trainedon
datasetfine-tuningimprovesperformancesignificantlyand
|     |     |     |     | 840billiontokens(Penningtonetal.,2014). |     |     | Whilethisisa |     |
| --- | --- | --- | --- | --------------------------------------- | --- | --- | ------------ | --- |
zero-shotCLIPisnotcompetitivewiththemostrecentwork.
simpleNLPbaselinebytoday’sstandard,andwellbelow
| Forboththesedatasetsweprependtheprompt“a |     |     |     | photo     |                |                         |     |        |
| ---------------------------------------- | --- | --- | --- | --------- | -------------- | ----------------------- | --- | ------ |
|                                          |     |     |     | the 97.5% | of the current | SOTA, it is encouraging |     | to see |
of”tothedescriptionofeachimagewhichwefoundboosts
thatCLIPisabletoturnanimageofrenderedtextintoa
CLIP’szero-shotR@1performancebetween1and2points.
|     |     |     |     | non-trivialsentencelevelrepresentation. |                   |                   | Fullysupervised |        |
| --- | --- | --- | --- | --------------------------------------- | ----------------- | ----------------- | --------------- | ------ |
|     |     |     |     | CLIP is                                 | also surprisingly | strong on Hateful | Meme            | detec- |
E.2.OpticalCharacterRecognition
tion,whereCLIPisonly0.7pointsbehindthecurrentsingle
modelSOTAandseveralpointsabovethebestbaselinefrom
AlthoughvisualizationshaveshownthatImageNetmodels
containfeaturesthatrespondtothepresenceoftextinan theoriginalpaper. SimilartoSST-2,theseotherresultson
HatefulMemesusethegroundtruthtextwhichCLIPdoes
| image (Zeiler                           | & Fergus,    | 2014), these | representations | are         |                     |                     |            |        |
| --------------------------------------- | ------------ | ------------ | --------------- | ----------- | ------------------- | ------------------- | ---------- | ------ |
|                                         |              |              |                 | not have    | access to. Finally, | we note that        | zero-shot  | CLIP   |
| not sufficiently                        | fine-grained | to use for   | the task of     | optical     |                     |                     |            |        |
|                                         |              |              |                 | outperforms | the best            | results using fully | supervised | linear |
| characterrecognition(OCR).Tocompensate, |              |              | modelsare       |             |                     |                     |            |        |
probesacrossallother56modelsincludedinourevaluation
| augmented | with the outputs | of custom | OCR engines | and |     |     |     |     |
| --------- | ---------------- | --------- | ----------- | --- | --- | --- | --- | --- |
featurestoboostperformanceontaskswherethiscapability suite. ThissuggestsCLIP’sOCRcapabilityisatleastsome-
whatuniquecomparedtoexistingworkonself-supervised
| isrequired(Singhetal.,2019;Yangetal.,2020). |     |     | Earlydur- |     |     |     |     |     |
| ------------------------------------------- | --- | --- | --------- | --- | --- | --- | --- | --- |
andsupervisedrepresentationlearning.
ingthedevelopmentofCLIP,wenoticedthatCLIPbeganto
learnprimitiveOCRcapabilitieswhichappearedtosteadily
| improve | over the course | of the project. | To evaluate | this |     |     |     |     |
| ------- | --------------- | --------------- | ----------- | ---- | --- | --- | --- | --- |
qualitativelynoticedbehavior,wemeasuredperformance

LearningTransferableVisualModelsFromNaturalLanguageSupervision 46
TextRetrieval ImageRetrieval
Flickr30k MSCOCO Flickr30k MSCOCO
R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10
enuteniF
Unicoder-VLa 86.2 96.3 99.0 62.3 87.1 92.8 71.5 90.9 94.9 46.7 76.0 85.3
Uniterb 87.3 98.0 99.2 65.7 88.6 93.8 75.6 94.1 96.8 52.9 79.9 88.0
VILLAc 87.9 97.5 98.8 - - - 76.3 94.2 96.8 - - -
Oscard - - - 73.5 92.2 96.0 - - - 57.5 82.8 89.8
ERNIE-ViLe 88.7 98.0 99.2 - - - 76.7 93.6 96.4 - - -
tohS-oreZ
VisualN-Gramsf 15.4 35.7 45.1 8.7 23.1 33.3 8.8 21.2 29.9 5.0 14.5 21.9
ImageBERTg - - - 44.0 71.2 80.4 - - - 32.3 59.0 70.2
Unicoder-VLa 64.3 86.8 92.3 - - - 48.4 76.0 85.2 - - -
Uniterb 83.6 95.7 97.7 - - - 68.7 89.2 93.9 - - -
CLIP 88.0 98.7 99.4 58.4 81.5 88.1 68.7 90.6 95.2 37.8 62.4 72.2
Table13.CLIPimproveszero-shotretrievalandiscompetitivewiththebestfine-tunedresultonFlickr30ktextretrieval. Bold
indicatesbestoverallperformancewhileanunderlineindicatesbestincategoryperformance(zero-shotorfine-tuned). Forallother
models,bestresultsfromthepaperarereportedregardlessofmodelsize/variant.MSCOCOperformanceisreportedonthe5ktestset.
a(Lietal.,2020a)b(Chenetal.,2019)c(Ganetal.,2020)d(Lietal.,2020b)e(Yuetal.,2020)f(Lietal.,2017)g(Qietal.,2020)
IIIT5K Hateful
MNIST SVHN 1k Memes SST-2
enuteniF SOTA 99.8a 96.4b 98.9c 78.0d 97.5e
JOINTf - - 89.6 - -
CBoWg - - - - 80.0
raeniL RawPixels 92.5 - - - -
ESBest 98.9h - - 58.6h 59.0i
CLIP 99.2 - - 77.3 80.5
SZ
UCF101 K700 RareAct
Top-1 AVG mWAP mWSAP
CLIP 88.4 51.0 90.0 63.3 67.9
Table14.OCRperformanceon5datasets.Allmetricsareaccuracy
onthetestsetexceptforHatefulMemeswhichreportsROCAUC
onthedevset.SinglemodelSOTAreportedtobestofknowledge.
ES Best reports the best performance across the 56 non-CLIP
modelsinourevaluationsuite.a(Assiri,2020)b(Jaderbergetal.,
2015)c(Wangetal.,2020)d(Lippeetal.,2020)f(Jaderbergetal.,
2014)g(Wangetal.,2018)h(Xieetal.,2020)i(Mahajanetal.,
2018)
E.3.ActionRecognitioninVideos
Forthepurposeoflearning,apotentiallyimportantaspect
ofnaturallanguageisitsabilitytoexpress,andthereforesu-
pervise,anextremelywidesetofconcepts. ACLIPmodel,
sinceitistrainedtopairsemi-arbitrarytextwithimages,is
likelytoreceivesupervisionforawiderangeofvisualcon-
ceptsinvolvingbothcommonandpropernouns,verbs,and
adjectives. ImageNet-1K,bycontrast,onlylabelscommon
nouns. DoesthelackofbroadersupervisioninImageNet
resultinweakertransferofImageNetmodelstotasksinvolv-
ingtherecognitionofvisualconceptsthatarenotnouns?
To investigate this, we measure and compare the perfor-
mance of CLIP and ImageNet models on several video
enuteniF R(2+1)D-BERTa 98.7 - - -
NSENet-L2b - 84.8 - -
HT100MS3Dd 91.3 - - -
BaselineI3De - 70.2 - -
raeniL MMVFACf 91.8 - - -
NSENet-L2c 89.4c 68.2c - -
CLIP 92.0 73.0 - -
SZ
HT100MS3Dd - - 30.5 34.8
CLIP 80.3 69.6 40.7 44.8
Table15.Actionrecognitionperformanceon3videodatasets.Sin-
glemodelSOTAreportedtobestofknowledge.Notethatlinear
CLIPandlinearNSENet-L2aretrainedandevaluatedonasingle
framesubsampledversionofeachdatasetandnotdirectlycompa-
rabletopriorwork. OnKinetics-700,wereporttheActivityNet
competitionmetricwhichistheaverageoftop-1andtop-5per-
formance.a(Kalfaogluetal.,2020)b(Luetal.,2020)c(Xieetal.,
2020)d(Miechetal.,2020b)e(Carreiraetal.,2019)f(Alayrac
etal.,2020)
actionclassificationdatasetswhichmeasuretheabilityofa
modeltorecognizeverbs. InTable15wereportresultson
UCF-101(Soomroetal.,2012)andKinetics-700(Carreira
etal.,2019), twocommondatasetsforthetask. Unfortu-
nately,ourCPUbasedlinearclassifiertakesaprohibitively
longtimetoevaluateonavideodatasetduetotheverylarge
number of training frames. To deal with this, we aggres-
sivelysub-sampleeachvideotoonlyasinglecenterframe,
effectively turning it into an image classification dataset.
Asaresult,ourreportedperformanceinalinearevaluation
settinglikelyunderestimatesperformancebyamoderate
amount.

47
LearningTransferableVisualModelsFromNaturalLanguageSupervision
|     |     |     | IN    |     | IN-V2 IN-A  | IN-R  | ObjectNet | IN-Sketch | IN-Vid |      | YTBB |      |
| --- | --- | --- | ----- | --- | ----------- | ----- | --------- | --------- | ------ | ---- | ---- | ---- |
|     |     |     | Top-1 |     | Top-1 Top-1 | Top-1 | Top-1     | Top-1     | PM0    | PM10 | PM0  | PM10 |
NSEfficientNet-L2a 88.3 80.2 84.9 74.7 68.5 47.6 88.0 82.1 67.7 63.5
FixResNeXt101-32x48dV2b 86.4 78.0 68.4 80.0 57.8 59.1 85.8 72.2 68.9 57.7
LinearProbeCLIP 85.4 75.9 75.3 84.2 66.2 57.4 89.1 77.2 68.7 63.1
Zero-ShotCLIP 76.2 70.1 77.2 88.9 72.3 60.2 95.3 89.2 95.2 88.5
Table16.DetailedImageNetrobustnessperformance.INisusedtoabbreviateforImageNet.a(Xieetal.,2020)b(Touvronetal.,2019)
Despitethishandicap,CLIPfeaturestransfersurprisingly E.4.Geolocalization
| welltothistask. | CLIPmatchesthebestpriorresultonUCF- |     |     |     |     |     |                  |     |                |     |             |     |
| --------------- | ----------------------------------- | --- | --- | --- | --- | --- | ---------------- | --- | -------------- | --- | ----------- | --- |
|                 |                                     |     |     |     |     |     | Another behavior | we  | noticed during | the | development | of  |
101inalinearprobeevaluationsettingandalsooutperforms
CLIPwasitsabilitytorecognizemanyplacesandlocations.
| allothermodelsinourevaluationsuite. |     |     |     |     | OnKinetics-700, |     |     |     |     |     |     |     |
| ----------------------------------- | --- | --- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- |
ToquantifythiswecreatedtheCountry211datasetasde-
CLIPalsooutperformsthefine-tunedI3Dbaselinefromthe
scribedinAppendixAandreportresultsonitthroughout
| original paper.             | Since  | it does   | not                    | require | a training     | stage, |                                               |                    |     |     |             |         |
| --------------------------- | ------ | --------- | ---------------------- | ------- | -------------- | ------ | --------------------------------------------- | ------------------ | --- | --- | ----------- | ------- |
|                             |        |           |                        |         |                |        | thepaper. Howeveritisanewbenchmarksotocompare |                    |     |     |             |         |
| we report                   | CLIP’s | zero-shot | performance            |         | when averaging |        |                                               |                    |     |     |             |         |
|                             |        |           |                        |         |                |        | with prior work                               | on geolocalization |     | we  | also report | results |
| predictionsacrossallframes. |        |           | CLIPalsoperformswellin |         |                |        |                                               |                    |     |     |             |         |
ontheIM2GPStestsetfromHays&Efros(2008)inTable
thissettingandonKinetics-700itsperformanceiswithin
|           |                  |     |              |     |          |         | 17. SinceIM2GPSisaregressionbenchmark,weguessthe |     |     |     |     |     |
| --------- | ---------------- | --- | ------------ | --- | -------- | ------- | ------------------------------------------------ | --- | --- | --- | --- | --- |
| 1% of the | fully supervised |     | I3D baseline |     | which is | trained |                                                  |     |     |     |     |     |
GPScoordinatesofthenearestimageinasetofreference
| on545000labeledvideos. |     |     | Encouragedbytheseresults,we |     |     |     |                                  |     |     |     |                 |     |
| ---------------------- | --- | --- | --------------------------- | --- | --- | --- | -------------------------------- | --- | --- | --- | --------------- | --- |
|                        |     |     |                             |     |     |     | imagesusingCLIP’sembeddingspace. |     |     |     | Thisisnotazero- |     |
alsomeasureCLIP’sperformanceontherecentlyintroduced
shotresultsinceitusesnearest-neighborregression.Despite
RareActdataset(Miechetal.,2020a)whichwasdesigned
|            |           |             |     |            |         |      | queryingonly1millionimages, |     |     | whichismuchlessthan |     |     |
| ---------- | --------- | ----------- | --- | ---------- | ------- | ---- | --------------------------- | --- | --- | ------------------- | --- | --- |
| to measure | zero-shot | recognition |     | of unusual | actions | like |                             |     |     |                     |     |     |
priorwork,CLIPperformssimilarlytoseveraltaskspecific
| “hammeringaphone”and“drillinganegg”. |     |     |     |     | CLIPimproves |     |     |     |     |     |     |     |
| ------------------------------------ | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
models.Itisnot,however,competitivewiththecurrentstate
overthepriorstateoftheart,aS3Dmodeltrainedonauto-
oftheart.
maticallyextractedcaptionsfrom100millioninstructional
videos,by10points.
E.5.RobustnesstoDistributionShift
WhileCLIPhasencouraginglystrongperformanceonthe
Section3.3providesahighlevelsummaryandanalysisof
taskofactionrecognition,wenotethattherearemanydiffer-
|     |     |     |     |     |     |     | ImageNet-related | robustness | results. | We  | briefly | provide |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | ---------- | -------- | --- | ------- | ------- |
encesbetweenthemodelsbeingcomparedbeyondjusttheir
form of supervision such as model architecture, training some additional numerical details in this appendix. Per-
formanceresultsperdatasetareprovidedinTable16and
| datadistribution,datasetsize,andcomputeused. |     |     |     |     |     | Further |     |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
workisneededtomorepreciselydeterminewhatspecific comparedwiththecurrentstateoftheartresultsreported
|     |     |     |     |     |     |     | inTaorietal.(2020)’sevaluationsuite. |     |     | Zero-shotCLIPim- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | ---------------- | --- | --- |
designdecisionscontributetoachievinghighperformance
provesthestateofthearton5ofthe7datasets,ImageNet-R,
onthistask.
ObjectNet,ImageNet-Sketch,ImageNet-Vid,andYoutube-
BB.CLIP’simprovementsarelargestonImageNet-Vidand
Youtube-BBduetoitsflexiblezero-shotcapabilityandon
ImageNet-R,whichlikelyreflectsCLIP’spre-trainingdis-
tributionincludingsignificantamountsofcreativecontent.
|     | 1km | 25km | 200km |     | 750km 2500km |     |     |     |     |     |     |     |
| --- | --- | ---- | ----- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
AsimilarbehaviorhasbeendocumentedfortheInstagram
| ISNsa                   | 16.9 | 43.0        | 51.9 |     | 66.7       | 80.2      |             |         |           |           |     |              |
| ----------------------- | ---- | ----------- | ---- | --- | ---------- | --------- | ----------- | ------- | --------- | --------- | --- | ------------ |
|                         |      |             |      |     |            |           | pre-trained | ResNeXt | models as | discussed | in  | Taori et al. |
| CPlaNetb                | 16.5 | 37.1        | 46.4 |     | 62.0       | 78.5      |             |         |           |           |     |              |
| CLIP                    | 13.9 | 32.9        | 43.0 |     | 62.0       | 79.3      | (2020).     |         |           |           |     |              |
| Deep-Ret+c              | 14.4 | 33.3        | 47.7 |     | 61.6       | 73.4      |             |         |           |           |     |              |
| PlaNetd                 | 8.4  | 24.5        | 37.6 |     | 53.6       | 71.3      |             |         |           |           |     |              |
| Table17.Geolocalization |      | performance |      | on  | the IM2GPS | test set. |             |         |           |           |     |              |
Metricispercentofimageslocalizedwithinagivenradius.Models
areorderedbyaverageperformance.a(Muller-Budacketal.,2018)
b(HongsuckSeoetal.,2018)c(Voetal.,2017)c(Weyandetal.,
2016)

48
LearningTransferableVisualModelsFromNaturalLanguageSupervision
F.ModelHyperparameters
|     | Hyperparameter     |     |     | Value |     |     |
| --- | ------------------ | --- | --- | ----- | --- | --- |
|     | Batchsize          |     |     | 32768 |     |     |
|     | Vocabularysize     |     |     | 49408 |     |     |
|     | Trainingepochs     |     |     | 32    |     |     |
|     | Maximumtemperature |     |     | 100.0 |     |     |
|     | Weightdecay        |     |     | 0.2   |     |     |
|     | Warm-upiterations  |     |     | 2000  |     |     |
|     | Adamβ              |     |     | 0.9   |     |     |
1
Adamβ
|     |              | 2   | 0.999(ResNet),0.98(ViT) |     |     |     |
| --- | ------------ | --- | ----------------------- | --- | --- | --- |
|     | Adam(cid:15) |     | 10−8(ResNet),10−6(ViT)  |     |     |     |
Table18. CommonCLIPhyperparameters
|     | Learning | Embedding | Input | ResNet | TextTransformer |     |
| --- | -------- | --------- | ----- | ------ | --------------- | --- |
Model rate dimension resolution blocks width layers width heads
5×10−4
| RN50 |     | 1024 | 224 (3,4,6,3) | 2048 | 12 512 | 8   |
| ---- | --- | ---- | ------------- | ---- | ------ | --- |
5×10−4
| RN101 |     | 512 | 224 (3,4,23,3) | 2048 | 12 512 | 8   |
| ----- | --- | --- | -------------- | ---- | ------ | --- |
5×10−4
| RN50x4  |          | 640       | 288 (4,6,10,6)             | 2560              | 12 640          | 10  |
| ------- | -------- | --------- | -------------------------- | ----------------- | --------------- | --- |
| RN50x16 | 4×10−4   | 768       | 384 (6,8,18,8)             | 3072              | 12 768          | 12  |
| RN50x64 | 3.6×10−4 | 1024      | 448 (3,15,36,10)           | 4096              | 12 1024         | 16  |
|         |          | Table19.  | CLIP-ResNethyperparameters |                   |                 |     |
|         | Learning | Embedding | Input                      | VisionTransformer | TextTransformer |     |
Model rate dimension resolution layers width heads layers width heads
| ViT-B/32       | 5×10−4 | 512      | 224                     | 12 768 12  | 12 512 | 8   |
| -------------- | ------ | -------- | ----------------------- | ---------- | ------ | --- |
| ViT-B/16       | 5×10−4 | 512      | 224                     | 12 768 12  | 12 512 | 8   |
| ViT-L/14       | 4×10−4 | 768      | 224                     | 24 1024 16 | 12 768 | 12  |
| ViT-L/14-336px | 2×10−5 | 768      | 336                     | 24 1024 16 | 12 768 | 12  |
|                |        | Table20. | CLIP-ViThyperparameters |            |        |     |