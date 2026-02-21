# Awesome Machine Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) ⭐ 438,983 | 🐛 70 | 📅 2026-01-28 [![Track Awesome List](https://www.trackawesomelist.com/badge.svg)](https://www.trackawesomelist.com/josephmisiti/awesome-machine-learning/) with stars

A curated list of awesome machine learning frameworks, libraries and software (by language). Inspired by `awesome-php`.

*If you want to contribute to this list (please do), send me a pull request or contact me [@josephmisiti](https://twitter.com/josephmisiti).*
Also, a listed repository should be deprecated if:

* Repository's owner explicitly says that "this library is not maintained".
* Not committed for a long time (2\~3 years).

Further resources:

* For a list of free machine learning books available for download, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md) ⭐ 71,712 | 🐛 14 | 🌐 Python | 📅 2026-01-29.

* For a list of professional machine learning events, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/events.md) ⭐ 71,712 | 🐛 14 | 🌐 Python | 📅 2026-01-29.

* For a list of (mostly) free machine learning courses available online, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md) ⭐ 71,712 | 🐛 14 | 🌐 Python | 📅 2026-01-29.

* For a list of blogs and newsletters on data science and machine learning, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/blogs.md) ⭐ 71,712 | 🐛 14 | 🌐 Python | 📅 2026-01-29.

* For a list of free-to-attend meetups and local events, go [here](https://github.com/josephmisiti/awesome-machine-learning/blob/master/meetups.md) ⭐ 71,712 | 🐛 14 | 🌐 Python | 📅 2026-01-29.

## Table of Contents

### Frameworks and Libraries

<!-- MarkdownTOC depth=4 -->

<!-- Contents-->

* [Awesome Machine Learning ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](#awesome-machine-learning-)
  * [Table of Contents](#table-of-contents)
    * [Frameworks and Libraries](#frameworks-and-libraries)
    * [Tools](#tools)
  * [APL](#apl)
    * [General-Purpose Machine Learning](#apl-general-purpose-machine-learning)
  * [C](#c)
    * [General-Purpose Machine Learning](#c-general-purpose-machine-learning)
    * [Computer Vision](#c-computer-vision)
  * [C++](#cpp)
    * [Computer Vision](#cpp-computer-vision)
    * [General-Purpose Machine Learning](#cpp-general-purpose-machine-learning)
    * [Natural Language Processing](#cpp-natural-language-processing)
    * [Speech Recognition](#cpp-speech-recognition)
    * [Sequence Analysis](#cpp-sequence-analysis)
    * [Gesture Detection](#cpp-gesture-detection)
    * [Reinforcement Learning](#cpp-reinforcement-learning)
  * [Common Lisp](#common-lisp)
    * [General-Purpose Machine Learning](#common-lisp-general-purpose-machine-learning)
  * [Clojure](#clojure)
    * [Natural Language Processing](#clojure-natural-language-processing)
    * [General-Purpose Machine Learning](#clojure-general-purpose-machine-learning)
    * [Deep Learning](#clojure-deep-learning)
    * [Data Analysis](#clojure-data-analysis--data-visualization)
    * [Data Visualization](#clojure-data-visualization)
    * [Interop](#clojure-interop)
    * [Misc](#clojure-misc)
    * [Extra](#clojure-extra)
  * [Crystal](#crystal)
    * [General-Purpose Machine Learning](#crystal-general-purpose-machine-learning)
  * [CUDA PTX](#cuda-ptx)
    * [Neurosymbolic AI](#cuda-ptx-neurosymbolic-ai)
  * [Elixir](#elixir)
    * [General-Purpose Machine Learning](#elixir-general-purpose-machine-learning)
    * [Natural Language Processing](#elixir-natural-language-processing)
  * [Erlang](#erlang)
    * [General-Purpose Machine Learning](#erlang-general-purpose-machine-learning)
  * [Fortran](#fortran)
    * [General-Purpose Machine Learning](#fortran-general-purpose-machine-learning)
    * [Data Analysis / Data Visualization](#fortran-data-analysis--data-visualization)
  * [Go](#go)
    * [Natural Language Processing](#go-natural-language-processing)
    * [General-Purpose Machine Learning](#go-general-purpose-machine-learning)
    * [Spatial analysis and geometry](#go-spatial-analysis-and-geometry)
    * [Data Analysis / Data Visualization](#go-data-analysis--data-visualization)
    * [Computer vision](#go-computer-vision)
    * [Reinforcement learning](#go-reinforcement-learning)
  * [Haskell](#haskell)
    * [General-Purpose Machine Learning](#haskell-general-purpose-machine-learning)
  * [Java](#java)
    * [Natural Language Processing](#java-natural-language-processing)
    * [General-Purpose Machine Learning](#java-general-purpose-machine-learning)
    * [Speech Recognition](#java-speech-recognition)
    * [Data Analysis / Data Visualization](#java-data-analysis--data-visualization)
    * [Deep Learning](#java-deep-learning)
  * [Javascript](#javascript)
    * [Natural Language Processing](#javascript-natural-language-processing)
    * [Data Analysis / Data Visualization](#javascript-data-analysis--data-visualization)
    * [General-Purpose Machine Learning](#javascript-general-purpose-machine-learning)
    * [Misc](#javascript-misc)
    * [Demos and Scripts](#javascript-demos-and-scripts)
  * [Julia](#julia)
    * [General-Purpose Machine Learning](#julia-general-purpose-machine-learning)
    * [Natural Language Processing](#julia-natural-language-processing)
    * [Data Analysis / Data Visualization](#julia-data-analysis--data-visualization)
    * [Misc Stuff / Presentations](#julia-misc-stuff--presentations)
  * [Kotlin](#kotlin)
    * [Deep Learning](#kotlin-deep-learning)
  * [Lua](#lua)
    * [General-Purpose Machine Learning](#lua-general-purpose-machine-learning)
    * [Demos and Scripts](#lua-demos-and-scripts)
  * [Matlab](#matlab)
    * [Computer Vision](#matlab-computer-vision)
    * [Natural Language Processing](#matlab-natural-language-processing)
    * [General-Purpose Machine Learning](#matlab-general-purpose-machine-learning)
    * [Data Analysis / Data Visualization](#matlab-data-analysis--data-visualization)
  * [.NET](#net)
    * [Computer Vision](#net-computer-vision)
    * [Natural Language Processing](#net-natural-language-processing)
    * [General-Purpose Machine Learning](#net-general-purpose-machine-learning)
    * [Data Analysis / Data Visualization](#net-data-analysis--data-visualization)
  * [Objective C](#objective-c)
    * [General-Purpose Machine Learning](#objective-c-general-purpose-machine-learning)
  * [OCaml](#ocaml)
    * [General-Purpose Machine Learning](#ocaml-general-purpose-machine-learning)
  * [OpenCV](#opencv)
    * [Computer Vision](#opencv-Computer-Vision)
    * [Text-Detection](#Text-Character-Number-Detection)
  * [Perl](#perl)
    * [Data Analysis / Data Visualization](#perl-data-analysis--data-visualization)
    * [General-Purpose Machine Learning](#perl-general-purpose-machine-learning)
  * [Perl 6](#perl-6)
    * [Data Analysis / Data Visualization](#perl-6-data-analysis--data-visualization)
    * [General-Purpose Machine Learning](#perl-6-general-purpose-machine-learning)
  * [PHP](#php)
    * [Natural Language Processing](#php-natural-language-processing)
    * [General-Purpose Machine Learning](#php-general-purpose-machine-learning)
  * [Python](#python)
    * [Computer Vision](#python-computer-vision)
    * [Natural Language Processing](#python-natural-language-processing)
    * [General-Purpose Machine Learning](#python-general-purpose-machine-learning)
    * [Data Analysis / Data Visualization](#python-data-analysis--data-visualization)
    * [Misc Scripts / iPython Notebooks / Codebases](#python-misc-scripts--ipython-notebooks--codebases)
    * [Neural Networks](#python-neural-networks)
    * [Survival Analysis](#python-survival-analysis)
    * [Federated Learning](#python-federated-learning)
    * [Kaggle Competition Source Code](#python-kaggle-competition-source-code)
    * [Reinforcement Learning](#python-reinforcement-learning)
    * [Speech Recognition](#python-speech-recognition)
  * [Ruby](#ruby)
    * [Natural Language Processing](#ruby-natural-language-processing)
    * [General-Purpose Machine Learning](#ruby-general-purpose-machine-learning)
    * [Data Analysis / Data Visualization](#ruby-data-analysis--data-visualization)
    * [Misc](#ruby-misc)
  * [Rust](#rust)
    * [General-Purpose Machine Learning](#rust-general-purpose-machine-learning)
    * [Deep Learning](#rust-deep-learning)
    * [Natural Language Processing](#rust-natural-language-processing)
  * [R](#r)
    * [General-Purpose Machine Learning](#r-general-purpose-machine-learning)
    * [Data Analysis / Data Visualization](#r-data-analysis--data-visualization)
  * [SAS](#sas)
    * [General-Purpose Machine Learning](#sas-general-purpose-machine-learning)
    * [Data Analysis / Data Visualization](#sas-data-analysis--data-visualization)
    * [Natural Language Processing](#sas-natural-language-processing)
    * [Demos and Scripts](#sas-demos-and-scripts)
  * [Scala](#scala)
    * [Natural Language Processing](#scala-natural-language-processing)
    * [Data Analysis / Data Visualization](#scala-data-analysis--data-visualization)
    * [General-Purpose Machine Learning](#scala-general-purpose-machine-learning)
  * [Scheme](#scheme)
    * [Neural Networks](#scheme-neural-networks)
  * [Swift](#swift)
    * [General-Purpose Machine Learning](#swift-general-purpose-machine-learning)
  * [TensorFlow](#tensorflow)
    * [General-Purpose Machine Learning](#tensorflow-general-purpose-machine-learning)

### [Tools](#tools-1)

* [Neural Networks](#tools-neural-networks)
* [Misc](#tools-misc)

[Credits](#credits)

<!-- /MarkdownTOC -->

<a name="apl"></a>

## APL

<a name="apl-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [naive-apl](https://github.com/mattcunningham/naive-apl) ⭐ 24 | 🐛 1 | 🌐 APL | 📅 2018-01-21 - Naive Bayesian Classifier implementation in APL. **\[Deprecated]**

<a name="c"></a>

## C

<a name="c-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Darknet](https://github.com/pjreddie/darknet) ⭐ 26,435 | 🐛 1,977 | 🌐 C | 📅 2024-05-03 - Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.
* [libonnx](https://github.com/xboot/libonnx) ⭐ 648 | 🐛 17 | 🌐 C | 📅 2025-08-05 - A lightweight, portable pure C99 onnx inference engine for embedded devices with hardware acceleration support.
* [Recommender](https://github.com/GHamrouni/Recommender) ⭐ 268 | 🐛 1 | 🌐 C | 📅 2022-07-19 - A C library for product recommendations/suggestions using collaborative filtering (CF).
* [cONNXr](https://github.com/alrevuelta/cONNXr) ⭐ 216 | 🐛 40 | 🌐 C | 📅 2023-10-29 - An `ONNX` runtime written in pure C (99) with zero dependencies focused on small embedded devices. Run inference on your machine learning models no matter which framework you train it with. Easy to install and compiles everywhere, even in very old devices.
* [neonrvm](https://github.com/siavashserver/neonrvm) ⚠️ Archived - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [Hybrid Recommender System](https://github.com/SeniorSA/hybrid-rs-trainner) ⭐ 16 | 🐛 0 | 🌐 Python | 📅 2016-11-14 - A hybrid recommender system based upon scikit-learn algorithms. **\[Deprecated]**
* [onnx-c](https://github.com/onnx/onnx-c) - A lightweight C library for ONNX model inference, optimized for performance and portability across platforms.

<a name="c-computer-vision"></a>

#### Computer Vision

* [YOLOv8](https://github.com/ultralytics/ultralytics) ⭐ 53,446 | 🐛 356 | 🌐 Python | 📅 2026-02-21 - Ultralytics' YOLOv8 implementation with C++ support for real-time object detection and tracking, optimized for edge devices.
* [CCV](https://github.com/liuliu/ccv) ⭐ 7,200 | 🐛 78 | 🌐 C++ | 📅 2026-02-19 - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library.
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has a Matlab toolbox.

<a name="cpp"></a>

## C++

<a name="cpp-computer-vision"></a>

#### Computer Vision

* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) ⭐ 33,793 | 🐛 358 | 🌐 C++ | 📅 2024-08-03 - A real-time multi-person keypoint detection library for body, face, hands, and foot estimation
* [VIGRA](https://github.com/ukoethe/vigra) ⭐ 438 | 🐛 101 | 🌐 C++ | 📅 2025-12-23 - VIGRA is a genertic cross-platform C++ computer vision and machine learning library for volumes of arbitrary dimensionality with Python bindings.
* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
* [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models **\[Deprecated]**
* [OpenCV](https://opencv.org) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.

<a name="cpp-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Caffe](https://github.com/BVLC/caffe) ⭐ 34,839 | 🐛 1,175 | 🌐 C++ | 📅 2024-07-31 - A deep learning framework developed with cleanliness, readability, and speed in mind. \[DEEP LEARNING]
* [XGBoost](https://github.com/dmlc/xgboost) ⭐ 28,029 | 🐛 477 | 🌐 C++ | 📅 2026-02-20 - A parallelized optimized general purpose gradient boosting library.
* [MXNet](https://github.com/apache/incubator-mxnet) ⚠️ Archived - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [LightGBM](https://github.com/Microsoft/LightGBM) ⭐ 18,094 | 🐛 477 | 🌐 C++ | 📅 2026-02-19 - Microsoft's fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.
* [Opik](https://www.comet.com/site/products/opik/) - Open source engineering platform to debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards. ([Source Code](https://github.com/comet-ml/opik/) ⭐ 17,792 | 🐛 171 | 🌐 Python | 📅 2026-02-21)
* [CNTK](https://github.com/Microsoft/CNTK) ⚠️ Archived - The Computational Network Toolkit (CNTK) by Microsoft Research, is a unified deep-learning toolkit that describes neural networks as a series of computational steps via a directed graph.
* [PyCaret](https://github.com/pycaret/pycaret) ⭐ 9,698 | 🐛 423 | 🌐 Jupyter Notebook | 📅 2025-04-21 - An open-source, low-code machine learning library in Python that automates machine learning workflows.
* [CatBoost](https://github.com/catboost/catboost) ⭐ 8,810 | 🐛 679 | 🌐 C++ | 📅 2026-02-21 - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, contains fast inference implementation and supports CPU and GPU (even multi-GPU) computation.
* [Vowpal Wabbit (VW)](https://github.com/VowpalWabbit/vowpal_wabbit) ⭐ 8,659 | 🐛 0 | 🌐 C++ | 📅 2026-02-20 - A fast out-of-core learning system.
* [Speedster](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/speedster) ⭐ 8,353 | 🐛 111 | 🌐 Python | 📅 2024-07-22 -Automatically apply SOTA optimization techniques to achieve the maximum inference speed-up on your hardware. \[DEEP LEARNING]
* [Featuretools](https://github.com/featuretools/featuretools) ⭐ 7,614 | 🐛 163 | 🌐 Python | 📅 2026-02-03 - A library for automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning using reusable feature engineering "primitives".
* [Feast](https://github.com/gojek/feast) ⭐ 6,727 | 🐛 272 | 🌐 Python | 📅 2026-02-21 - A feature store for the management, discovery, and access of machine learning features. Feast provides a consistent view of feature data for both model training and model serving.
* [DSSTNE](https://github.com/amznlabs/amazon-dsstne) ⚠️ Archived - A software library created by Amazon for training and deploying deep neural networks using GPUs which emphasizes speed and scale over experimental flexibility.
* [Warp-CTC](https://github.com/baidu-research/warp-ctc) ⭐ 4,075 | 🐛 89 | 🌐 Cuda | 📅 2024-03-04 - A fast parallel implementation of Connectionist Temporal Classification (CTC), on both CPU and GPU.
* [oneDNN](https://github.com/oneapi-src/oneDNN) ⭐ 3,958 | 🐛 144 | 🌐 C++ | 📅 2026-02-21 - An open-source cross-platform performance library for deep learning applications.
* [Polyaxon](https://github.com/polyaxon/polyaxon) ⭐ 3,697 | 🐛 123 | 📅 2026-02-16 - A platform for reproducible and scalable machine learning and deep learning.
* [DyNet](https://github.com/clab/dynet) ⭐ 3,435 | 🐛 233 | 🌐 C++ | 📅 2023-12-01 - A dynamic neural network library working well with networks that have dynamic structures that change for every training instance. Written in C++ with bindings in Python.
* [xLearn](https://github.com/aksnzhy/xlearn) ⭐ 3,097 | 🐛 194 | 🌐 C++ | 📅 2023-08-28 - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertising and recommender systems.
* [Shogun](https://github.com/shogun-toolbox/shogun) ⭐ 3,068 | 🐛 423 | 🌐 C++ | 📅 2023-12-19 - The Shogun Machine Learning Toolbox.
* [DeepDetect](https://github.com/jolibrain/deepdetect) ⭐ 2,548 | 🐛 96 | 🌐 C++ | 📅 2026-01-13 - A machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications.
* * [Agentic Context Engine](https://github.com/kayba-ai/agentic-context-engine) ⭐ 1,895 | 🐛 20 | 🌐 Python | 📅 2026-02-20 -In-context learning framework that allows agents to learn from execution feedback.
* [nndeploy](https://github.com/nndeploy/nndeploy) ⭐ 1,744 | 🐛 21 | 🌐 C++ | 📅 2026-02-20 - An Easy-to-Use and High-Performance AI deployment framework.
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) ⭐ 1,619 | 🐛 87 | 🌐 C++ | 📅 2024-04-01 - A fast SVM library on GPUs and CPUs.
* [libfm](https://github.com/srendle/libfm) ⭐ 1,490 | 🐛 21 | 🌐 C++ | 📅 2020-03-28 - A generic approach that allows to mimic most factorization models by feature engineering.
* [Hopsworks](https://github.com/logicalclocks/hopsworks) ⭐ 1,285 | 🐛 16 | 🌐 Java | 📅 2025-02-10 - A data-intensive platform for AI with the industry's first open-source feature store. The Hopsworks Feature Store provides both a feature warehouse for training and batch based on Apache Hive and a feature serving database, based on MySQL Cluster, for online applications.
* [ThunderGBM](https://github.com/Xtra-Computing/thundergbm) ⭐ 711 | 🐛 39 | 🌐 C++ | 📅 2025-03-19 - A fast library for GBDTs and Random Forests on GPUs.
* [Intel® oneAPI Data Analytics Library](https://github.com/oneapi-src/oneDAL) ⭐ 647 | 🐛 69 | 🌐 C++ | 📅 2026-02-20 - A high performance software library developed by Intel and optimized for Intel's architectures. Library provides algorithmic building blocks for all stages of data analytics and allows to process data in batch, online and distributed modes.
* [Fido](https://github.com/FidoProject/Fido) ⭐ 462 | 🐛 15 | 🌐 C++ | 📅 2020-01-05 - A highly-modular C++ machine learning library for embedded electronics and robotics.
* [XAD](https://github.com/auto-differentiation/XAD) ⭐ 411 | 🐛 3 | 🌐 C++ | 📅 2026-02-06 - Comprehensive backpropagation tool for C++.
* [ParaMonte](https://github.com/cdslaborg/paramonte) ⭐ 300 | 🐛 20 | 🌐 Fortran | 📅 2025-12-18 - A general-purpose library with C/C++ interface for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [N2D2](https://github.com/CEA-LIST/N2D2) ⭐ 158 | 🐛 6 | 🌐 C | 📅 2024-07-03 - CEA-List's CAD framework for designing and simulating Deep Neural Network, and building full DNN-based applications on embedded platforms
* [BanditLib](https://github.com/jkomiyama/banditlib) ⭐ 140 | 🐛 0 | 🌐 C++ | 📅 2023-11-09 - A simple Multi-armed Bandit library. **\[Deprecated]**
* [skynet](https://github.com/Tyill/skynet) ⭐ 62 | 🐛 0 | 🌐 C++ | 📅 2021-08-15 - A library for learning neural networks, has C-interface, net set in JSON. Written in C++ with bindings in Python, C++ and C#.
* [LKYDeepNN](https://github.com/mosdeo/LKYDeepNN) ⭐ 50 | 🐛 6 | 🌐 C++ | 📅 2021-01-10 - A header-only C++11 Neural Network library. Low dependency, native traditional chinese document.
* [FlexML](https://github.com/ozguraslank/flexml) ⭐ 29 | 🐛 4 | 🌐 Python | 📅 2026-01-24 - Easy-to-use and flexible AutoML library for Python.
* [proNet-core](https://github.com/cnclabs/proNet-core) ⭐ 3 | 🐛 0 | 📅 2019-05-15 - A general-purpose network embedding framework: pair-wise representations optimization Network Edit.
* [CUDA](https://code.google.com/p/cuda-convnet/) - This is a fast C++/CUDA implementation of convolutional \[DEEP LEARNING]
* [Distributed Machine learning Tool Kit (DMTK)](http://www.dmtk.io/) - A distributed machine learning (parameter server) framework by Microsoft. Enables training models on large data sets across multiple machines. Current tools bundled with it include: LightLDA and Distributed (Multisense) Word Embedding.
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications.
* [igraph](http://igraph.org/) - General purpose graph library.
* [MLDB](https://mldb.ai) - The Machine Learning Database is a database designed for machine learning. Send it commands over a RESTful API to store data, explore it using SQL, then train machine learning models and expose them as APIs.
* [mlpack](https://www.mlpack.org/) - A scalable C++ machine learning library.
* [PyCUDA](https://mathema.tician.de/software/pycuda/) - Python interface to CUDA
* [ROOT](https://root.cern.ch) - A modular scientific software framework. It provides all the functionalities needed to deal with big data processing, statistical analysis, visualization and storage.
* [shark](http://image.diku.dk/shark/sphinx_pages/build/html/index.html) - A fast, modular, feature-rich open-source C++ machine learning library.
* [sofia-ml](https://code.google.com/archive/p/sofia-ml) - Suite of fast incremental algorithms.
* [Stan](http://mc-stan.org/) - A probabilistic programming language implementing full Bayesian statistical inference with Hamiltonian Monte Carlo sampling.
* [Timbl](https://languagemachines.github.io/timbl/) - A software package/C++ library implementing several memory-based learning algorithms, among which IB1-IG, an implementation of k-nearest neighbor classification, and IGTree, a decision-tree approximation of IB1-IG. Commonly used for NLP.
* [QuestDB](https://questdb.io/) - A relational column-oriented database designed for real-time analytics on time series and event data.
* [Phoenix](https://phoenix.arize.com) - Uncover insights, surface problems, monitor and fine tune your generative LLM, CV and tabular models.
* [Truss](https://truss.baseten.co) - An open source framework for packaging and serving ML models.

<a name="cpp-natural-language-processing"></a>

#### Natural Language Processing

* [SentencePiece](https://github.com/google/sentencepiece) ⭐ 11,656 | 🐛 25 | 🌐 C++ | 📅 2026-02-11 - A C++ library for unsupervised text tokenization and detokenization, widely used in modern NLP models.
* [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) ⭐ 2,963 | 🐛 18 | 🌐 C++ | 📅 2025-09-28 - C, C++, and Python tools for named entity recognition and relation extraction
* [MeTA](https://github.com/meta-toolkit/meta) ⭐ 713 | 🐛 56 | 🌐 C++ | 📅 2023-04-17 - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.
* [BLLIP Parser](https://github.com/BLLIP/bllip-parser) ⭐ 228 | 🐛 25 | 🌐 GAP | 📅 2021-11-07 - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser).
* [colibri-core](https://github.com/proycon/colibri-core) ⭐ 129 | 🐛 9 | 🌐 C++ | 📅 2026-02-05 - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [frog](https://github.com/LanguageMachines/frog) ⭐ 80 | 🐛 13 | 🌐 C++ | 📅 2025-12-11 - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
* [ucto](https://github.com/LanguageMachines/ucto) ⭐ 70 | 🐛 12 | 🌐 C++ | 📅 2026-02-09 - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
* [libfolia](https://github.com/LanguageMachines/libfolia) ⭐ 17 | 🐛 5 | 🌐 C++ | 📅 2025-12-11 - C++ library for the [FoLiA format](https://proycon.github.io/folia/)
* [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks. **\[Deprecated]**
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) - CRFsuite is an implementation of Conditional Random Fields (CRFs) for labeling sequential data. **\[Deprecated]**

<a name="cpp-speech-recognition"></a>

#### Speech Recognition

* [Kaldi](https://github.com/kaldi-asr/kaldi) ⭐ 15,324 | 🐛 254 | 🌐 Shell | 📅 2025-09-22 - Kaldi is a toolkit for speech recognition written in C++ and licensed under the Apache License v2.0. Kaldi is intended for use by speech recognition researchers.
* [Vosk](https://github.com/alphacep/vosk-api) ⭐ 14,252 | 🐛 585 | 🌐 Jupyter Notebook | 📅 2025-12-08 - An offline speech recognition toolkit with C++ support, designed for low-resource devices and multiple languages.

<a name="cpp-sequence-analysis"></a>

#### Sequence Analysis

* [ToPS](https://github.com/ayoshiaki/tops) ⭐ 37 | 🐛 1 | 🌐 HTML | 📅 2024-06-10 - This is an object-oriented framework that facilitates the integration of probabilistic models for sequences over a user defined alphabet. **\[Deprecated]**

<a name="cpp-gesture-detection"></a>

#### Gesture Detection

* [grt](https://github.com/nickgillian/grt) ⭐ 884 | 🐛 88 | 🌐 C++ | 📅 2019-11-01 - The Gesture Recognition Toolkit (GRT) is a cross-platform, open-source, C++ machine learning library designed for real-time gesture recognition.

<a name="cpp-reinforcement-learning"></a>

#### Reinforcement Learning

* [RLtools](https://github.com/rl-tools/rl-tools) ⭐ 925 | 🐛 19 | 🌐 C++ | 📅 2026-02-11 - The fastest deep reinforcement learning library for continuous control, implemented header-only in pure, dependency-free C++ (Python bindings available as well).

<a name="common-lisp"></a>

## Common Lisp

<a name="common-lisp-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [mgl](https://github.com/melisgl/mgl/) ⭐ 639 | 🐛 6 | 🌐 Common Lisp | 📅 2025-06-07 - Neural networks (boltzmann machines, feed-forward and recurrent nets), Gaussian Processes.
* [mgl-gpr](https://github.com/melisgl/mgl-gpr/) ⭐ 66 | 🐛 0 | 🌐 Common Lisp | 📅 2025-06-17 - Evolutionary algorithms. **\[Deprecated]**
* [cl-random-forest](https://github.com/masatoi/cl-random-forest) ⭐ 60 | 🐛 5 | 🌐 Common Lisp | 📅 2022-07-27 - Implementation of Random Forest in Common Lisp.
* [cl-online-learning](https://github.com/masatoi/cl-online-learning) ⭐ 49 | 🐛 0 | 🌐 Common Lisp | 📅 2022-03-20 - Online learning algorithms (Perceptron, AROW, SCW, Logistic Regression).
* [cl-libsvm](https://github.com/melisgl/cl-libsvm/) ⭐ 16 | 🐛 0 | 🌐 C++ | 📅 2021-10-10 - Wrapper for the libsvm support vector machine library. **\[Deprecated]**

<a name="clojure"></a>

## Clojure

<a name="clojure-natural-language-processing"></a>

#### Natural Language Processing

* [Clojure-openNLP](https://github.com/dakrone/clojure-opennlp) ⭐ 758 | 🐛 4 | 🌐 Clojure | 📅 2018-11-27 - Natural Language Processing in Clojure (opennlp).
* [Infections-clj](https://github.com/r0man/inflections-clj) ⭐ 222 | 🐛 4 | 🌐 Clojure | 📅 2025-08-14 - Rails-like inflection library for Clojure and ClojureScript.

<a name="clojure-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Clojush](https://github.com/lspector/Clojush) ⭐ 337 | 🐛 53 | 🌐 Clojure | 📅 2023-05-06 - The Push programming language and the PushGP genetic programming system implemented in Clojure.
* [scicloj.ml](https://github.com/scicloj/scicloj.ml) ⭐ 238 | 🐛 2 | 🌐 Clojure | 📅 2025-11-02 -  A idiomatic Clojure machine learning library based on tech.ml.dataset with a unique approach for immutable data processing pipelines.
* [clortex](https://github.com/htm-community/clortex) ⭐ 183 | 🐛 11 | 🌐 Clojure | 📅 2015-12-02 - General Machine Learning library using Numenta’s Cortical Learning Algorithm. **\[Deprecated]**
* [Infer](https://github.com/aria42/infer) ⭐ 177 | 🐛 2 | 🌐 Clojure | 📅 2015-12-29 - Inference and machine learning in Clojure. **\[Deprecated]**
* [comportex](https://github.com/htm-community/comportex) ⭐ 154 | 🐛 6 | 🌐 Clojure | 📅 2016-09-23 - Functionally composable Machine Learning library using Numenta’s Cortical Learning Algorithm. **\[Deprecated]**
* [Touchstone](https://github.com/ptaoussanis/touchstone) ⭐ 140 | 🐛 1 | 🌐 Clojure | 📅 2024-03-19 - Clojure A/B testing library.
* [Encog](https://github.com/jimpil/enclog) ⭐ 137 | 🐛 0 | 🌐 Clojure | 📅 2016-05-04 - Clojure wrapper for Encog (v3) (Machine-Learning framework that specializes in neural-nets). **\[Deprecated]**
* [clj-ml](https://github.com/joshuaeckroth/clj-ml/) ⭐ 136 | 🐛 3 | 🌐 Clojure | 📅 2021-12-05 - A machine learning library for Clojure built on top of Weka and friends.
* [Fungp](https://github.com/vollmerm/fungp) ⭐ 100 | 🐛 0 | 🌐 Clojure | 📅 2014-05-19 - A genetic programming library for Clojure. **\[Deprecated]**
* [lambda-ml](https://github.com/cloudkj/lambda-ml) ⭐ 79 | 🐛 0 | 🌐 Clojure | 📅 2018-11-03 - Simple, concise implementations of machine learning techniques and utilities in Clojure.
* [Statistiker](https://github.com/clojurewerkz/statistiker) ⭐ 65 | 🐛 2 | 🌐 Clojure | 📅 2015-07-06 - Basic Machine Learning algorithms in Clojure. **\[Deprecated]**
* [clj-boost](https://gitlab.com/alanmarazzi/clj-boost) - Wrapper for XGBoost

<a name="clojure-deep-learning"></a>

#### Deep Learning

* [cortex](https://github.com/originrose/cortex) ⭐ 1,273 | 🐛 29 | 🌐 Clojure | 📅 2018-09-10 - Neural networks, regression and feature learning in Clojure.
* [Deep Diamond](https://github.com/uncomplicate/deep-diamond) ⭐ 458 | 🐛 1 | 🌐 Clojure | 📅 2026-02-10 - A fast Clojure Tensor & Deep Learning library
* [Flare](https://github.com/aria42/flare) ⭐ 288 | 🐛 1 | 🌐 Clojure | 📅 2019-06-28 - Dynamic Tensor Graph library in Clojure (think PyTorch, DynNet, etc.)
* [jutsu.ai](https://github.com/hswick/jutsu.ai) ⭐ 102 | 🐛 3 | 🌐 Clojure | 📅 2018-06-08 - Clojure wrapper for deeplearning4j with some added syntactic sugar.
* [dl4clj](https://github.com/yetanalytics/dl4clj) ⭐ 100 | 🐛 1 | 🌐 Clojure | 📅 2018-07-17 - Clojure wrapper for Deeplearning4j.
* [MXNet](https://mxnet.apache.org/versions/1.7.0/api/clojure) - Bindings to Apache MXNet - part of the MXNet project

<a name="clojure-data-analysis--data-visualization"></a>

#### Data Analysis

* [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset) ⭐ 737 | 🐛 30 | 🌐 Clojure | 📅 2026-02-10 - Clojure dataframe library and pipeline for data processing and machine learning
* [PigPen](https://github.com/Netflix/PigPen) ⭐ 566 | 🐛 19 | 🌐 Clojure | 📅 2023-04-10 - Map-Reduce for Clojure.
* [Tablecloth](https://github.com/scicloj/tablecloth) ⭐ 352 | 🐛 45 | 🌐 Clojure | 📅 2026-02-10 - A dataframe grammar wrapping tech.ml.dataset, inspired by several R libraries
* [Geni](https://github.com/zero-one-group/geni) ⭐ 292 | 🐛 18 | 🌐 Clojure | 📅 2023-11-28 - a Clojure dataframe library that runs on Apache Spark
* [Panthera](https://github.com/alanmarazzi/panthera) ⭐ 190 | 🐛 3 | 🌐 Clojure | 📅 2020-05-03 - Clojure API wrapping Python's Pandas library
* [Incanter](http://incanter.org/) - Incanter is a Clojure-based, R-like platform for statistical computing and graphics.

<a name="clojure-data-visualization"></a>

#### Data Visualization

* [clojupyter](https://github.com/clojupyter/clojupyter) ⭐ 858 | 🐛 13 | 🌐 Clojure | 📅 2025-03-18 -  A Jupyter kernel for Clojure - run Clojure code in Jupyter Lab, Notebook and Console.
* [Oz](https://github.com/metasoarous/oz) ⭐ 836 | 🐛 66 | 🌐 Clojure | 📅 2024-04-02 - Data visualisation using Vega/Vega-Lite and Hiccup, and a live-reload platform for literate-programming
* [Hanami](https://github.com/jsa-aerial/hanami) ⭐ 407 | 🐛 8 | 🌐 Clojure | 📅 2025-07-02 - Clojure(Script) library and framework for creating interactive visualization applications based in Vega-Lite (VGL) and/or Vega (VG) specifications. Automatic framing and layouts along with a powerful templating system for abstracting visualization specs
* [Delight](https://github.com/datamechanics/delight) ⚠️ Archived - A listener that streams your spark events logs to delight, a free and improved spark UI
* [notespace](https://github.com/scicloj/notespace) ⭐ 149 | 🐛 21 | 🌐 Clojure | 📅 2024-01-03 - Notebook experience in your Clojure namespace
* [Saite](https://github.com/jsa-aerial/saite) ⭐ 142 | 🐛 2 | 🌐 Clojure | 📅 2025-07-02 -  Clojure(Script) client/server application for dynamic interactive explorations and the creation of live shareable documents capturing them using Vega/Vega-Lite, CodeMirror, markdown, and LaTeX
* [Pink Gorilla Notebook](https://github.com/pink-gorilla/gorilla-notebook) ⭐ 107 | 🐛 5 | 🌐 Clojure | 📅 2021-06-27 - A Clojure/Clojurescript notebook application/-library based on Gorilla-REPL
* [Envision](https://github.com/clojurewerkz/envision) ⭐ 77 | 🐛 1 | 🌐 Clojure | 📅 2018-02-13 - Clojure Data Visualisation library, based on Statistiker and D3.

<a name="clojure-interop"></a>

#### Interop

* [Libpython-clj](https://github.com/clj-python/libpython-clj) ⭐ 1,191 | 🐛 33 | 🌐 Clojure | 📅 2026-01-19 - Interop with Python
* [ClojisR](https://github.com/scicloj/clojisr) ⭐ 158 | 🐛 30 | 🌐 Clojure | 📅 2025-01-15 - Interop with R and Renjin (R on the JVM)
* [Java Interop](https://clojure.org/reference/java_interop) - Clojure has Native Java Interop from which Java's ML ecosystem can be accessed
* [JavaScript Interop](https://clojurescript.org/reference/javascript-api) - ClojureScript has Native JavaScript Interop from which JavaScript's ML ecosystem can be accessed

<a name="clojure-misc"></a>

#### Misc

* [kixistats](https://github.com/MastodonC/kixi.stats) ⭐ 368 | 🐛 1 | 🌐 Clojure | 📅 2025-11-10 - A library of statistical distribution sampling and transducing functions
* [fastmath](https://github.com/generateme/fastmath) ⭐ 276 | 🐛 32 | 🌐 Clojure | 📅 2026-02-16 - A collection of functions for mathematical and statistical computing, macine learning, etc., wrapping several JVM libraries
* [matlib](https://github.com/atisharma/matlib) ⭐ 26 | 🐛 0 | 🌐 Clojure | 📅 2020-09-25 - A Clojure library of optimisation and control theory tools and convenience functions based on Neanderthal.
* [Neanderthal](https://neanderthal.uncomplicate.org/) - Fast Clojure Matrix Library (native CPU, GPU, OpenCL, CUDA)

<a name="clojure-extra"></a>

#### Extra

* [Scicloj](https://scicloj.github.io/pages/libraries/) - Curated list of ML related resources for Clojure.

<a name="crystal"></a>

## Crystal

<a name="crystal-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [crystal-fann](https://github.com/NeuraLegion/crystal-fann) ⭐ 87 | 🐛 4 | 🌐 Crystal | 📅 2026-01-15 - FANN (Fast Artificial Neural Network) binding.
* [machine](https://github.com/mathieulaporte/machine) ⭐ 40 | 🐛 5 | 🌐 Crystal | 📅 2021-08-27 - Simple machine learning algorithm.

<a name="cuda-ptx"></a>

## CUDA PTX

<a name="cuda-ptx-neurosymbolic-ai"></a>

#### Neurosymbolic AI

* [Knowledge3D (K3D)](https://github.com/danielcamposramos/Knowledge3D) ⭐ 30 | 🐛 2 | 🌐 Python | 📅 2026-02-21 - Sovereign GPU-native spatial AI architecture with PTX-first cognitive engine (RPN/TRM reasoning), tri-modal fusion (text/visual/audio), and 3D persistent memory ("Houses"). Features sub-100µs inference, procedural knowledge compression (69:1 ratio), and multi-agent swarm architecture. Zero external dependencies for core inference paths.

<a name="elixir"></a>

## Elixir

<a name="elixir-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Simple Bayes](https://github.com/fredwu/simple_bayes) ⭐ 394 | 🐛 1 | 🌐 Elixir | 📅 2017-09-25 - A Simple Bayes / Naive Bayes implementation in Elixir.
* [Tensorflex](https://github.com/anshuman23/tensorflex) ⭐ 308 | 🐛 5 | 🌐 C | 📅 2019-07-02 - Tensorflow bindings for the Elixir programming language.
* [emel](https://github.com/mrdimosthenis/emel) ⭐ 115 | 🐛 0 | 🌐 Gleam | 📅 2024-09-26 - A simple and functional machine learning library written in Elixir.

<a name="elixir-natural-language-processing"></a>

#### Natural Language Processing

* [Stemmer](https://github.com/fredwu/stemmer) ⭐ 154 | 🐛 0 | 🌐 Elixir | 📅 2024-01-14 - An English (Porter2) stemming implementation in Elixir.

<a name="erlang"></a>

## Erlang

<a name="erlang-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Disco](https://github.com/discoproject/disco/) ⭐ 1,629 | 🐛 140 | 🌐 Erlang | 📅 2018-01-30 - Map Reduce in Erlang. **\[Deprecated]**

<a name="fortran"></a>

## Fortran

<a name="fortran-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [neural-fortran](https://github.com/modern-fortran/neural-fortran) ⭐ 460 | 🐛 42 | 🌐 Fortran | 📅 2026-01-06 - A parallel neural net microframework.
  Read the paper [here](https://arxiv.org/abs/1902.06714).

<a name="fortran-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [ParaMonte](https://github.com/cdslaborg/paramonte) ⭐ 300 | 🐛 20 | 🌐 Fortran | 📅 2025-12-18 - A general-purpose Fortran library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).

<a name="go"></a>

## Go

<a name="go-natural-language-processing"></a>

#### Natural Language Processing

* [word-embedding](https://github.com/ynqa/word-embedding) ⭐ 505 | 🐛 5 | 🌐 Go | 📅 2023-04-02 - Word Embeddings: the full implementation of word2vec, GloVe in Go.
* [sentences](https://github.com/neurosnap/sentences) ⭐ 463 | 🐛 5 | 🌐 Go | 📅 2024-02-28 - Golang implementation of Punkt sentence tokenizer.
* [Cybertron](https://github.com/nlpodyssey/cybertron) ⭐ 324 | 🐛 25 | 🌐 Go | 📅 2024-06-08 - Cybertron: the home planet of the Transformers in Go.
* [go-porterstemmer](https://github.com/reiver/go-porterstemmer) ⭐ 192 | 🐛 6 | 🌐 Go | 📅 2021-06-23 - A native Go clean room implementation of the Porter Stemming algorithm. **\[Deprecated]**
* [go-ngram](https://github.com/Lazin/go-ngram) ⭐ 114 | 🐛 0 | 🌐 Go | 📅 2016-05-27 - In-memory n-gram index with compression. *\[Deprecated]*
* [snowball](https://github.com/tebeka/snowball) ⭐ 48 | 🐛 0 | 🌐 C | 📅 2025-06-10 - Snowball Stemmer for Go.
* [paicehusk](https://github.com/Rookii/paicehusk) ⭐ 29 | 🐛 2 | 🌐 Go | 📅 2013-12-16 - Golang implementation of the Paice/Husk Stemming Algorithm. *\[Deprecated]*

<a name="go-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [golearn](https://github.com/sjwhitworth/golearn) ⭐ 9,452 | 🐛 89 | 🌐 Go | 📅 2024-01-15 - Machine learning for Go.
* [gorgonia](https://github.com/gorgonia/gorgonia) ⭐ 5,907 | 🐛 119 | 🌐 Go | 📅 2024-08-12 - Deep learning in Go.
* [Spago](https://github.com/nlpodyssey/spago) ⭐ 1,849 | 🐛 13 | 🌐 Go | 📅 2025-04-01 - Self-contained Machine Learning and Natural Language Processing library in Go.
* [goml](https://github.com/cdipaolo/goml) ⭐ 1,616 | 🐛 4 | 🌐 Go | 📅 2022-07-15 - Machine learning library written in pure Go.
* [eaopt](https://github.com/MaxHalford/eaopt) ⭐ 905 | 🐛 8 | 🌐 Go | 📅 2025-01-27 - An evolutionary optimization library.
* [bayesian](https://github.com/jbrukh/bayesian) ⭐ 810 | 🐛 0 | 🌐 Go | 📅 2025-12-07 - Naive Bayesian Classification for Golang. **\[Deprecated]**
* [Cloudforest](https://github.com/ryanbressler/CloudForest) ⭐ 749 | 🐛 34 | 🌐 Go | 📅 2022-02-05 - Ensembles of decision trees in Go/Golang. **\[Deprecated]**
* [gobrain](https://github.com/goml/gobrain) ⭐ 567 | 🐛 2 | 🌐 Go | 📅 2020-12-12 - Neural Networks written in Go.
* [leaves](https://github.com/dmitryikh/leaves) ⭐ 468 | 🐛 36 | 🌐 Go | 📅 2024-07-03 - A pure Go implementation of the prediction part of GBRTs, including XGBoost and LightGBM.
* [goro](https://github.com/aunum/goro) ⭐ 374 | 🐛 1 | 🌐 Go | 📅 2024-03-04 - A high-level machine learning library in the vein of Keras.
* [GoNN](https://github.com/fxsjy/gonn) ⭐ 361 | 🐛 4 | 🌐 Go | 📅 2016-01-29 - GoNN is an implementation of Neural Network in Go Language, which includes BPNN, RBF, PCN. **\[Deprecated]**
* [go-ml](https://github.com/alonsovidales/go_ml) ⭐ 200 | 🐛 3 | 🌐 Go | 📅 2017-04-17 - Linear / Logistic regression, Neural Networks, Collaborative Filtering and Gaussian Multivariate Distribution. **\[Deprecated]**
* [go-galib](https://github.com/thoj/go-galib) ⭐ 200 | 🐛 0 | 🌐 Go | 📅 2015-12-28 - Genetic Algorithms library written in Go / Golang. **\[Deprecated]**
* [go-featureprocessing](https://github.com/nikolaydubina/go-featureprocessing) ⚠️ Archived - Fast and convenient feature processing for low latency machine learning in Go.
* [neat](https://github.com/jinyeom/neat) ⚠️ Archived - Plug-and-play, parallel Go framework for NeuroEvolution of Augmenting Topologies (NEAT). **\[Deprecated]**
* [go-pr](https://github.com/daviddengcn/go-pr) ⭐ 68 | 🐛 0 | 🌐 Go | 📅 2013-06-08 - Pattern recognition package in Go lang. **\[Deprecated]**
* [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor) ⭐ 54 | 🐛 3 | 🌐 Go | 📅 2018-06-06 - Go binding for MXNet c\_predict\_api to do inference with a pre-trained model.
* [birdland](https://github.com/rlouf/birdland) ⭐ 46 | 🐛 0 | 🌐 Go | 📅 2019-08-28 - A recommendation library in Go.
* [go-ml-benchmarks](https://github.com/nikolaydubina/go-ml-benchmarks) ⭐ 32 | 🐛 6 | 🌐 Go | 📅 2024-05-09 — benchmarks of machine learning inference for Go.
* [therfoo](https://github.com/therfoo/therfoo) ⚠️ Archived - An embedded deep learning library for Go.
* [go-dnn](https://github.com/sudachen/go-dnn) ⚠️ Archived - Deep Neural Networks for Golang (powered by MXNet)
* [gorse](https://github.com/zhenghaoz/gorse) ⭐ 7 | 🐛 1 | 🌐 Go | 📅 2026-02-03 - An offline recommender system backend based on collaborative filtering written in Go.
* [go-ml-transpiler](https://github.com/znly/go-ml-transpiler) - An open source Go transpiler for machine learning models.

<a name="go-spatial-analysis-and-geometry"></a>

#### Spatial analysis and geometry

* [gogeo](https://github.com/golang/geo) ⭐ 1,821 | 🐛 33 | 🌐 Go | 📅 2026-01-29 - Spherical geometry in Go.
* [go-geom](https://github.com/twpayne/go-geom) ⭐ 955 | 🐛 9 | 🌐 Go | 📅 2025-11-20 - Go library to handle geometries.

<a name="go-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [gota](https://github.com/go-gota/gota) ⚠️ Archived - Dataframes.
* [SVGo](https://github.com/ajstarks/svgo) ⭐ 2,236 | 🐛 16 | 🌐 Go | 📅 2022-12-09 - The Go Language library for SVG generation.
* [globe](https://github.com/mmcloughlin/globe) ⭐ 1,598 | 🐛 6 | 🌐 Go | 📅 2024-02-09 - Globe wireframe visualization.
* [dataframe-go](https://github.com/rocketlaunchr/dataframe-go) ⭐ 1,285 | 🐛 19 | 🌐 Go | 📅 2022-04-02 - Dataframes for machine-learning and statistics (similar to pandas).
* [glot](https://github.com/arafatk/glot) ⭐ 406 | 🐛 16 | 🌐 Go | 📅 2025-05-20 - Glot is a plotting library for Golang built on top of gnuplot.
* [RF](https://github.com/fxsjy/RF.go) ⭐ 115 | 🐛 2 | 🌐 Go | 📅 2014-07-10 - Random forests implementation in Go. **\[Deprecated]**
* [go-graph](https://github.com/StepLg/go-graph) ⚠️ Archived - Graph library for Go/Golang language. **\[Deprecated]**
* [gonum/mat](https://godoc.org/gonum.org/v1/gonum/mat) - A linear algebra package for Go.
* [gonum/optimize](https://godoc.org/gonum.org/v1/gonum/optimize) - Implementations of optimization algorithms.
* [gonum/plot](https://godoc.org/gonum.org/v1/plot) - A plotting library.
* [gonum/stat](https://godoc.org/gonum.org/v1/gonum/stat) - A statistics library.
* [gonum/graph](https://godoc.org/gonum.org/v1/gonum/graph) - General-purpose graph library.

<a name="go-computer-vision"></a>

#### Computer vision

* [GoCV](https://github.com/hybridgroup/gocv) ⭐ 7,383 | 🐛 349 | 🌐 Go | 📅 2026-02-18 - Package for computer vision using OpenCV 4 and beyond.

<a name="go-reinforcement-learning"></a>

#### Reinforcement learning

* [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) ⭐ 12,764 | 🐛 81 | 🌐 Python | 📅 2026-02-20 - PyTorch implementations of Stable Baselines (deep) reinforcement learning algorithms.
* [gold](https://github.com/aunum/gold) ⭐ 353 | 🐛 7 | 🌐 Go | 📅 2020-10-22 - A reinforcement learning library.

<a name="haskell"></a>

## Haskell

<a name="haskell-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [HLearn](https://github.com/mikeizbicki/HLearn) ⭐ 1,729 | 🐛 23 | 🌐 Haskell | 📅 2016-05-29 - a suite of libraries for interpreting machine learning models according to their algebraic structure. **\[Deprecated]**
* [DNNGraph](https://github.com/ajtulloch/dnngraph) ⭐ 711 | 🐛 9 | 🌐 Haskell | 📅 2015-12-07 - A DSL for deep neural networks. **\[Deprecated]**
* [LambdaNet](https://github.com/jbarrow/LambdaNet) ⭐ 382 | 🐛 5 | 🌐 Haskell | 📅 2016-03-25 - Configurable Neural Networks in Haskell. **\[Deprecated]**
* [hnn](https://github.com/alpmestan/HNN) ⭐ 113 | 🐛 0 | 🌐 Haskell | 📅 2017-03-15 - Haskell Neural Network library.
* [haskell-ml](https://github.com/ajtulloch/haskell-ml) ⭐ 60 | 🐛 0 | 🌐 Haskell | 📅 2014-05-29 - Haskell implementations of various ML algorithms. **\[Deprecated]**
* [hopfield-networks](https://github.com/ajtulloch/hopfield-networks) ⭐ 16 | 🐛 0 | 🌐 Haskell | 📅 2014-04-13 - Hopfield Networks for unsupervised learning in Haskell. **\[Deprecated]**

<a name="java"></a>

## Java

<a name="java-natural-language-processing"></a>

#### Natural Language Processing

* [Twitter Text Java](https://github.com/twitter/twitter-text/tree/master/java) ⭐ 3,126 | 🐛 93 | 🌐 HTML | 📅 2024-04-26 - A Java implementation of Twitter's text processing library.
* [CogcompNLP](https://github.com/CogComp/cogcomp-nlp) ⭐ 480 | 🐛 200 | 🌐 Java | 📅 2023-07-07 - This project collects a number of core libraries for Natural Language Processing (NLP) developed in the University of Illinois' Cognitive Computation Group, for example `illinois-core-utilities` which provides a set of NLP-friendly data structures and a number of NLP-related utilities that support writing NLP applications, running experiments, etc, `illinois-edison` a library for feature extraction from illinois-core-utilities data structures and many other packages.
* [NLP4J](https://github.com/emorynlp/nlp4j) ⭐ 155 | 🐛 22 | 🌐 Java | 📅 2021-04-26 - The NLP4J project provides software and resources for natural language processing. The project started at the Center for Computational Language and EducAtion Research, and is currently developed by the Center for Language and Information Research at Emory University. **\[Deprecated]**
* [ClearTK](https://github.com/ClearTK/cleartk) ⭐ 132 | 🐛 59 | 🌐 Java | 📅 2023-06-14 - ClearTK provides a framework for developing statistical natural language processing (NLP) components in Java and is built on top of Apache UIMA. **\[Deprecated]**
* [Cortical.io](https://www.cortical.io/) - Retina: an API performing complex NLP operations (disambiguation, classification, streaming text filtering, etc...) as quickly and intuitively as the brain.
* [IRIS](https://github.com/cortical-io/Iris) - [Cortical.io's](https://cortical.io) FREE NLP, Retina API Analysis Tool (written in JavaFX!) - [See the Tutorial Video](https://www.youtube.com/watch?v=CsF4pd7fGF0).
* [CoreNLP](https://nlp.stanford.edu/software/corenlp.shtml) - Stanford CoreNLP provides a set of natural language analysis tools which can take raw English language text input and give the base forms of words.
* [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml) - A natural language parser is a program that works out the grammatical structure of sentences.
* [Stanford POS Tagger](https://nlp.stanford.edu/software/tagger.shtml) - A Part-Of-Speech Tagger (POS Tagger).
* [Stanford Name Entity Recognizer](https://nlp.stanford.edu/software/CRF-NER.shtml) - Stanford NER is a Java implementation of a Named Entity Recognizer.
* [Stanford Word Segmenter](https://nlp.stanford.edu/software/segmenter.shtml) - Tokenization of raw text is a standard pre-processing step for many NLP tasks.
* [Tregex, Tsurgeon and Semgrex](https://nlp.stanford.edu/software/tregex.shtml) - Tregex is a utility for matching patterns in trees, based on tree relationships and regular expression matches on nodes (the name is short for "tree regular expressions").
* [Stanford Phrasal: A Phrase-Based Translation System](https://nlp.stanford.edu/phrasal/)
* [Stanford English Tokenizer](https://nlp.stanford.edu/software/tokenizer.shtml) - Stanford Phrasal is a state-of-the-art statistical phrase-based machine translation system, written in Java.
* [Stanford Tokens Regex](https://nlp.stanford.edu/software/tokensregex.shtml) - A tokenizer divides text into a sequence of tokens, which roughly correspond to "words".
* [Stanford Temporal Tagger](https://nlp.stanford.edu/software/sutime.shtml) - SUTime is a library for recognizing and normalizing time expressions.
* [Stanford SPIED](https://nlp.stanford.edu/software/patternslearning.shtml) - Learning entities from unlabeled text starting with seed sets using patterns in an iterative fashion.
* [MALLET](http://mallet.cs.umass.edu/) - A Java-based package for statistical natural language processing, document classification, clustering, topic modelling, information extraction, and other machine learning applications to text.
* [OpenNLP](https://opennlp.apache.org/) - A machine learning based toolkit for the processing of natural language text.
* [LingPipe](http://alias-i.com/lingpipe/index.html) - A tool kit for processing text using computational linguistics.
* [Apache cTAKES](https://ctakes.apache.org/) - Apache Clinical Text Analysis and Knowledge Extraction System (cTAKES) is an open-source natural language processing system for information extraction from electronic medical record clinical free-text.

<a name="java-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [H2O](https://github.com/h2oai/h2o-3) ⭐ 7,510 | 🐛 2,876 | 🌐 Jupyter Notebook | 📅 2026-02-21 - ML engine that supports distributed learning on Hadoop, Spark or your laptop via APIs in R, Python, Scala, REST/JSON.
* [aerosolve](https://github.com/airbnb/aerosolve) ⭐ 4,798 | 🐛 10 | 🌐 Scala | 📅 2025-11-06 - A machine learning library by Airbnb designed from the ground up to be human friendly.
* [Mahout](https://github.com/apache/mahout) ⭐ 2,209 | 🐛 69 | 🌐 Rust | 📅 2026-02-19 - Distributed machine learning.
* [ORYX](https://github.com/oryxproject/oryx) ⚠️ Archived - Lambda Architecture Framework using Apache Spark and Apache Kafka with a specialization for real-time large-scale machine learning.
* [Datumbox](https://github.com/datumbox/datumbox-framework) ⭐ 1,084 | 🐛 2 | 🌐 Java | 📅 2023-11-30 - Machine Learning framework for rapid development of Machine Learning and Statistical applications.
* [SystemML](https://github.com/apache/systemml) ⭐ 1,080 | 🐛 78 | 🌐 Java | 📅 2026-02-20 - flexible, scalable machine learning (ML) language.
* [Encog](https://github.com/encog/encog-java-core) ⭐ 751 | 🐛 69 | 🌐 Java | 📅 2023-03-30 - An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trainings using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) ⭐ 324 | 🐛 31 | 🌐 Scala | 📅 2020-10-29 - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [htm.java](https://github.com/numenta/htm.java) ⭐ 315 | 🐛 49 | 🌐 Java | 📅 2021-10-23 - General Machine Learning library using Numenta’s Cortical Learning Algorithm.
* [liblinear-java](https://github.com/bwaldvogel/liblinear-java) ⭐ 309 | 🐛 7 | 🌐 Java | 📅 2024-12-31 - Java version of liblinear.
* [rapaio](https://github.com/padreati/rapaio) ⭐ 76 | 🐛 7 | 🌐 Java | 📅 2026-02-16 - statistics, data mining and machine learning toolbox in Java.
* [Chips-n-Salsa](https://github.com/cicirello/Chips-n-Salsa) ⭐ 72 | 🐛 3 | 🌐 Java | 📅 2026-02-16 - A Java library for genetic algorithms, evolutionary computation, and stochastic local search, with a focus on self-adaptation / self-tuning, as well as parallel execution.
* [jSciPy](https://github.com/hissain/jscipy) ⭐ 17 | 🐛 0 | 🌐 Java | 📅 2026-02-03 - A Java port of SciPy's signal processing module, offering filters, transformations, and other scientific computing utilities.
* [LBJava](https://github.com/CogComp/lbjava) ⭐ 14 | 🐛 41 | 🌐 Java | 📅 2022-07-01 - Learning Based Java is a modelling language for the rapid development of software systems, offers a convenient, declarative syntax for classifier and constraint definition directly in terms of the objects in the programmer's application.
* [knn-java-library](https://github.com/felipexw/knn-java-library) ⭐ 7 | 🐛 1 | 🌐 Java | 📅 2016-09-08 - Just a simple implementation of K-Nearest Neighbors algorithm using with a bunch of similarity measures.
* [AMIDST Toolbox](http://www.amidsttoolbox.com/) - A Java Toolbox for Scalable Probabilistic Machine Learning.
* [ELKI](https://elki-project.github.io/) - Java toolkit for data mining. (unsupervised: clustering, outlier detection etc.)
* [FlinkML in Apache Flink](https://ci.apache.org/projects/flink/flink-docs-master/dev/libs/ml/index.html) - Distributed machine learning library in Flink.
* [Meka](http://meka.sourceforge.net/) - An open source implementation of methods for multi-label classification and evaluation (extension to Weka).
* [MLlib in Apache Spark](https://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark.
* [Neuroph](http://neuroph.sourceforge.net/) - Neuroph is lightweight Java neural network framework.
* [Samoa](https://samoa.incubator.apache.org/) SAMOA is a framework that includes distributed machine learning for data streams with an interface to plug-in different stream processing platforms.
* [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) - RankLib is a library of learning to rank algorithms. **\[Deprecated]**
* [RapidMiner](https://rapidminer.com) - RapidMiner integration into Java code.
* [Stanford Classifier](https://nlp.stanford.edu/software/classifier.shtml) - A classifier is a machine learning tool that will take data items and place them into one of k classes.
* [Smile](https://haifengl.github.io/) - Statistical Machine Intelligence & Learning Engine.
* [Tribou](https://tribuo.org) - A machine learning library written in Java by Oracle.
* [Weka](https://www.cs.waikato.ac.nz/ml/weka/) - Weka is a collection of machine learning algorithms for data mining tasks.

<a name="java-speech-recognition"></a>

#### Speech Recognition

* [CMU Sphinx](https://cmusphinx.github.io) - Open Source Toolkit For Speech Recognition purely based on Java speech recognition library.

<a name="java-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [Spark](https://github.com/apache/spark) ⭐ 42,848 | 🐛 263 | 🌐 Scala | 📅 2026-02-21 - Spark is a fast and general engine for large-scale data processing.
* [Hadoop](https://github.com/apache/hadoop) ⭐ 15,488 | 🐛 91 | 🌐 Java | 📅 2026-02-19 - Hadoop/HDFS.
* [Onyx](https://github.com/onyx-platform/onyx) ⚠️ Archived - Distributed, masterless, high performance, fault tolerant data processing. Written entirely in Clojure.
* [Impala](https://github.com/cloudera/impala) ⭐ 34 | 🐛 28 | 🌐 C++ | 📅 2022-12-27 - Real-time Query for Hadoop.
* [Flink](https://flink.apache.org/) - Open source platform for distributed stream and batch data processing.
* [Storm](https://storm.apache.org/) - Storm is a distributed realtime computation system.
* [DataMelt](https://jwork.org/dmelt/) - Mathematics software for numeric computation, statistics, symbolic calculations, data analysis and data visualization.
* [Dr. Michael Thomas Flanagan's Java Scientific Library.](https://www.ee.ucl.ac.uk/~mflanaga/java/) **\[Deprecated]**

<a name="java-deep-learning"></a>

#### Deep Learning

* [Deeplearning4j](https://github.com/deeplearning4j/deeplearning4j) ⭐ 14,205 | 🐛 618 | 🌐 Java | 📅 2026-02-20 - Scalable deep learning for industry with parallel GPUs.
* [deepjavalibrary/djl](https://github.com/deepjavalibrary/djl) ⭐ 4,770 | 🐛 220 | 🌐 Java | 📅 2026-02-03 - Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework for deep learning, designed to be easy to get started with and simple to use for Java developers.
* [Keras Beginner Tutorial](https://victorzhou.com/blog/keras-neural-network-tutorial/) - Friendly guide on using Keras to implement a simple Neural Network in Python.

<a name="javascript"></a>

## JavaScript

<a name="javascript-natural-language-processing"></a>

#### Natural Language Processing

* [NLP Compromise](https://github.com/spencermountain/compromise) ⭐ 12,032 | 🐛 118 | 🌐 JavaScript | 📅 2025-12-12 - Natural Language processing in the browser.
* [natural](https://github.com/NaturalNode/natural) ⭐ 10,864 | 🐛 80 | 🌐 JavaScript | 📅 2026-02-19 - General natural language facilities for node.
* [nlp.js](https://github.com/axa-group/nlp.js) ⭐ 6,553 | 🐛 116 | 🌐 JavaScript | 📅 2025-01-09 - An NLP library built in node over Natural, with entity extraction, sentiment analysis, automatic language identify, and so more.
* [Knwl.js](https://github.com/loadfive/Knwl.js) ⭐ 5,288 | 🐛 13 | 🌐 JavaScript | 📅 2023-09-28 - A Natural Language Processor in JS.
* [Twitter-text](https://github.com/twitter/twitter-text) ⭐ 3,126 | 🐛 93 | 🌐 HTML | 📅 2024-04-26 - A JavaScript implementation of Twitter's text processing library.
* [Retext](https://github.com/retextjs/retext) ⭐ 2,432 | 🐛 0 | 🌐 JavaScript | 📅 2025-02-04 - Extensible system for analyzing and manipulating natural language.

<a name="javascript-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [science.js](https://github.com/jasondavies/science.js/) ⭐ 892 | 🐛 16 | 🌐 JavaScript | 📅 2021-01-11 - Scientific and statistical computing in JavaScript. **\[Deprecated]**
* [D3xter](https://github.com/NathanEpstein/D3xter) ⭐ 336 | 🐛 0 | 🌐 JavaScript | 📅 2020-10-13 - Straight forward plotting built on D3. **\[Deprecated]**
* [datakit](https://github.com/nathanepstein/datakit) ⭐ 287 | 🐛 0 | 🌐 JavaScript | 📅 2017-04-09 - A lightweight framework for data analysis in JavaScript
* [Z3d](https://github.com/NathanEpstein/Z3d) ⭐ 88 | 🐛 0 | 🌐 JavaScript | 📅 2015-01-08 - Easily make interactive 3d plots built on Three.js **\[Deprecated]**
* [cheminfo](https://www.cheminfo.org/) - Platform for data visualization and analysis, using the [visualizer](https://github.com/npellet/visualizer) ⭐ 51 | 🐛 187 | 🌐 JavaScript | 📅 2026-02-14 project.
* [statkit](https://github.com/rigtorp/statkit) ⭐ 50 | 🐛 0 | 🌐 JavaScript | 📅 2014-12-03 - Statistics kit for JavaScript. **\[Deprecated]**
* [D3.js](https://d3js.org/)
* [High Charts](https://www.highcharts.com/)
* [NVD3.js](http://nvd3.org/)
* [dc.js](https://dc-js.github.io/dc.js/)
* [chartjs](https://www.chartjs.org/)
* [dimple](http://dimplejs.org/)
* [amCharts](https://www.amcharts.com/)
* [Sigma.js](http://sigmajs.org/) - JavaScript library dedicated to graph drawing.
* [C3.js](https://c3js.org/) - customizable library based on D3.js for easy chart drawing.
* [Datamaps](https://datamaps.github.io/) - Customizable SVG map/geo visualizations using D3.js. **\[Deprecated]**
* [ZingChart](https://www.zingchart.com/) - library written on Vanilla JS for big data visualization.
* [Learn JS Data](http://learnjsdata.com/)
* [AnyChart](https://www.anychart.com/)
* [FusionCharts](https://www.fusioncharts.com/)
* [Nivo](https://nivo.rocks) - built on top of the awesome d3 and Reactjs libraries

<a name="javascript-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Netron](https://github.com/lutzroeder/netron) ⭐ 32,415 | 🐛 19 | 🌐 JavaScript | 📅 2026-02-20 - Visualizer for machine learning models.
* [MXNet](https://github.com/apache/incubator-mxnet) ⚠️ Archived - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [Brain.js](https://github.com/BrainJS/brain.js) ⭐ 14,860 | 🐛 91 | 🌐 TypeScript | 📅 2024-09-26 - Neural networks in JavaScript - continued community fork of [Brain](https://github.com/harthur/brain) ⚠️ Archived.
* [Brain](https://github.com/harthur/brain) ⚠️ Archived - Neural networks in JavaScript **\[Deprecated]**
* [Synaptic](https://github.com/cazala/synaptic) ⭐ 6,923 | 🐛 161 | 🌐 JavaScript | 📅 2020-09-03 - Architecture-free neural network library for Node.js and the browser.
* [ml5](https://github.com/ml5js/ml5-library) ⭐ 6,585 | 🐛 288 | 🌐 JavaScript | 📅 2024-10-11 - Friendly machine learning for the web!
* [Keras.js](https://github.com/transcranial/keras-js) ⭐ 4,968 | 🐛 81 | 🌐 JavaScript | 📅 2022-06-15 - Run Keras models in the browser, with GPU support provided by WebGL 2.
* [ml.js](https://github.com/mljs/ml) ⭐ 2,707 | 🐛 27 | 🌐 JavaScript | 📅 2024-10-21 - Machine learning and numerical analysis tools for Node.js and the Browser!
* [WebDNN](https://github.com/mil-tokyo/webdnn) ⭐ 1,998 | 🐛 81 | 🌐 TypeScript | 📅 2025-06-07 - Fast Deep Neural Network JavaScript Framework. WebDNN uses next generation JavaScript API, WebGPU for GPU execution, and WebAssembly for CPU execution.
* [Auto ML](https://github.com/ClimbsRocks/auto_ml) ⭐ 1,658 | 🐛 187 | 🌐 Python | 📅 2021-02-10 - Automated machine learning, data formatting, ensembling, and hyperparameter optimization for competitions and exploration- just give it a .csv file! **\[Deprecated]**
* [machinelearn.js](https://github.com/machinelearnjs/machinelearnjs) ⭐ 541 | 🐛 33 | 🌐 TypeScript | 📅 2026-02-07 - Machine Learning library for the web, Node.js and developers
* [Pavlov.js](https://github.com/NathanEpstein/Pavlov.js) ⭐ 500 | 🐛 0 | 🌐 C++ | 📅 2018-04-14 - Reinforcement learning using Markov Decision Processes.
* [DN2A](https://github.com/antoniodeluca/dn2a.js) ⭐ 465 | 🐛 6 | 🌐 TypeScript | 📅 2023-10-07 - Digital Neural Networks Architecture. **\[Deprecated]**
* [Node-SVM](https://github.com/nicolaspanel/node-svm) ⭐ 301 | 🐛 10 | 🌐 JavaScript | 📅 2019-03-26 - Support Vector Machine for Node.js
* [LDA.js](https://github.com/primaryobjects/lda) ⭐ 297 | 🐛 2 | 🌐 JavaScript | 📅 2024-08-20 - LDA topic modelling for Node.js
* [NeuralN](https://github.com/totemstech/neuraln) ⭐ 274 | 🐛 3 | 🌐 C++ | 📅 2015-06-29 - C++ Neural Network library for Node.js. It has advantage on large dataset and multi-threaded training. **\[Deprecated]**
* [Decision Trees](https://github.com/serendipious/nodejs-decision-tree-id3) ⭐ 217 | 🐛 5 | 🌐 TypeScript | 📅 2026-01-21 - NodeJS Implementation of Decision Tree using ID3 Algorithm. **\[Deprecated]**
* [Node-fann](https://github.com/rlidwka/node-fann) ⭐ 183 | 🐛 11 | 🌐 C++ | 📅 2017-01-11 - FANN (Fast Artificial Neural Network Library) bindings for Node.js **\[Deprecated]**
* [kalman](https://github.com/itamarwe/kalman) ⭐ 114 | 🐛 3 | 🌐 JavaScript | 📅 2015-09-05 - Kalman filter for JavaScript. **\[Deprecated]**
* [shaman](https://github.com/luccastera/shaman) ⭐ 107 | 🐛 0 | 🌐 JavaScript | 📅 2016-02-04 - Node.js library with support for both simple and multiple linear regression. **\[Deprecated]**
* [Learning.js](https://github.com/yandongliu/learningjs) ⭐ 65 | 🐛 1 | 🌐 JavaScript | 📅 2019-05-15 - JavaScript implementation of logistic regression/c4.5 decision tree **\[Deprecated]**
* [kNear](https://github.com/NathanEpstein/kNear) ⭐ 48 | 🐛 0 | 🌐 JavaScript | 📅 2017-11-14 - JavaScript implementation of the k nearest neighbors algorithm for supervised learning.
* [Kmeans.js](https://github.com/emilbayes/kMeans.js) ⭐ 46 | 🐛 2 | 🌐 CoffeeScript | 📅 2013-07-30 - Simple JavaScript implementation of the k-means algorithm, for node.js and the browser. **\[Deprecated]**
* [xgboost-node](https://github.com/nuanio/xgboost-node) ⭐ 46 | 🐛 3 | 🌐 Cuda | 📅 2017-10-30 - Run XGBoost model and make predictions in Node.js.
* [Bayesian-Bandit](https://github.com/omphalos/bayesian-bandit.js) ⭐ 45 | 🐛 0 | 🌐 JavaScript | 📅 2017-08-10 - Bayesian bandit implementation for Node and the browser. **\[Deprecated]**
* [tensor-js](https://github.com/Hoff97/tensorjs) ⭐ 38 | 🐛 3 | 🌐 TypeScript | 📅 2021-04-07 - A deep learning library for the browser, accelerated by WebGL and WebAssembly.
* [Clustering.js](https://github.com/emilbayes/clustering.js) ⭐ 30 | 🐛 0 | 🌐 JavaScript | 📅 2014-07-18 - Clustering algorithms implemented in JavaScript for Node.js and the browser. **\[Deprecated]**
* [Gaussian Mixture Model](https://github.com/lukapopijac/gaussian-mixture-model) ⭐ 30 | 🐛 2 | 🌐 JavaScript | 📅 2025-01-16 - Unsupervised machine learning with multivariate Gaussian mixture model.
* [JSMLT](https://github.com/jsmlt/jsmlt) ⭐ 26 | 🐛 24 | 🌐 JavaScript | 📅 2022-12-30 - Machine learning toolkit with classification and clustering for Node.js; supports visualization (see [visualml.io](https://visualml.io)).
* [Creatify MCP](https://github.com/TSavo/creatify-mcp) ⭐ 15 | 🐛 2 | 🌐 TypeScript | 📅 2025-05-26 - Model Context Protocol server that exposes Creatify AI's video generation capabilities to AI assistants, enabling natural language video creation workflows.
* [Catniff](https://github.com/nguyenphuminh/catniff) ⭐ 8 | 🐛 0 | 🌐 TypeScript | 📅 2026-01-21 - Torch-like deep learning framework for Javascript with support for tensors, autograd, optimizers, and other neural net constructs.
* [Kandle](https://github.com/final-kk/kandle) ⭐ 6 | 🐛 0 | 🌐 TypeScript | 📅 2026-01-30 - A JavaScript Native PyTorch-aligned Machine Learning Framework, built from scratch on WebGPU.
* [Convnet.js](https://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a JavaScript library for training Deep Learning models\[DEEP LEARNING] **\[Deprecated]**
* [Clusterfck](https://harthur.github.io/clusterfck/) - Agglomerative hierarchical clustering implemented in JavaScript for Node.js and the browser. **\[Deprecated]**
* [figue](https://code.google.com/archive/p/figue) - K-means, fuzzy c-means and agglomerative clustering.
* [mil-tokyo](https://github.com/mil-tokyo) - List of several machine learning libraries.
* [TensorFlow.js](https://js.tensorflow.org/) - A WebGL accelerated, browser based JavaScript library for training and deploying ML models.
* [WebNN](https://webnn.dev) - A new web standard that allows web apps and frameworks to accelerate deep neural networks with on-device hardware such as GPUs, CPUs, or purpose-built AI accelerators.

<a name="javascript-misc"></a>

#### Misc

* [stdlib](https://github.com/stdlib-js/stdlib) ⭐ 5,736 | 🐛 1,187 | 🌐 JavaScript | 📅 2026-02-20 - A standard library for JavaScript and Node.js, with an emphasis on numeric computing. The library provides a collection of robust, high performance libraries for mathematics, statistics, streams, utilities, and more.
* [simple-statistics](https://github.com/simple-statistics/simple-statistics) ⭐ 3,497 | 🐛 25 | 🌐 JavaScript | 📅 2026-02-01 - A JavaScript implementation of descriptive, regression, and inference statistics. Implemented in literate JavaScript with no dependencies, designed to work in all modern browsers (including IE) as well as in Node.js.
* [Pipcook](https://github.com/alibaba/pipcook) ⭐ 2,590 | 🐛 110 | 🌐 TypeScript | 📅 2026-02-21 - A JavaScript application framework for machine learning and its engineering.
* [sylvester](https://github.com/jcoglan/sylvester) ⭐ 1,158 | 🐛 29 | 🌐 JavaScript | 📅 2019-05-13 - Vector and Matrix math for JavaScript. **\[Deprecated]**
* [regression-js](https://github.com/Tom-Alexander/regression-js) ⭐ 951 | 🐛 45 | 🌐 JavaScript | 📅 2022-12-06 - A javascript library containing a collection of least squares fitting methods for finding a trend in a set of data.
* [GreatCircle](https://github.com/mwgg/GreatCircle) ⭐ 78 | 🐛 0 | 🌐 PHP | 📅 2022-03-20 - Library for calculating great circle distance.
* [Lyric](https://github.com/flurry/Lyric) ⚠️ Archived - Linear Regression library. **\[Deprecated]**
* [MLPleaseHelp](https://github.com/jgreenemi/MLPleaseHelp) - MLPleaseHelp is a simple ML resource search engine. You can use this search engine right now at <https://jgreenemi.github.io/MLPleaseHelp/>, provided via GitHub Pages.

<a name="javascript-demos-and-scripts"></a>

#### Demos and Scripts

* [The Bot](https://github.com/sta-ger/TheBot) ⭐ 6 | 🐛 0 | 🌐 JavaScript | 📅 2018-07-04 - Example of how the neural network learns to predict the angle between two points created with [Synaptic](https://github.com/cazala/synaptic) ⭐ 6,923 | 🐛 161 | 🌐 JavaScript | 📅 2020-09-03.
* [Half Beer](https://github.com/sta-ger/HalfBeer) ⭐ 6 | 🐛 0 | 🌐 JavaScript | 📅 2018-06-26 - Beer glass classifier created with [Synaptic](https://github.com/cazala/synaptic) ⭐ 6,923 | 🐛 161 | 🌐 JavaScript | 📅 2020-09-03.
* [NSFWJS](http://nsfwjs.com) - Indecent content checker with TensorFlow\.js
* [Rock Paper Scissors](https://rps-tfjs.netlify.com/) - Rock Paper Scissors trained in the browser with TensorFlow\.js
* [Heroes Wear Masks](https://heroeswearmasks.fun/) - A fun TensorFlow\.js-based oracle that tells, whether one wears a face mask or not. It can even tell when one wears the mask incorrectly.

<a name="julia"></a>

## Julia

<a name="julia-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [MXNet](https://github.com/apache/incubator-mxnet) ⚠️ Archived - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [MLJ](https://github.com/alan-turing-institute/MLJ.jl) ⭐ 1,895 | 🐛 84 | 🌐 Julia | 📅 2026-02-08 - A Julia machine learning framework.
* [Knet](https://github.com/denizyuret/Knet.jl) ⭐ 1,436 | 🐛 153 | 🌐 Jupyter Notebook | 📅 2024-11-15 - Koç University Deep Learning Framework.
* [Mocha](https://github.com/pluskid/Mocha.jl) ⭐ 1,286 | 🐛 36 | 🌐 Julia | 📅 2018-12-06 - Deep Learning framework for Julia inspired by Caffe. **\[Deprecated]**
* [GLM](https://github.com/JuliaStats/GLM.jl) ⭐ 633 | 🐛 79 | 🌐 Julia | 📅 2026-02-05 - Generalized linear models in Julia.
* [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl) ⭐ 557 | 🐛 38 | 🌐 Julia | 📅 2025-05-28 - Julia implementation of the scikit-learn API.
* [Distances](https://github.com/JuliaStats/Distances.jl) ⭐ 469 | 🐛 42 | 🌐 Julia | 📅 2025-03-28 - Julia module for Distance evaluation.
* [Mixed Models](https://github.com/dmbates/MixedModels.jl) ⭐ 438 | 🐛 56 | 🌐 Julia | 📅 2026-02-03 - A Julia package for fitting (statistical) mixed-effects models.
* [MultivariateStats](https://github.com/JuliaStats/MultivariateStats.jl) ⭐ 388 | 🐛 53 | 🌐 Julia | 📅 2026-02-20 - Methods for dimensionality reduction.
* [Clustering](https://github.com/JuliaStats/Clustering.jl) ⭐ 372 | 🐛 44 | 🌐 Julia | 📅 2025-11-24 - Basic functions for clustering data: k-means, dp-means, etc.
* [Gaussian Processes](https://github.com/STOR-i/GaussianProcesses.jl) ⭐ 319 | 🐛 31 | 🌐 Jupyter Notebook | 📅 2026-01-09 - Julia package for Gaussian processes.
* [XGBoost](https://github.com/dmlc/XGBoost.jl) ⭐ 303 | 🐛 33 | 🌐 Julia | 📅 2025-09-19 - eXtreme Gradient Boosting Package in Julia.
* [Mamba](https://github.com/brian-j-smith/Mamba.jl) ⭐ 259 | 🐛 47 | 🌐 Julia | 📅 2022-02-14 - Markov chain Monte Carlo (MCMC) for Bayesian analysis in Julia.
* [Kernel Density](https://github.com/JuliaStats/KernelDensity.jl) ⭐ 200 | 🐛 34 | 🌐 Julia | 📅 2026-01-20 - Kernel density estimators for Julia.
* [MLBase](https://github.com/JuliaStats/MLBase.jl) ⭐ 187 | 🐛 12 | 🌐 Julia | 📅 2023-08-17 - A set of functions to support the development of machine learning algorithms.
* [Merlin](https://github.com/hshindo/Merlin.jl) ⭐ 146 | 🐛 1 | 🌐 Julia | 📅 2019-11-25 - Flexible Deep Learning Framework in Julia.
* [MachineLearning](https://github.com/benhamner/MachineLearning.jl) ⭐ 117 | 🐛 2 | 🌐 Julia | 📅 2015-09-13 - Julia Machine Learning library. **\[Deprecated]**
* [Local Regression](https://github.com/JuliaStats/Loess.jl) ⭐ 110 | 🐛 9 | 🌐 Julia | 📅 2025-12-15 - Local regression, so smooooth!
* [GaussianMixtures](https://github.com/davidavdav/GaussianMixtures.jl) ⭐ 104 | 🐛 23 | 🌐 Julia | 📅 2026-01-08 - Large scale Gaussian Mixture Models.
* [GLMNet](https://github.com/simonster/GLMNet.jl) ⭐ 103 | 🐛 13 | 🌐 Julia | 📅 2026-01-13 - Julia wrapper for fitting Lasso/ElasticNet GLM models using glmnet.
* [NMF](https://github.com/JuliaStats/NMF.jl) ⭐ 94 | 🐛 5 | 🌐 Julia | 📅 2024-08-22 - A Julia package for non-negative matrix factorization.
* [ManifoldLearning](https://github.com/wildart/ManifoldLearning.jl) ⭐ 94 | 🐛 4 | 🌐 Julia | 📅 2024-03-02 - A Julia package for manifold learning and nonlinear dimensionality reduction.
* [Regression](https://github.com/lindahua/Regression.jl) ⭐ 65 | 🐛 7 | 🌐 Julia | 📅 2017-05-23 - Algorithms for regression analysis (e.g. linear regression and logistic regression). **\[Deprecated]**
* [ANN](https://github.com/EricChiang/ANN.jl) ⚠️ Archived - Julia artificial neural networks. **\[Deprecated]**
* [PGM](https://github.com/JuliaStats/PGM.jl) ⭐ 53 | 🐛 4 | 🌐 Julia | 📅 2020-02-08 - A Julia framework for probabilistic graphical models.
* [Neural](https://github.com/compressed/BackpropNeuralNet.jl) ⭐ 48 | 🐛 0 | 🌐 Julia | 📅 2017-05-23 - A neural network in Julia.
* [SVM](https://github.com/JuliaStats/SVM.jl) ⚠️ Archived - SVM for Julia. **\[Deprecated]**
* [MCMC](https://github.com/doobwa/MCMC.jl) ⭐ 36 | 🐛 4 | 🌐 Julia | 📅 2013-04-16 - MCMC tools for Julia. **\[Deprecated]**
* [ROCAnalysis](https://github.com/davidavdav/ROCAnalysis.jl) ⭐ 33 | 🐛 5 | 🌐 Jupyter Notebook | 📅 2023-09-15 - Receiver Operating Characteristics and functions for evaluation probabilistic binary classifiers.
* [Online Learning](https://github.com/lendle/OnlineLearning.jl) ⭐ 14 | 🐛 0 | 🌐 Julia | 📅 2014-12-09 **\[Deprecated]**
* [Simple MCMC](https://github.com/fredo-dedup/SimpleMCMC.jl) ⚠️ Archived - basic MCMC sampler implemented in Julia. **\[Deprecated]**
* [DA](https://github.com/trthatcher/DiscriminantAnalysis.jl) ⭐ 10 | 🐛 5 | 🌐 Julia | 📅 2021-01-10 - Julia package for Regularized Discriminant Analysis.
* [Decision Tree](https://github.com/bensadeghi/DecisionTree.jl) ⭐ 9 | 🐛 0 | 🌐 Julia | 📅 2022-05-05 - Decision Tree Classifier and Regressor.
* [Naive Bayes](https://github.com/nutsiepully/NaiveBayes.jl) ⭐ 8 | 🐛 0 | 🌐 Julia | 📅 2013-06-03 - Simple Naive Bayes implementation in Julia. **\[Deprecated]**
* [CluGen](https://github.com/clugen/CluGen.jl/) ⭐ 8 | 🐛 0 | 🌐 Julia | 📅 2024-06-04 - Multidimensional cluster generation in Julia.
* [Flux](https://fluxml.ai/) - Relax! Flux is the ML library that doesn't make you tensor

<a name="julia-natural-language-processing"></a>

#### Natural Language Processing

* [Text Analysis](https://github.com/JuliaText/TextAnalysis.jl) ⭐ 379 | 🐛 37 | 🌐 Julia | 📅 2025-11-17 - Julia package for text analysis.
* [Word Tokenizers](https://github.com/JuliaText/WordTokenizers.jl) ⭐ 100 | 🐛 12 | 🌐 Julia | 📅 2021-12-30 - Tokenizers for Natural Language Processing in Julia
* [Embeddings](https://github.com/JuliaText/Embeddings.jl) ⭐ 82 | 🐛 11 | 🌐 Julia | 📅 2024-03-08 - Functions and data dependencies for loading various word embeddings
* [Languages](https://github.com/JuliaText/Languages.jl) ⭐ 56 | 🐛 7 | 🌐 Julia | 📅 2025-10-25 - Julia package for working with various human languages
* [Topic Models](https://github.com/slycoder/TopicModels.jl) ⭐ 38 | 🐛 0 | 🌐 Julia | 📅 2020-05-31 - TopicModels for Julia. **\[Deprecated]**
* [WordNet](https://github.com/JuliaText/WordNet.jl) ⭐ 36 | 🐛 3 | 🌐 Julia | 📅 2023-05-17 - A Julia package for Princeton's WordNet
* [Corpus Loaders](https://github.com/JuliaText/CorpusLoaders.jl) ⭐ 32 | 🐛 11 | 🌐 Julia | 📅 2022-09-17 - A Julia package providing a variety of loaders for various NLP corpora.

<a name="julia-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [Gadfly](https://github.com/GiovineItalia/Gadfly.jl) ⭐ 1,921 | 🐛 276 | 🌐 Julia | 📅 2025-10-04 - Crafty statistical graphics for Julia.
* [DataFrames](https://github.com/JuliaData/DataFrames.jl) ⭐ 1,812 | 🐛 160 | 🌐 Julia | 📅 2026-02-20 - library for working with tabular data in Julia.
* [Distributions](https://github.com/JuliaStats/Distributions.jl) ⭐ 1,184 | 🐛 465 | 🌐 Julia | 📅 2026-02-16 - A Julia package for probability distributions and associated functions.
* [LightGraphs](https://github.com/JuliaGraphs/LightGraphs.jl) ⚠️ Archived - Graph modelling and analysis.
* [Data Frames Meta](https://github.com/JuliaData/DataFramesMeta.jl) ⭐ 497 | 🐛 38 | 🌐 Julia | 📅 2025-11-24 - Metaprogramming tools for DataFrames.
* [Time Series](https://github.com/JuliaStats/TimeSeries.jl) ⭐ 367 | 🐛 50 | 🌐 Julia | 📅 2026-01-26 - Time series toolkit for Julia.
* [Hypothesis Tests](https://github.com/JuliaStats/HypothesisTests.jl) ⭐ 314 | 🐛 84 | 🌐 Julia | 📅 2025-11-24 - Hypothesis tests for Julia.
* [RDataSets](https://github.com/johnmyleswhite/RDatasets.jl) ⭐ 166 | 🐛 19 | 🌐 R | 📅 2026-01-03 - Julia package for loading many of the data sets available in R.
* [Stats](https://github.com/JuliaStats/StatsKit.jl) ⭐ 142 | 🐛 3 | 🌐 Julia | 📅 2022-10-17 - Statistical tests for Julia.
* [Data Read](https://github.com/queryverse/ReadStat.jl) ⭐ 80 | 🐛 11 | 🌐 Julia | 📅 2023-12-29 - Read files from Stata, SAS, and SPSS.
* [Data Arrays](https://github.com/JuliaStats/DataArrays.jl) ⚠️ Archived - Data structures that allow missing values. **\[Deprecated]**
* [Graph Layout](https://github.com/IainNZ/GraphLayout.jl) ⚠️ Archived - Graph layout algorithms in pure Julia.
* [Julia Data](https://github.com/nfoti/JuliaData) ⭐ 6 | 🐛 0 | 🌐 Julia | 📅 2013-09-04 - library for working with tabular data in Julia. **\[Deprecated]**
* [Sampling](https://github.com/lindahua/Sampling.jl) ⭐ 1 | 🐛 1 | 🌐 Julia | 📅 2014-06-19 - Basic sampling algorithms for Julia.

<a name="julia-misc-stuff--presentations"></a>

#### Misc Stuff / Presentations

* [Images](https://github.com/JuliaImages/Images.jl) ⭐ 551 | 🐛 41 | 🌐 Julia | 📅 2026-01-31 - An image library for Julia.
* [DSP](https://github.com/JuliaDSP/DSP.jl) ⭐ 417 | 🐛 87 | 🌐 Julia | 📅 2026-02-12 - Digital Signal Processing (filtering, periodograms, spectrograms, window functions).
* [SignalProcessing](https://github.com/JuliaDSP/DSP.jl) ⭐ 417 | 🐛 87 | 🌐 Julia | 📅 2026-02-12 - Signal Processing tools for Julia.
* [DataDeps](https://github.com/oxinabox/DataDeps.jl) ⭐ 159 | 🐛 37 | 🌐 Julia | 📅 2024-02-13 - Reproducible data setup for reproducible science.
* [JuliaCon Presentations](https://github.com/JuliaCon/presentations) ⭐ 70 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2016-03-07 - Presentations for JuliaCon.

<a name="kotlin"></a>

## Kotlin

<a name="kotlin-deep-learning"></a>

#### Deep Learning

* [KotlinDL](https://github.com/JetBrains/KotlinDL) ⭐ 1,569 | 🐛 81 | 🌐 Kotlin | 📅 2024-05-31 - Deep learning framework written in Kotlin.

<a name="lua"></a>

## Lua

<a name="lua-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Torch7](http://torch.ch/)
  * [wav2letter](https://github.com/facebookresearch/wav2letter) ⭐ 6,445 | 🐛 107 | 🌐 C++ | 📅 2026-01-12 - a simple and efficient end-to-end Automatic Speech Recognition (ASR) system from Facebook AI Research.
  * [nn](https://github.com/torch/nn) ⭐ 1,357 | 🐛 173 | 🌐 Lua | 📅 2021-01-12 - Neural Network package for Torch.
  * [torchnet](https://github.com/torchnet/torchnet) ⚠️ Archived - framework for torch which provides a set of abstractions aiming at encouraging code re-use as well as encouraging modular programming.
  * [rnn](https://github.com/Element-Research/rnn) ⭐ 943 | 🐛 78 | 🌐 Lua | 📅 2017-12-21 - A Recurrent Neural Network library that extends Torch's nn. RNNs, LSTMs, GRUs, BRNNs, BLSTMs, etc.
  * [OverFeat](https://github.com/sermanet/OverFeat) ⭐ 601 | 🐛 34 | 🌐 C | 📅 2014-08-12 - A state-of-the-art generic dense feature extractor. **\[Deprecated]**
  * [autograd](https://github.com/twitter/torch-autograd) ⚠️ Archived - Autograd automatically differentiates native Torch code. Inspired by the original Python version.
  * [dp](https://github.com/nicholas-leonard/dp) ⭐ 339 | 🐛 27 | 🌐 Lua | 📅 2016-09-01 - A deep learning library designed for streamlining research and development using the Torch7 distribution. It emphasizes flexibility through the elegant use of object-oriented design patterns. **\[Deprecated]**
  * [cutorch](https://github.com/torch/cutorch) ⭐ 339 | 🐛 144 | 🌐 Cuda | 📅 2017-09-11 - Torch CUDA Implementation.
  * [nngraph](https://github.com/torch/nngraph) ⭐ 302 | 🐛 29 | 🌐 Lua | 📅 2017-08-01 - This package provides graphical computation for nn library in Torch7.
  * [cunn](https://github.com/torch/cunn) ⭐ 213 | 🐛 57 | 🌐 Cuda | 📅 2019-08-27 - Torch CUDA Neural Network Implementation.
  * [optim](https://github.com/torch/optim) ⭐ 196 | 🐛 31 | 🌐 Lua | 📅 2017-11-27 - An optimization library for Torch. SGD, Adagrad, Conjugate-Gradient, LBFGS, RProp and more.
  * [dpnn](https://github.com/Element-Research/dpnn) ⭐ 193 | 🐛 16 | 🌐 Lua | 📅 2017-05-26 - Many useful features that aren't part of the main nn package.
  * [manifold](https://github.com/clementfarabet/manifold) ⭐ 142 | 🐛 7 | 🌐 Lua | 📅 2017-07-03 - A package to manipulate manifolds.
  * [nnx](https://github.com/clementfarabet/lua---nnx) ⭐ 98 | 🐛 12 | 🌐 Lua | 📅 2017-05-30 - A completely unstable and experimental package that extends Torch's builtin nn library.
  * [unsup](https://github.com/koraykv/unsup) ⭐ 86 | 🐛 10 | 🌐 Lua | 📅 2017-05-23 - A package for unsupervised learning in Torch. Provides modules that are compatible with nn (LinearPsd, ConvPsd, AutoEncoder, ...), and self-contained algorithms (k-means, PCA). **\[Deprecated]**
  * [cephes](https://github.com/deepmind/torch-cephes) ⚠️ Archived - Cephes mathematical functions library, wrapped for Torch. Provides and wraps the 180+ special mathematical functions from the Cephes mathematical library, developed by Stephen L. Moshier. It is used, among many other places, at the heart of SciPy. **\[Deprecated]**
  * [signal](https://github.com/soumith/torch-signal) ⭐ 49 | 🐛 5 | 🌐 Lua | 📅 2017-07-02 - A signal processing toolbox for Torch-7. FFT, DCT, Hilbert, cepstrums, stft.
  * [svm](https://github.com/koraykv/torch-svm) ⭐ 44 | 🐛 9 | 🌐 C++ | 📅 2016-05-31 - Torch-SVM library. **\[Deprecated]**
  * [graph](https://github.com/torch/graph) ⭐ 38 | 🐛 1 | 🌐 Lua | 📅 2016-11-24 - Graph package for Torch. **\[Deprecated]**
  * [randomkit](https://github.com/deepmind/torch-randomkit) ⭐ 34 | 🐛 9 | 🌐 Lua | 📅 2019-04-23 - Numpy's randomkit, wrapped for Torch. **\[Deprecated]**
  * [imgraph](https://github.com/clementfarabet/lua---imgraph) ⭐ 22 | 🐛 5 | 🌐 C | 📅 2017-10-27 - An image/graph library for Torch. This package provides routines to construct graphs on images, segment them, build trees out of them, and convert them back to images. **\[Deprecated]**
  * [fex](https://github.com/koraykv/fex) ⭐ 10 | 🐛 0 | 🌐 Lua | 📅 2014-01-31 - A package for feature extraction in Torch. Provides SIFT and dSIFT modules. **\[Deprecated]**
  * [videograph](https://github.com/clementfarabet/videograph) ⭐ 9 | 🐛 0 | 🌐 C | 📅 2013-07-04 - A video/graph library for Torch. This package provides routines to construct graphs on videos, segment them, build trees out of them, and convert them back to videos. **\[Deprecated]**
  * [OpenGM](https://github.com/clementfarabet/lua---opengm) ⭐ 8 | 🐛 1 | 🌐 C++ | 📅 2012-03-02 - OpenGM is a C++ library for graphical modelling, and inference. The Lua bindings provide a simple way of describing graphs, from Lua, and then optimizing them with OpenGM. **\[Deprecated]**
  * [saliency](https://github.com/marcoscoffier/torch-saliency) ⭐ 7 | 🐛 0 | 🌐 Lua | 📅 2013-12-13 - code and tools around integral images. A library for finding interest points based on fast integral histograms. **\[Deprecated]**
  * [kernel smoothing](https://github.com/rlowrance/kernel-smoothers) ⭐ 5 | 🐛 1 | 🌐 Lua | 📅 2012-10-26 - KNN, kernel-weighted average, local linear regression smoothers. **\[Deprecated]**
  * [stitch](https://github.com/marcoscoffier/lua---stitch) ⭐ 4 | 🐛 0 | 🌐 Lua | 📅 2012-05-06 - allows us to use hugin to stitch images and apply same stitching to a video sequence. **\[Deprecated]**
  * [LuaSHKit](https://github.com/ocallaco/LuaSHkit) ⭐ 3 | 🐛 0 | 🌐 C++ | 📅 2014-05-21 - A Lua wrapper around the Locality sensitive hashing library SHKit **\[Deprecated]**
  * [sfm](https://github.com/marcoscoffier/lua---sfm) ⭐ 3 | 🐛 0 | 🌐 C | 📅 2012-02-15 - A bundle adjustment/structure from motion package. **\[Deprecated]**
  * [lbfgs](https://github.com/clementfarabet/lbfgs) ⭐ 2 | 🐛 0 | 🌐 Lua | 📅 2013-03-08 - FFI Wrapper for liblbfgs. **\[Deprecated]**
  * [vowpalwabbit](https://github.com/clementfarabet/vowpal_wabbit) ⭐ 2 | 🐛 0 | 🌐 C++ | 📅 2012-05-09 - An old vowpalwabbit interface to torch. **\[Deprecated]**
  * [spaghetti](https://github.com/MichaelMathieu/lua---spaghetti) ⭐ 2 | 🐛 0 | 🌐 Lua | 📅 2013-08-05 - Spaghetti (sparse linear) module for torch7 by @MichaelMathieu **\[Deprecated]**
* [Lunum](https://github.com/jzrake/lunum) ⭐ 40 | 🐛 2 | 🌐 C | 📅 2012-02-20 **\[Deprecated]**
* [Keras GPT Copilot](https://github.com/fabprezja/keras-gpt-copilot) ⭐ 28 | 🐛 0 | 🌐 Python | 📅 2023-09-23 - A python package that integrates an LLM copilot inside the keras model development workflow.
* [Numeric Lua](http://numlua.luaforge.net/)
* [Lunatic Python](https://labix.org/lunatic-python)
* [SciLua](http://scilua.org/)
* [Lua - Numerical Algorithms](https://bitbucket.org/lucashnegri/lna) **\[Deprecated]**

<a name="lua-demos-and-scripts"></a>

#### Demos and Scripts

* [Core torch7 demos repository](https://github.com/e-lab/torch7-demos) ⭐ 43 | 🐛 0 | 🌐 Lua | 📅 2017-02-15.
  * linear-regression, logistic-regression
  * face detector (training and detection as separate demos)
  * mst-based-segmenter
  * train-a-digit-classifier
  * train-autoencoder
  * optical flow demo
  * train-on-housenumbers
  * train-on-cifar
  * tracking with deep nets
  * kinect demo
  * filter-bank visualization
  * saliency-networks
* [torch-datasets](https://github.com/rosejn/torch-datasets) ⭐ 36 | 🐛 3 | 🌐 Lua | 📅 2014-03-12 - Scripts to load several popular datasets including:
  * BSR 500
  * CIFAR-10
  * COIL
  * Street View House Numbers
  * MNIST
  * NORB
* [Training a Convnet for the Galaxy-Zoo Kaggle challenge(CUDA demo)](https://github.com/soumith/galaxyzoo) ⭐ 35 | 🐛 1 | 🌐 Lua | 📅 2014-11-01
* [Atari2600](https://github.com/fidlej/aledataset) ⭐ 19 | 🐛 2 | 🌐 Lua | 📅 2014-05-09 - Scripts to generate a dataset with static frames from the Arcade Learning Environment.

<a name="matlab"></a>

## Matlab

<a name="matlab-computer-vision"></a>

#### Computer Vision

* [Contourlets](http://www.ifp.illinois.edu/~minhdo/software/contourlet_toolbox.tar) - MATLAB source code that implements the contourlet transform and its utility functions.
* [Shearlets](https://www3.math.tu-berlin.de/numerik/www.shearlab.org/software) - MATLAB code for shearlet transform.
* [Curvelets](http://www.curvelet.org/software.html) - The Curvelet transform is a higher dimensional generalization of the Wavelet transform designed to represent images at different scales and different angles.
* [Bandlets](http://www.cmap.polytechnique.fr/~peyre/download/) - MATLAB code for bandlet transform.
* [mexopencv](https://kyamagu.github.io/mexopencv/) - Collection and a development kit of MATLAB mex functions for OpenCV library.

<a name="matlab-natural-language-processing"></a>

#### Natural Language Processing

* [NLP](https://amplab.cs.berkeley.edu/an-nlp-library-for-matlab/) - A NLP library for Matlab.

<a name="matlab-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Caffe](https://github.com/BVLC/caffe) ⭐ 34,839 | 🐛 1,175 | 🌐 C++ | 📅 2024-07-31 - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [MXNet](https://github.com/apache/incubator-mxnet/) ⚠️ Archived - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [Pattern Recognition and Machine Learning](https://github.com/PRML/PRMLT) ⭐ 6,204 | 🐛 0 | 🌐 MATLAB | 📅 2020-03-04 - This package contains the matlab implementation of the algorithms described in the book Pattern Recognition and Machine Learning by C. Bishop.
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) ⭐ 1,619 | 🐛 87 | 🌐 C++ | 📅 2024-04-01 - An Open-Source SVM Library on GPUs and CPUs
* [Machine Learning in MatLab/Octave](https://github.com/trekhleb/machine-learning-octave) ⭐ 894 | 🐛 0 | 🌐 MATLAB | 📅 2025-11-23 - Examples of popular machine learning algorithms (neural networks, linear/logistic regressions, K-Means, etc.) with code examples and mathematics behind them being explained.
* [Machine Learning Module](https://github.com/josephmisiti/machine-learning-module) ⭐ 477 | 🐛 0 | 🌐 Objective-C | 📅 2011-04-28 - Class on machine w/ PDF, lectures, code
* [Pattern Recognition Toolbox](https://github.com/covartech/PRT) ⭐ 146 | 🐛 19 | 🌐 MATLAB | 📅 2025-10-28 - A complete object-oriented environment for machine learning in Matlab.
* [MOCluGen](https://github.com/clugen/MOCluGen/) ⭐ 5 | 🐛 0 | 🌐 MATLAB | 📅 2024-08-05 - Multidimensional cluster generation in MATLAB/Octave.
* [Training a deep autoencoder or a classifier
  on MNIST digits](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html) - Training a deep autoencoder or a classifier
  on MNIST digits\[DEEP LEARNING].
* [Convolutional-Recursive Deep Learning for 3D Object Classification](https://www.socher.org/index.php/Main/Convolutional-RecursiveDeepLearningFor3DObjectClassification) - Convolutional-Recursive Deep Learning for 3D Object Classification\[DEEP LEARNING].
* [Spider](https://people.kyb.tuebingen.mpg.de/spider/) - The spider is intended to be a complete object orientated environment for machine learning in Matlab.
* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/#matlab) - A Library for Support Vector Machines.
* [LibLinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/#download) - A Library for Large Linear Classification.
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search. Optunity is written in Python but interfaces seamlessly with MATLAB.

<a name="matlab-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [ParaMonte](https://github.com/cdslaborg/paramonte) ⭐ 300 | 🐛 20 | 🌐 Fortran | 📅 2025-12-18 - A general-purpose MATLAB library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [matlab\_bgl](https://www.cs.purdue.edu/homes/dgleich/packages/matlab_bgl/) - MatlabBGL is a Matlab package for working with graphs.
* [gaimc](https://www.mathworks.com/matlabcentral/fileexchange/24134-gaimc---graph-algorithms-in-matlab-code) - Efficient pure-Matlab implementations of graph algorithms to complement MatlabBGL's mex functions.

<a name="net"></a>

## .NET

<a name="net-computer-vision"></a>

#### Computer Vision

* [OpenCVDotNet](https://code.google.com/archive/p/opencvdotnet) - A wrapper for the OpenCV project to be used with .NET applications.
* [Emgu CV](http://www.emgu.com/wiki/index.php/Main_Page) - Cross platform wrapper of OpenCV which can be compiled in Mono to be run on Windows, Linus, Mac OS X, iOS, and Android.
* [AForge.NET](http://www.aforgenet.com/framework/) - Open source C# framework for developers and researchers in the fields of Computer Vision and Artificial Intelligence. Development has now shifted to GitHub.
* [Accord.NET](http://accord-framework.net) - Together with AForge.NET, this library can provide image processing and computer vision algorithms to Windows, Windows RT and Windows Phone. Some components are also available for Java and Android.

<a name="net-natural-language-processing"></a>

#### Natural Language Processing

* [Stanford.NLP for .NET](https://github.com/sergey-tihon/Stanford.NLP.NET/) ⭐ 610 | 🐛 1 | 🌐 C# | 📅 2024-03-22 - A full port of Stanford NLP packages to .NET and also available precompiled as a NuGet package.

<a name="net-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [ML.NET](https://github.com/dotnet/machinelearning) ⭐ 9,326 | 🐛 994 | 🌐 C# | 📅 2026-02-18 - ML.NET is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers. ML.NET was originally developed in Microsoft Research and evolved into a significant framework over the last decade and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel and more.
* [GeneticSharp](https://github.com/giacomelli/GeneticSharp) ⭐ 1,356 | 🐛 12 | 🌐 C# | 📅 2025-11-13 - Multi-platform genetic algorithm library for .NET Core and .NET Framework. The library has several implementations of GA operators, like: selection, crossover, mutation, reinsertion and termination.
* [MxNet.Sharp](https://github.com/tech-quantum/MxNet.Sharp) ⭐ 151 | 🐛 3 | 🌐 C# | 📅 2023-04-12 - .NET Standard bindings for Apache MxNet with Imperative, Symbolic and Gluon Interface for developing, training and deploying Machine Learning models in C#. <https://mxnet.tech-quantum.com/>
* [Vulpes](https://github.com/fsprojects/Vulpes) ⚠️ Archived - Deep belief and deep learning implementation written in F# and leverages CUDA GPU execution with Alea.cuBase.
* [Synapses](https://github.com/mrdimosthenis/Synapses) ⭐ 73 | 🐛 0 | 📅 2021-09-23 - Neural network library in F#.
* [Accord-Framework](http://accord-framework.net/) -The Accord.NET Framework is a complete framework for building machine learning, computer vision, computer audition, signal processing and statistical applications.
* [Accord.MachineLearning](https://www.nuget.org/packages/Accord.MachineLearning/) - Support Vector Machines, Decision Trees, Naive Bayesian models, K-means, Gaussian Mixture models and general algorithms such as Ransac, Cross-validation and Grid-Search for machine-learning applications. This package is part of the Accord.NET Framework.
* [DiffSharp](https://diffsharp.github.io/DiffSharp/) - An automatic differentiation (AD) library providing exact and efficient derivatives (gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products) for machine learning and optimization applications. Operations can be nested to any level, meaning that you can compute exact higher-order derivatives and differentiate functions that are internally making use of differentiation, for applications such as hyperparameter optimization.
* [Encog](https://www.nuget.org/packages/encog-dotnet-core/) - An advanced neural network and machine learning framework. Encog contains classes to create a wide variety of networks, as well as support classes to normalize and process data for these neural networks. Encog trains using multithreaded resilient propagation. Encog can also make use of a GPU to further speed processing time. A GUI based workbench is also provided to help model and train neural networks.
* [Infer.NET](https://dotnet.github.io/infer/) - Infer.NET is a framework for running Bayesian inference in graphical models. One can use Infer.NET to solve many different kinds of machine learning problems, from standard problems like classification, recommendation or clustering through customized solutions to domain-specific problems. Infer.NET has been used in a wide variety of domains including information retrieval, bioinformatics, epidemiology, vision, and many others.
* [Neural Network Designer](https://sourceforge.net/projects/nnd/) - DBMS management system and designer for neural networks. The designer application is developed using WPF, and is a user interface which allows you to design your neural network, query the network, create and configure chat bots that are capable of asking questions and learning from your feedback. The chat bots can even scrape the internet for information to return in their output as well as to use for learning.

<a name="net-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [numl](https://www.nuget.org/packages/numl/) - numl is a machine learning library intended to ease the use of using standard modelling techniques for both prediction and clustering.
* [Math.NET Numerics](https://www.nuget.org/packages/MathNet.Numerics/) - Numerical foundation of the Math.NET project, aiming to provide methods and algorithms for numerical computations in science, engineering and everyday use. Supports .Net 4.0, .Net 3.5 and Mono on Windows, Linux and Mac; Silverlight 5, WindowsPhone/SL 8, WindowsPhone 8.1 and Windows 8 with PCL Portable Profiles 47 and 344; Android/iOS with Xamarin.
* [Sho](https://www.microsoft.com/en-us/research/project/sho-the-net-playground-for-data/) - Sho is an interactive environment for data analysis and scientific computing that lets you seamlessly connect scripts (in IronPython) with compiled code (in .NET) to enable fast and flexible prototyping. The environment includes powerful and efficient libraries for linear algebra as well as data visualization that can be used from any .NET language, as well as a feature-rich interactive shell for rapid development.

<a name="objective-c"></a>

## Objective C

<a name="objective-c-general-purpose-machine-learning"></a>

### General-Purpose Machine Learning

* [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) ⭐ 902 | 🐛 1 | 🌐 Objective-C | 📅 2016-09-30 - Fast multilayer perceptron neural network library for iOS and Mac OS X. MLPNeuralNet predicts new examples by trained neural networks. It is built on top of the Apple's Accelerate Framework, using vectorized operations and hardware acceleration if available. **\[Deprecated]**
* [YCML](https://github.com/yconst/YCML) ⚠️ Archived - A Machine Learning framework for Objective-C and Swift (OS X / iOS).
* [MAChineLearning](https://github.com/gianlucabertani/MAChineLearning) ⭐ 37 | 🐛 0 | 🌐 Objective-C | 📅 2019-08-29 - An Objective-C multilayer perceptron library, with full support for training through backpropagation. Implemented using vDSP and vecLib, it's 20 times faster than its Java equivalent. Includes sample code for use from Swift.
* [BPN-NeuralNetwork](https://github.com/Kalvar/ios-BPN-NeuralNetwork) ⭐ 33 | 🐛 0 | 🌐 Objective-C | 📅 2015-11-15 - It implemented 3 layers of neural networks ( Input Layer, Hidden Layer and Output Layer ) and it was named Back Propagation Neural Networks (BPN). This network can be used in products recommendation, user behavior analysis, data mining and data analysis. **\[Deprecated]**
* [Multi-Perceptron-NeuralNetwork](https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork) ⭐ 24 | 🐛 0 | 🌐 Objective-C | 📅 2017-06-22 - It implemented multi-perceptrons neural network (ニューラルネットワーク) based on Back Propagation Neural Networks (BPN) and designed unlimited-hidden-layers.
* [KRKmeans-Algorithm](https://github.com/Kalvar/ios-KRKmeans-Algorithm) ⭐ 23 | 🐛 1 | 🌐 Objective-C | 📅 2016-06-10 - It implemented K-Means  clustering and classification algorithm. It could be used in data mining and image compression. **\[Deprecated]**
* [KRHebbian-Algorithm](https://github.com/Kalvar/ios-KRHebbian-Algorithm) ⭐ 13 | 🐛 0 | 🌐 Objective-C | 📅 2015-11-08 - It is a non-supervisory and self-learning algorithm (adjust the weights) in the neural network of Machine Learning. **\[Deprecated]**
* [KRFuzzyCMeans-Algorithm](https://github.com/Kalvar/ios-KRFuzzyCMeans-Algorithm) ⭐ 12 | 🐛 0 | 🌐 Objective-C | 📅 2016-01-27 - It implemented Fuzzy C-Means (FCM) the fuzzy clustering / classification algorithm on Machine Learning. It could be used in data mining and image compression. **\[Deprecated]**

<a name="ocaml"></a>

## OCaml

<a name="ocaml-general-purpose-machine-learning"></a>

### General-Purpose Machine Learning

* [TensorFlow](https://github.com/LaurentMazare/tensorflow-ocaml) ⭐ 287 | 🐛 5 | 🌐 OCaml | 📅 2019-07-06 - OCaml bindings for TensorFlow.
* [Oml](https://github.com/rleonid/oml) ⭐ 119 | 🐛 24 | 🌐 OCaml | 📅 2018-02-01 - A general statistics and machine learning library.
* [GPR](https://mmottl.github.io/gpr/) - Efficient Gaussian Process Regression in OCaml.
* [Libra-Tk](https://libra.cs.uoregon.edu) - Algorithms for learning and inference with discrete probabilistic models.

<a name="opencv"></a>

## OpenCV

<a name="opencv-ComputerVision and Text Detection"></a>

### OpenSource-Computer-Vision

* [OpenCV](https://github.com/opencv/opencv) ⭐ 86,243 | 🐛 2,723 | 🌐 C++ | 📅 2026-02-20 - A OpenSource Computer Vision Library

<a name="perl"></a>

## Perl

<a name="perl-data-analysis--data-visualization"></a>

### Data Analysis / Data Visualization

* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning), a pluggable architecture for data and image processing, which can
  be [used for machine learning](https://github.com/zenogantner/PDL-ML) ⭐ 14 | 🐛 0 | 🌐 Perl | 📅 2011-06-23.

<a name="perl-general-purpose-machine-learning"></a>

### General-Purpose Machine Learning

* [MXnet for Deep Learning, in Perl](https://github.com/apache/incubator-mxnet/tree/master/perl-package) ⚠️ Archived,
  also [released in CPAN](https://metacpan.org/pod/AI::MXNet).
* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning),
  using AWS machine learning platform from Perl.
* [Algorithm::SVMLight](https://metacpan.org/pod/Algorithm::SVMLight),
  implementation of Support Vector Machines with SVMLight under it. **\[Deprecated]**
* Several machine learning and artificial intelligence models are
  included in the [`AI`](https://metacpan.org/search?size=20\&q=AI)
  namespace. For instance, you can
  find [Naïve Bayes](https://metacpan.org/pod/AI::NaiveBayes).

<a name="perl6"></a>

## Perl 6

* [Support Vector Machines](https://github.com/titsuki/p6-Algorithm-LibSVM) ⭐ 8 | 🐛 1 | 🌐 C++ | 📅 2023-08-26
* [Naïve Bayes](https://github.com/titsuki/p6-Algorithm-NaiveBayes) ⭐ 3 | 🐛 1 | 🌐 Raku | 📅 2022-03-28

<a name="perl-6-data-analysis--data-visualization"></a>

### Data Analysis / Data Visualization

* [Perl Data Language](https://metacpan.org/pod/Paws::MachineLearning),
  a pluggable architecture for data and image processing, which can
  be
  [used for machine learning](https://github.com/zenogantner/PDL-ML) ⭐ 14 | 🐛 0 | 🌐 Perl | 📅 2011-06-23.

<a name="perl-6-general-purpose-machine-learning"></a>

### General-Purpose Machine Learning

<a name="php"></a>

## PHP

<a name="php-natural-language-processing"></a>

### Natural Language Processing

* [jieba-php](https://github.com/fukuball/jieba-php) ⭐ 1,374 | 🐛 4 | 🌐 PHP | 📅 2025-12-16 - Chinese Words Segmentation Utilities.

<a name="php-general-purpose-machine-learning"></a>

### General-Purpose Machine Learning

* [PredictionBuilder](https://github.com/denissimon/prediction-builder) ⭐ 115 | 🐛 0 | 🌐 PHP | 📅 2024-11-14 - A library for machine learning that builds predictions using a linear regression.
* [19 Questions](https://github.com/fulldecent/19-questions) ⭐ 15 | 🐛 0 | 🌐 PHP | 📅 2025-08-18 - A machine learning / bayesian inference assigning attributes to objects.
* [PHP-ML](https://gitlab.com/php-ai/php-ml) - Machine Learning library for PHP. Algorithms, Cross Validation, Neural Network, Preprocessing, Feature Extraction and much more in one library.
* [Rubix ML](https://github.com/RubixML) - A high-level machine learning (ML) library that lets you build programs that learn from data using the PHP language.

<a name="python"></a>

## Python

<a name="python-computer-vision"></a>

#### Computer Vision

* [face\_recognition](https://github.com/ageitgey/face_recognition) ⭐ 56,124 | 🐛 833 | 🌐 Python | 📅 2024-08-21 - Face recognition library that recognizes and manipulates faces from Python or from the command line.
* [timm](https://github.com/rwightman/pytorch-image-models) ⭐ 36,386 | 🐛 65 | 🌐 Python | 📅 2026-02-18 - PyTorch image models, scripts, pretrained weights -- ResNet, ResNeXT, EfficientNet, EfficientNetV2, NFNet, Vision Transformer, MixNet, MobileNet-V3/V2, RegNet, DPN, CSPNet, and more.
* [detectron2](https://github.com/facebookresearch/detectron2) ⭐ 34,088 | 🐛 589 | 🌐 Python | 📅 2025-11-17 - FAIR's next-generation research platform for object detection and segmentation. It is a ground-up rewrite of the previous version, Detectron, and is powered by the PyTorch deep learning framework.
* [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) ⭐ 33,793 | 🐛 358 | 🌐 C++ | 📅 2024-08-03 - A real-time multi-person keypoint detection library for body, face, hands, and foot estimation
* [Detectron](https://github.com/facebookresearch/Detectron) ⚠️ Archived - FAIR's software system that implements state-of-the-art object detection algorithms, including Mask R-CNN. It is written in Python and powered by the Caffe2 deep learning framework. **\[Deprecated]**
* [MLX](https://github.com/ml-explore/mlx) ⭐ 23,992 | 🐛 170 | 🌐 C++ | 📅 2026-02-20- MLX is an array framework for machine learning on Apple silicon, developed by Apple machine learning research.
* [deepface](https://github.com/serengil/deepface) ⭐ 22,231 | 🐛 6 | 🌐 Python | 📅 2026-02-10 - A lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for Python covering cutting-edge models such as VGG-Face, FaceNet, OpenFace, DeepFace, DeepID, Dlib and ArcFace.
* [albumentations](https://github.com/albu/albumentations) ⚠️ Archived - А fast and framework agnostic image augmentation library that implements a diverse set of augmentation techniques. Supports classification, segmentation, detection out of the box. Was used to win a number of Deep Learning competitions at Kaggle, Topcoder and those that were a part of the CVPR workshops.
* [segmentation\_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) ⭐ 11,340 | 🐛 88 | 🌐 Python | 📅 2025-12-23 - A PyTorch-based toolkit that offers pre-trained segmentation models for computer vision tasks. It simplifies the development of image segmentation applications by providing a collection of popular architecture implementations, such as UNet and PSPNet, along with pre-trained weights, making it easier for researchers and developers to achieve high-quality pixel-level object segmentation in images.
* [Exadel CompreFace](https://github.com/exadel-inc/CompreFace) ⭐ 7,785 | 🐛 232 | 🌐 Java | 📅 2024-10-05 - face recognition system that can be easily integrated into any system without prior machine learning skills. CompreFace provides REST API for face recognition, face verification, face detection, face mask detection, landmark detection, age, and gender recognition and is easily deployed with docker.
* [Scikit-Image](https://github.com/scikit-image/scikit-image) ⭐ 6,459 | 🐛 871 | 🌐 Python | 📅 2026-02-18 - A collection of algorithms for image processing in Python.
* [Scikit-Opt](https://github.com/guofei9987/scikit-opt) ⭐ 6,371 | 🐛 70 | 🌐 Python | 📅 2025-08-31 - Swarm Intelligence in Python (Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Algorithm, Immune Algorithm, Artificial Fish Swarm Algorithm in Python)
* [pytessarct](https://github.com/madmaze/pytesseract) ⭐ 6,315 | 🐛 18 | 🌐 Python | 📅 2026-01-19 - Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and "read" the text embedded in images. Python-tesseract is a wrapper for [Google's Tesseract-OCR Engine](https://github.com/tesseract-ocr/tesseract) ⭐ 72,474 | 🐛 466 | 🌐 C++ | 📅 2026-02-20.
* [segmentation\_models](https://github.com/qubvel/segmentation_models) ⭐ 4,918 | 🐛 274 | 🌐 Python | 📅 2024-08-21 - A TensorFlow Keras-based toolkit that offers pre-trained segmentation models for computer vision tasks. It simplifies the development of image segmentation applications by providing a collection of popular architecture implementations, such as UNet and PSPNet, along with pre-trained weights, making it easier for researchers and developers to achieve high-quality pixel-level object segmentation in images.
* [imutils](https://github.com/jrosebr1/imutils) ⭐ 4,593 | 🐛 164 | 🌐 Python | 📅 2024-06-24 - A library containing Convenience functions to make basic image processing operations such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images easier with OpenCV and Python.
* [Deep High-Resolution-Net](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) ⭐ 4,462 | 🐛 209 | 🌐 Cuda | 📅 2024-08-30 - A PyTorch implementation of CVPR2019 paper "Deep High-Resolution Representation Learning for Human Pose Estimation"
* [lightly](https://github.com/lightly-ai/lightly) ⭐ 3,681 | 🐛 83 | 🌐 Python | 📅 2026-02-19 - Lightly is a computer vision framework for self-supervised learning.
* [computer-vision-in-action](https://github.com/Charmve/computer-vision-in-action) ⭐ 2,831 | 🐛 60 | 🌐 Jupyter Notebook | 📅 2024-05-27 - as known as `L0CV`, is a new generation of computer vision open source online learning media, a cross-platform interactive learning framework integrating graphics, source code and HTML. the L0CV ecosystem — Notebook, Datasets, Source Code, and from Diving-in to Advanced — as well as the L0CV Hub.
* [PCV](https://github.com/jesolem/PCV) ⭐ 1,959 | 🐛 26 | 🌐 Python | 📅 2020-12-28 - Open source Python module for computer vision. **\[Deprecated]**
* [retinaface](https://github.com/serengil/retinaface) ⭐ 1,908 | 🐛 1 | 🌐 Python | 📅 2025-08-11 - deep learning based cutting-edge facial detector for Python coming with facial landmarks
* [LightlyTrain](https://github.com/lightly-ai/lightly-train) ⭐ 1,321 | 🐛 52 | 🌐 Python | 📅 2026-02-20 - Pretrain computer vision models on unlabeled data for industrial applications
* [TF-GAN](https://github.com/tensorflow/gan) ⭐ 967 | 🐛 21 | 🌐 Jupyter Notebook | 📅 2025-01-16 - TF-GAN is a lightweight library for training and evaluating Generative Adversarial Networks (GANs).
* [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt) ⭐ 862 | 🐛 25 | 🌐 Python | 📅 2022-10-15 - A PyTorch implementation of Justin Johnson's neural-style (neural style transfer).
* [Lucent](https://github.com/greentfrapp/lucent) ⭐ 653 | 🐛 22 | 🌐 Python | 📅 2025-03-21 - Tensorflow and OpenAI Clarity's Lucid adapted for PyTorch.
* [Detecto](https://github.com/alankbi/detecto) ⭐ 625 | 🐛 48 | 🌐 Python | 📅 2024-07-25 - Train and run a computer vision model with 5-10 lines of code.
* [Vigranumpy](https://github.com/ukoethe/vigra) ⭐ 438 | 🐛 101 | 🌐 C++ | 📅 2025-12-23 - Python bindings for the VIGRA C++ computer vision library.
* [joliGEN](https://github.com/jolibrain/joliGEN) ⭐ 280 | 🐛 32 | 🌐 Python | 📅 2026-01-16 - Generative AI Image Toolset with GANs and Diffusion for Real-World Applications.
* [dockerface](https://github.com/natanielruiz/dockerface) ⭐ 191 | 🐛 4 | 🌐 Dockerfile | 📅 2020-06-20 - Easy to install and use deep learning Faster R-CNN face detection for images and video in a docker container. **\[Deprecated]**
* [neural-dream](https://github.com/ProGamerGov/neural-dream) ⭐ 148 | 🐛 10 | 🌐 Python | 📅 2021-09-29 - A PyTorch implementation of DeepDream.
* [dream-creator](https://github.com/ProGamerGov/dream-creator) ⭐ 71 | 🐛 7 | 🌐 Python | 📅 2022-08-05 - A PyTorch implementation of DeepDream. Allows individuals to quickly and easily train their own custom GoogleNet models with custom datasets for DeepDream.
* [Learnergy](https://github.com/gugarosa/learnergy) ⭐ 69 | 🐛 3 | 🌐 Python | 📅 2026-01-01 - Energy-based machine learning models built upon PyTorch.
* [PyTorchCV](https://github.com/donnyyou/PyTorchCV) ⭐ 53 | 🐛 0 | 🌐 Shell | 📅 2019-02-09 - A PyTorch-Based Framework for Deep Learning in Computer Vision.
* [IoT Owl](https://github.com/Ret2Me/IoT-Owl) ⭐ 9 | 🐛 0 | 🌐 Python | 📅 2021-07-28 - Light face detection and recognition system with huge possibilities, based on Microsoft Face API and TensorFlow made for small IoT devices like raspberry pi.
* [SimpleCV](http://simplecv.org/) - An open source computer vision framework that gives access to several high-powered computer vision libraries, such as OpenCV. Written on Python and runs on Mac, Windows, and Ubuntu Linux.
* [OpenFace](https://cmusatyalab.github.io/openface/) - Free and open source face recognition with deep neural networks.
* [Gempix2](https://gempix2.site) - Free production platform for text-to-image generation using Nano Banana V2 model.
* [Self-supervised learning](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html)
* [OpenVisionAPI](https://github.com/openvisionapi) - Open source computer vision API based on open source models.

<a name="python-natural-language-processing"></a>

#### Natural Language Processing

* [Transformers](https://github.com/huggingface/transformers) ⭐ 156,761 | 🐛 2,280 | 🌐 Python | 📅 2026-02-21 - A deep learning library containing thousands of pre-trained models on different tasks. The goto place for anything related to Large Language Models.
* [jieba](https://github.com/fxsjy/jieba#jieba-1) ⭐ 34,766 | 🐛 697 | 🌐 Python | 📅 2024-08-21 - Chinese Words Segmentation Utilities.
* [spaCy](https://github.com/explosion/spaCy) ⭐ 33,226 | 🐛 189 | 🌐 Python | 📅 2025-11-27 - Industrial strength NLP with Python and Cython.
* [Haystack](https://github.com/deepset-ai/haystack) ⭐ 24,249 | 🐛 108 | 🌐 MDX | 📅 2026-02-20 - A framework for building industrial-strength applications with Transformer models and LLMs.
* [Rasa](https://github.com/RasaHQ/rasa) ⭐ 21,056 | 🐛 144 | 🌐 Python | 📅 2026-01-29 - A "machine learning framework to automate text-and voice-based conversations."
* [CometLLM](https://github.com/comet-ml/comet-llm) ⭐ 17,792 | 🐛 171 | 🌐 Python | 📅 2026-02-21 - Track, log, visualize and evaluate your LLM prompts and prompt chains.
* [Fuzzy Wuzzy](https://github.com/seatgeek/fuzzywuzzy) ⚠️ Archived - Fuzzy String Matching in Python.
* [Pattern](https://github.com/clips/pattern) ⭐ 8,853 | 🐛 176 | 🌐 Python | 📅 2024-06-10 - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.
* [DeepPavlov](https://github.com/deepmipt/DeepPavlov/) ⭐ 6,968 | 🐛 50 | 🌐 Python | 📅 2025-08-06 - conversational AI library with many pre-trained Russian NLP models.
* [pkuseg-python](https://github.com/lancopku/pkuseg-python) ⭐ 6,699 | 🐛 135 | 🌐 Python | 📅 2022-11-05 - A better version of Jieba, developed by Peking University.
* [SnowNLP](https://github.com/isnowfy/snownlp) ⭐ 6,604 | 🐛 44 | 🌐 Python | 📅 2020-01-19 - A library for processing Chinese text.
* [DrQA](https://github.com/facebookresearch/DrQA) ⚠️ Archived - Reading Wikipedia to answer open-domain questions.
* [Dedupe](https://github.com/dedupeio/dedupe) ⭐ 4,442 | 🐛 88 | 🌐 Python | 📅 2025-07-29 - A python library for accurate and scalable fuzzy matching, record deduplication and entity-resolution.
* [Snips NLU](https://github.com/snipsco/snips-nlu) ⭐ 3,958 | 🐛 67 | 🌐 Python | 📅 2023-05-22 - Natural Language Understanding library for intent classification and entity extraction
* [Polyglot](https://github.com/aboSamoor/polyglot) ⭐ 2,362 | 🐛 170 | 🌐 Python | 📅 2023-11-10 - Multilingual text (NLP) processing toolkit.
* [textacy](https://github.com/chartbeat-labs/textacy) ⭐ 2,234 | 🐛 35 | 🌐 Python | 📅 2023-09-22 - higher-level NLP built on Spacy.
* [jellyfish](https://github.com/jamesturk/jellyfish) ⭐ 2,192 | 🐛 4 | 🌐 Jupyter Notebook | 📅 2025-12-15 - a python library for doing approximate and phonetic matching of strings.
* [NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER) ⭐ 1,721 | 🐛 91 | 🌐 Python | 📅 2023-03-24 - Named-entity recognition using neural networks providing state-of-the-art-results
* [Quepy](https://github.com/machinalis/quepy) ⭐ 1,265 | 🐛 28 | 🌐 Python | 📅 2020-12-29 - A python framework to transform natural language questions to queries in a database query language.
* [CLTK](https://github.com/cltk/cltk) ⭐ 894 | 🐛 5 | 🌐 Python | 📅 2026-02-12 - The Classical Language Toolkit.
* [NobodyWho](https://github.com/nobodywho-ooo/nobodywho) ⭐ 706 | 🐛 37 | 🌐 Rust | 📅 2026-02-20 - The simplest way to run an LLM locally. Supports tool calling and grammar constrained sampling.
* [BigARTM](https://github.com/bigartm/bigartm) ⭐ 672 | 🐛 138 | 🌐 C++ | 📅 2026-02-05 - topic modelling platform.
* [stanford-corenlp-python](https://github.com/dasmith/stanford-corenlp-python) ⭐ 610 | 🐛 47 | 🌐 Python | 📅 2018-03-14 - Python wrapper for [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) ⭐ 10,046 | 🐛 182 | 🌐 Java | 📅 2026-02-10 **\[Deprecated]**
* [DL Translate](https://github.com/xhlulu/dl-translate) ⭐ 496 | 🐛 5 | 🌐 Python | 📅 2024-09-02 - A deep learning-based translation library between 50 languages, built with `transformers`.
* [PyNLPl](https://github.com/proycon/pynlpl) ⭐ 477 | 🐛 3 | 🌐 Python | 📅 2023-09-14 - Python Natural Language Processing Library. General purpose NLP library for Python. Also contains some specific modules for parsing common NLP formats, most notably for [FoLiA](https://proycon.github.io/folia/), but also ARPA language models, Moses phrasetables, GIZA++ alignments.
* [PySS3](https://github.com/sergioburdisso/pyss3) ⭐ 349 | 🐛 6 | 🌐 Python | 📅 2025-10-16 - Python package that implements a novel white-box machine learning model for text classification, called SS3. Since SS3 has the ability to visually explain its rationale, this package also comes with easy-to-use interactive visualizations tools ([online demos](http://tworld.io/ss3/)).
* [VeritasGraph](https://github.com/bibinprathap/VeritasGraph) ⭐ 241 | 🐛 0 | 📅 2026-02-03 - Enterprise-Grade Graph RAG for Secure, On-Premise AI with Verifiable Attribution.
* [genius](https://github.com/duanhongyi/genius) ⭐ 234 | 🐛 0 | 🌐 Python | 📅 2018-12-19 - A Chinese segment based on Conditional Random Field.
* [Rosetta](https://github.com/columbia-applied-data-science/rosetta) ⭐ 207 | 🐛 15 | 🌐 Jupyter Notebook | 📅 2022-11-09 - Text processing tools and wrappers (e.g. Vowpal Wabbit)
* [spammy](https://github.com/tasdikrahman/spammy) ⭐ 144 | 🐛 3 | 🌐 Python | 📅 2019-10-03 - A library for email Spam filtering built on top of NLTK
* [YAlign](https://github.com/machinalis/yalign) ⭐ 130 | 🐛 11 | 🌐 Python | 📅 2016-05-19 - A sentence aligner, a friendly tool for extracting parallel sentences from comparable corpora. **\[Deprecated]**
* [colibri-core](https://github.com/proycon/colibri-core) ⭐ 129 | 🐛 9 | 🌐 C++ | 📅 2026-02-05 - Python binding to C++ library for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [nut](https://github.com/pprett/nut) ⭐ 119 | 🐛 1 | 🌐 C | 📅 2014-05-07 - Natural language Understanding Toolkit. **\[Deprecated]**
* [Distance](https://github.com/doukremt/distance) ⭐ 117 | 🐛 10 | 🌐 C | 📅 2019-10-31 - Levenshtein and Hamming distance computation. **\[Deprecated]**
* [loso](https://github.com/fangpenlin/loso) ⭐ 81 | 🐛 0 | 🌐 Python | 📅 2011-04-15 - Another Chinese segmentation library. **\[Deprecated]**
* [PyStanfordDependencies](https://github.com/dmcc/PyStanfordDependencies) ⭐ 69 | 🐛 8 | 🌐 Python | 📅 2019-03-15 - Python interface for converting Penn Treebank trees to Stanford Dependencies.
* [Neofuzz](https://github.com/x-tabdeveloping/neofuzz) ⭐ 51 | 🐛 4 | 🌐 Python | 📅 2025-04-19 - Blazing fast, lightweight and customizable fuzzy and semantic text search in Python with fuzzywuzzy/thefuzz compatible API.
* [python-zpar](https://github.com/EducationalTestingService/python-zpar) ⚠️ Archived - Python bindings for [ZPar](https://github.com/frcchang/zpar) ⭐ 135 | 🐛 11 | 🌐 C++ | 📅 2016-07-15, a statistical part-of-speech-tagger, constituency parser, and dependency parser for English.
* [python-frog](https://github.com/proycon/python-frog) ⭐ 49 | 🐛 6 | 🌐 Cython | 📅 2026-02-02 - Python binding to Frog, an NLP suite for Dutch. (pos tagging, lemmatisation, dependency parsing, NER)
* [python-ucto](https://github.com/proycon/python-ucto) ⭐ 31 | 🐛 5 | 🌐 Cython | 📅 2026-02-02 - Python binding to ucto (a unicode-aware rule-based tokenizer for various languages).
* [NALP](https://github.com/gugarosa/nalp) ⭐ 23 | 🐛 2 | 🌐 Python | 📅 2026-01-01 - A Natural Adversarial Language Processing framework built over Tensorflow.
* [yase](https://github.com/PPACI/yase) ⚠️ Archived - Transcode sentence (or other sequence) to list of word vector.
* [TextCL](https://github.com/alinapetukhova/textcl) ⭐ 11 | 🐛 0 | 🌐 Python | 📅 2024-08-09 - Text preprocessing package for use in NLP tasks.
* [NLTK](https://www.nltk.org/) - A leading platform for building Python programs to work with human language data.
* [TextBlob](http://textblob.readthedocs.io/en/dev/) - Providing a consistent API for diving into common natural language processing (NLP) tasks. Stands on the giant shoulders of NLTK and Pattern, and plays nicely with both.
* [KoNLPy](http://konlpy.org) - A Python package for Korean natural language processing.
* [BLLIP Parser](https://pypi.org/project/bllipparser/) - Python bindings for the BLLIP Natural Language Parser (also known as the Charniak-Johnson parser). **\[Deprecated]**
* [editdistance](https://pypi.org/project/editdistance/) - fast implementation of edit distance.

<a name="python-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [TensorFlow](https://github.com/tensorflow/tensorflow/) ⭐ 193,874 | 🐛 3,531 | 🌐 C++ | 📅 2026-02-21 - Open source software library for numerical computation using data flow graphs.
* [PyTorch](https://github.com/pytorch/pytorch) ⭐ 97,629 | 🐛 17,952 | 🌐 Python | 📅 2026-02-21 - Tensors and Dynamic neural networks in Python with strong GPU acceleration
* [keras](https://github.com/keras-team/keras) ⭐ 63,863 | 🐛 282 | 🌐 Python | 📅 2026-02-21 - High-level neural networks frontend for [TensorFlow](https://github.com/tensorflow/tensorflow) ⭐ 193,874 | 🐛 3,531 | 🌐 C++ | 📅 2026-02-21, [CNTK](https://github.com/Microsoft/CNTK) ⚠️ Archived and [Theano](https://github.com/Theano/Theano) ⭐ 9,986 | 🐛 698 | 🌐 Python | 📅 2024-01-15.
* [Streamlit](https://github.com/streamlit/streamlit) ⭐ 43,566 | 🐛 1,310 | 🌐 Python | 📅 2026-02-20: Streamlit is an framework to create beautiful data apps in hours, not weeks.
* [Gradio](https://github.com/gradio-app/gradio) ⭐ 41,770 | 🐛 481 | 🌐 Python | 📅 2026-02-20 - A Python library for quickly creating and sharing demos of models. Debug models interactively in your browser, get feedback from collaborators, and generate public links without deploying anything.
* [Colossal-AI](https://github.com/hpcaitech/ColossalAI) ⭐ 41,348 | 🐛 493 | 🌐 Python | 📅 2026-01-19: An open-source deep learning system for large-scale model training and inference with high efficiency and low cost.
* [MindsDB](https://github.com/mindsdb/mindsdb) ⭐ 38,544 | 🐛 177 | 🌐 Python | 📅 2026-02-21 - Open Source framework to streamline use of neural networks.
* [JAX](https://github.com/google/jax) ⭐ 34,913 | 🐛 2,202 | 🌐 Python | 📅 2026-02-21 - JAX is Autograd and XLA, brought together for high-performance machine learning research.
* [Caffe](https://github.com/BVLC/caffe) ⭐ 34,839 | 🐛 1,175 | 🌐 C++ | 📅 2024-07-31 - A deep learning framework developed with cleanliness, readability, and speed in mind.
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) ⭐ 30,855 | 🐛 953 | 🌐 Python | 📅 2026-02-16 - The lightweight PyTorch wrapper for high-performance AI research.
* [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) ⭐ 30,855 | 🐛 71 | 🌐 Python | 📅 2023-10-15 - Implementations of Machine Learning models from scratch in Python with a focus on transparency. Aims to showcase the nuts and bolts of ML in an accessible way.
* [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) ⭐ 28,481 | 🐛 204 | 🌐 Jupyter Notebook | 📅 2024-06-25 - Book/iPython notebooks on Probabilistic Programming in Python.
* [XGBoost](https://github.com/dmlc/xgboost) ⭐ 28,029 | 🐛 477 | 🌐 C++ | 📅 2026-02-20 - Python bindings for eXtreme Gradient Boosting (Tree) Library.
* [Fastai](https://github.com/fastai/fastai) ⭐ 27,862 | 🐛 265 | 🌐 Jupyter Notebook | 📅 2026-02-14 - High-level wrapper built on the top of Pytorch which supports vision, text, tabular data and collaborative filtering.
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) ⭐ 23,485 | 🐛 1,247 | 🌐 Python | 📅 2026-02-19 -> Graph Neural Network Library for PyTorch.
* [Microsoft Recommenders](https://github.com/Microsoft/Recommenders) ⭐ 21,455 | 🐛 170 | 🌐 Python | 📅 2026-02-15: Examples and best practices for building recommendation systems, provided as Jupyter notebooks. The repo contains some of the latest state of the art algorithms from Microsoft Research as well as from other companies and institutions.
* [MXNet](https://github.com/apache/incubator-mxnet) ⚠️ Archived - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [Opik](https://github.com/comet-ml/opik) ⭐ 17,792 | 🐛 171 | 🌐 Python | 📅 2026-02-21: Evaluate, trace, test, and ship LLM applications across your dev and production lifecycles.
* [CNTK](https://github.com/Microsoft/CNTK) ⚠️ Archived - Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit. Documentation can be found [here](https://docs.microsoft.com/cognitive-toolkit/).
* [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) ⭐ 17,454 | 🐛 7 | 🌐 Python | 📅 2024-06-02 - Code samples for my book "Neural Networks and Deep Learning" \[DEEP LEARNING].
* [gensim](https://github.com/RaRe-Technologies/gensim) ⭐ 16,358 | 🐛 434 | 🌐 Python | 📅 2025-11-01 - Topic Modelling for Humans.
* [numpy-ML](https://github.com/ddbourgin/numpy-ml) ⭐ 16,274 | 🐛 40 | 🌐 Python | 📅 2023-10-29: Reference implementations of ML models written in numpy
* [Annoy](https://github.com/spotify/annoy) ⭐ 14,157 | 🐛 75 | 🌐 C++ | 📅 2025-10-29 - Approximate nearest neighbours implementation.
* [Optuna](https://github.com/optuna/optuna) ⭐ 13,548 | 🐛 35 | 🌐 Python | 📅 2026-02-18: Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.
* [cleanlab](https://github.com/cleanlab/cleanlab) ⭐ 11,328 | 🐛 100 | 🌐 Python | 📅 2026-01-13: The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels.
* [Turi Create](https://github.com/apple/turicreate) ⚠️ Archived - Machine learning from Apple. Turi Create simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
* [TPOT](https://github.com/EpistasisLab/tpot) ⭐ 10,047 | 🐛 304 | 🌐 Jupyter Notebook | 📅 2025-09-11 - Tool that automatically creates and optimizes machine learning pipelines using genetic programming. Consider it your personal data science assistant, automating a tedious part of machine learning.
* [AutoGluon](https://github.com/awslabs/autogluon) ⭐ 10,007 | 🐛 382 | 🌐 Python | 📅 2026-02-20: AutoML for Image, Text, Tabular, Time-Series, and MultiModal Data.
* [Theano](https://github.com/Theano/Theano/) ⭐ 9,986 | 🐛 698 | 🌐 Python | 📅 2024-01-15 - Optimizing GPU-meta-programming code generating array oriented optimizing math compiler in Python.
* [PySyft](https://github.com/OpenMined/PySyft) ⭐ 9,855 | 🐛 65 | 🌐 Python | 📅 2025-07-15 - A Python library for secure and private Deep Learning built on PyTorch and TensorFlow.
* [PyOD](https://github.com/yzhao062/pyod) ⭐ 9,719 | 🐛 240 | 🌐 Python | 📅 2026-01-05 -> Python Outlier Detection, comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. Featured for Advanced models, including Neural Networks/Deep Learning and Outlier Ensembles.
* [TFLearn](https://github.com/tflearn/tflearn) ⭐ 9,606 | 🐛 579 | 🌐 Python | 📅 2024-05-06 - Deep learning library featuring a higher-level API for TensorFlow.
* [sktime](https://github.com/alan-turing-institute/sktime) ⭐ 9,520 | 🐛 1,800 | 🌐 Python | 📅 2026-02-20 - A unified framework for machine learning with time series
* [einops](https://github.com/arogozhnikov/einops) ⭐ 9,398 | 🐛 33 | 🌐 Python | 📅 2026-02-20 - Deep learning operations reinvented (for pytorch, tensorflow, jax and others).
* [Hub](https://github.com/activeloopai/Hub) ⭐ 9,008 | 🐛 62 | 🌐 C++ | 📅 2026-02-16 - Fastest unstructured dataset management for TensorFlow/PyTorch. Stream & version-control data. Store even petabyte-scale data in a single numpy-like array on the cloud accessible on any machine. Visit [activeloop.ai](https://activeloop.ai) for more info.
* [pattern](https://github.com/clips/pattern) ⭐ 8,853 | 🐛 176 | 🌐 Python | 📅 2024-06-10 - Web mining module for Python.
* [CatBoost](https://github.com/catboost/catboost) ⭐ 8,810 | 🐛 679 | 🌐 C++ | 📅 2026-02-21 - General purpose gradient boosting on decision trees library with categorical features support out of the box. It is easy to install, well documented and supports CPU and GPU (even multi-GPU) computation.
* [BentoML](https://github.com/bentoml/bentoml) ⭐ 8,458 | 🐛 139 | 🌐 Python | 📅 2026-02-11: Toolkit for package and deploy machine learning models for serving in production
* [Cortex](https://github.com/cortexlabs/cortex) ⭐ 8,031 | 🐛 131 | 🌐 Go | 📅 2024-06-12 - Open source platform for deploying machine learning models in production.
* [Evidently](https://github.com/evidentlyai/evidently) ⭐ 7,139 | 🐛 250 | 🌐 Jupyter Notebook | 📅 2026-02-20: Interactive reports to analyze machine learning models during validation or production monitoring.
* [InterpretML](https://github.com/interpretml/interpret) ⭐ 6,798 | 🐛 111 | 🌐 C++ | 📅 2026-02-20 - InterpretML implements the Explainable Boosting Machine (EBM), a modern, fully interpretable machine learning model based on Generalized Additive Models (GAMs). This open-source package also provides visualization tools for EBMs, other glass-box models, and black-box explanations.
* [NuPIC](https://github.com/numenta/nupic) ⭐ 6,359 | 🐛 464 | 🌐 Python | 📅 2024-12-03 - Numenta Platform for Intelligent Computing.
* [deap](https://github.com/deap/deap) ⭐ 6,336 | 🐛 285 | 🌐 Python | 📅 2025-11-16 - Evolutionary algorithm framework.
* [skorch](https://github.com/skorch-dev/skorch) ⭐ 6,151 | 🐛 70 | 🌐 Jupyter Notebook | 📅 2026-02-16 - A scikit-learn compatible neural network library that wraps PyTorch.
* [Aim](https://github.com/aimhubio/aim) ⭐ 6,001 | 🐛 451 | 🌐 Python | 📅 2026-02-20 -> An easy-to-use & supercharged open-source AI metadata tracker.
* [Chainer](https://github.com/chainer/chainer) ⭐ 5,923 | 🐛 12 | 🌐 Python | 📅 2023-08-28 - Flexible neural network framework.
* [River](https://github.com/online-ml/river) ⭐ 5,720 | 🐛 117 | 🌐 Python | 📅 2026-02-09: A framework for general purpose online machine learning.
* [Microsoft ML for Apache Spark](https://github.com/Azure/mmlspark) ⭐ 5,199 | 🐛 378 | 🌐 Scala | 📅 2026-02-14 -> A distributed machine learning framework Apache Spark
* [mlxtend](https://github.com/rasbt/mlxtend) ⭐ 5,114 | 🐛 156 | 🌐 Python | 📅 2026-01-24 - A library consisting of useful tools for data science and machine learning tasks.
* [DIGITS](https://github.com/NVIDIA/DIGITS) ⚠️ Archived - The Deep Learning GPU Training System (DIGITS) is a web application for training deep learning models.
* [Deepchecks](https://github.com/deepchecks/deepchecks) ⭐ 3,979 | 🐛 258 | 🌐 Python | 📅 2025-12-28: Validation & testing of machine learning models and data during model development, deployment, and production. This includes checks and suites related to various types of issues, such as model performance, data integrity, distribution mismatches, and more.
* [neon](https://github.com/NervanaSystems/neon) ⚠️ Archived - Nervana's [high-performance](https://github.com/soumith/convnet-benchmarks) ⭐ 2,689 | 🐛 34 | 🌐 Python | 📅 2017-06-09 Python-based Deep Learning framework \[DEEP LEARNING]. **\[Deprecated]**
* [Lasagne](https://github.com/Lasagne/Lasagne) ⭐ 3,866 | 🐛 139 | 🌐 Python | 📅 2022-03-26 - Lightweight library to build and train neural networks in Theano.
* [pomegranate](https://github.com/jmschrei/pomegranate) ⭐ 3,506 | 🐛 41 | 🌐 Python | 📅 2025-03-06 - Hidden Markov Models for Python, implemented in Cython for speed and efficiency.
* [Catalyst](https://github.com/catalyst-team/catalyst) ⭐ 3,371 | 🐛 3 | 🌐 Python | 📅 2025-06-27 - High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. Being able to research/develop something new, rather than write another regular train loop.
* [mljar-supervised](https://github.com/mljar/mljar-supervised) ⭐ 3,243 | 🐛 146 | 🌐 Python | 📅 2025-07-07 - An Automated Machine Learning (AutoML) python package for tabular data. It can handle: Binary Classification, MultiClass Classification and Regression. It provides explanations and markdown reports.
* [Determined](https://github.com/determined-ai/determined) ⭐ 3,213 | 🐛 106 | 🌐 Go | 📅 2025-03-20 - Scalable deep learning training platform, including integrated support for distributed training, hyperparameter tuning, experiment tracking, and model management.
* [PyBroker](https://github.com/edtechre/pybroker) ⭐ 3,199 | 🐛 4 | 🌐 Python | 📅 2026-02-03 - Algorithmic Trading with Machine Learning.
* [pgmpy](https://github.com/pgmpy/pgmpy) ⭐ 3,158 | 🐛 443 | 🌐 Python | 📅 2026-02-20 A python library for working with Probabilistic Graphical Models.
* [Shapash](https://github.com/MAIF/shapash) ⭐ 3,148 | 🐛 48 | 🌐 Jupyter Notebook | 📅 2026-02-06 : Shapash is a Python library that provides several types of visualization that display explicit labels that everyone can understand.
* [igel](https://github.com/nidhaloff/igel) ⭐ 3,138 | 🐛 19 | 🌐 Python | 📅 2025-12-07 -> A delightful machine learning tool that allows you to train/fit, test and use models **without writing code**
* [xLearn](https://github.com/aksnzhy/xlearn) ⭐ 3,097 | 🐛 194 | 🌐 C++ | 📅 2023-08-28 - A high performance, easy-to-use, and scalable machine learning package, which can be used to solve large-scale machine learning problems. xLearn is especially useful for solving machine learning problems on large-scale sparse data, which is very common in Internet services such as online advertisement and recommender systems.
* [Shogun](https://github.com/shogun-toolbox/shogun) ⭐ 3,068 | 🐛 423 | 🌐 C++ | 📅 2023-12-19 - The Shogun Machine Learning Toolbox.
* [StellarGraph](https://github.com/stellargraph/stellargraph) ⭐ 3,044 | 🐛 326 | 🌐 Python | 📅 2024-04-10: Machine Learning on Graphs, a Python library for machine learning on graph-structured (network-structured) data.
* [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) ⭐ 2,945 | 🐛 29 | 🌐 Python | 📅 2025-09-18 -> A temporal extension of PyTorch Geometric for dynamic graph representation learning.
* [PyBrain](https://github.com/pybrain/pybrain) ⭐ 2,866 | 🐛 156 | 🌐 Python | 📅 2024-06-27 - Another Python Machine Learning Library.
* [Pylearn2](https://github.com/lisa-lab/pylearn2) ⭐ 2,776 | 🐛 201 | 🌐 Python | 📅 2021-08-20 - A Machine Learning library based on [Theano](https://github.com/Theano/Theano) ⭐ 9,986 | 🐛 698 | 🌐 Python | 📅 2024-01-15. **\[Deprecated]**
* [modAL](https://github.com/modAL-python/modAL) ⭐ 2,338 | 🐛 108 | 🌐 Python | 📅 2024-02-26 - A modular active learning framework for Python, built on top of scikit-learn.
* [Karate Club](https://github.com/benedekrozemberczki/karateclub) ⭐ 2,276 | 🐛 12 | 🌐 Python | 📅 2024-07-17 -> An unsupervised machine learning extension library for NetworkX with a Scikit-Learn like API.
* [Feature-engine](https://github.com/feature-engine/feature_engine) ⭐ 2,198 | 🐛 70 | 🌐 Python | 📅 2026-02-11 - Open source library with an exhaustive battery of feature engineering and selection methods based on pandas and scikit-learn.
* [PyTorch Lightning Bolts](https://github.com/PyTorchLightning/pytorch-lightning-bolts) ⚠️ Archived - Toolbox of models, callbacks, and datasets for AI/ML researchers.
* [thinking bayes](https://github.com/AllenDowney/ThinkBayes) ⭐ 1,684 | 🐛 12 | 🌐 TeX | 📅 2021-03-12 - Book on Bayesian Analysis.
* [auto\_ml](https://github.com/ClimbsRocks/auto_ml) ⭐ 1,658 | 🐛 187 | 🌐 Python | 📅 2021-02-10 - Automated machine learning for production and analytics. Lets you focus on the fun parts of ML, while outputting production-ready code, and detailed analytics of your dataset and results. Includes support for NLP, XGBoost, CatBoost, LightGBM, and soon, deep learning.
* [skrub](https://github.com/skrub-data/skrub) ⭐ 1,567 | 🐛 148 | 🌐 Python | 📅 2026-02-20 - Skrub is a Python library that eases preprocessing and feature engineering for machine learning on dataframes.
* [Spearmint](https://github.com/HIPS/Spearmint) ⭐ 1,564 | 🐛 77 | 🌐 Python | 📅 2019-12-27 - Spearmint is a package to perform Bayesian optimization according to the algorithms outlined in the paper: Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle and Ryan P. Adams. Advances in Neural Information Processing Systems, 2012. **\[Deprecated]**
* [python-recsys](https://github.com/ocelma/python-recsys) ⭐ 1,481 | 🐛 9 | 🌐 Python | 📅 2020-12-29 - A Python library for implementing a Recommender System.
* [skforecast](https://github.com/skforecast/skforecast) ⭐ 1,444 | 🐛 20 | 🌐 Python | 📅 2026-02-19 - Python library for time series forecasting using machine learning models. It works with any regressor compatible with the scikit-learn API, including popular options like LightGBM, XGBoost, CatBoost, Keras, and many others.
* [metric-learn](https://github.com/metric-learn/metric-learn) ⭐ 1,430 | 🐛 52 | 🌐 Python | 📅 2024-08-03 - A Python module for metric learning.
* [pydeep](https://github.com/andersbll/deeppy) ⭐ 1,379 | 🐛 22 | 🌐 Python | 📅 2020-12-28 - Deep Learning In Python. **\[Deprecated]**
* [nilearn](https://github.com/nilearn/nilearn) ⭐ 1,364 | 🐛 296 | 🌐 Python | 📅 2026-02-20 - Machine learning for NeuroImaging in Python.
* [MCP Memory Service](https://github.com/doobidoo/mcp-memory-service) ⭐ 1,353 | 🐛 5 | 🌐 Python | 📅 2026-02-20 - Universal memory service with semantic search, autonomous consolidation, and multi-client support for AI applications.
* [Intel(R) Extension for Scikit-learn](https://github.com/intel/scikit-learn-intelex) ⭐ 1,331 | 🐛 81 | 🌐 Python | 📅 2026-02-20 - A seamless way to speed up your Scikit-learn applications with no accuracy loss and code changes.
* [Brainstorm](https://github.com/IDSIA/brainstorm) ⭐ 1,304 | 🐛 27 | 🌐 Python | 📅 2022-09-13 - Fast, flexible and fun neural networks. This is the successor of PyBrain.
* [ML/DL project template](https://github.com/PyTorchLightning/deep-learning-project-template) ⚠️ Archived
* [Xcessiv](https://github.com/reiinakano/xcessiv) ⭐ 1,267 | 🐛 22 | 🌐 Python | 📅 2018-06-06 - A web-based application for quick, scalable, and automated hyperparameter tuning and stacked ensembling.
* [dtaidistance](https://github.com/wannesm/dtaidistance) ⭐ 1,215 | 🐛 21 | 🌐 Python | 📅 2026-02-12 - High performance library for time series distances (DTW) and time series clustering.
* [Crab](https://github.com/muricoca/crab) ⭐ 1,175 | 🐛 46 | 🌐 Python | 📅 2020-12-30 - A flexible, fast recommender engine. **\[Deprecated]**
* [hebel](https://github.com/hannes-brt/hebel) ⭐ 1,170 | 🐛 6 | 🌐 Python | 📅 2020-12-29 - GPU-Accelerated Deep Learning Library in Python. **\[Deprecated]**
* [Cornac](https://github.com/PreferredAI/cornac) ⭐ 1,018 | 🐛 33 | 🌐 Python | 📅 2025-12-18 - A comparative framework for multimodal recommender systems with a focus on models leveraging auxiliary data.
* [SimpleAI](https://github.com/simpleai-team/simpleai) ⭐ 987 | 🐛 13 | 🌐 Python | 📅 2023-11-06 Python implementation of many of the artificial intelligence algorithms described in the book "Artificial Intelligence, a Modern Approach". It focuses on providing an easy to use, well documented and tested library.
* [OpenMetricLearning](https://github.com/OML-Team/open-metric-learning) ⭐ 984 | 🐛 33 | 🌐 Python | 📅 2025-11-26 - A PyTorch-based framework to train and validate the models producing high-quality embeddings.
* [Restricted Boltzmann Machines](https://github.com/echen/restricted-boltzmann-machines) ⭐ 970 | 🐛 6 | 🌐 Python | 📅 2020-04-01 -Restricted Boltzmann Machines in Python. \[DEEP LEARNING]
* [Couler](https://github.com/couler-proj/couler) ⭐ 944 | 🐛 21 | 🌐 Python | 📅 2024-10-08 - Unified interface for constructing and managing machine learning workflows on different workflow engines, such as Argo Workflows, Tekton Pipelines, and Apache Airflow.
* [mlens](https://github.com/flennerhag/mlens) ⭐ 863 | 🐛 27 | 🌐 Python | 📅 2023-11-13 - A high performance, memory efficient, maximally parallelized ensemble learning, integrated with scikit-learn.
* [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow) ⭐ 792 | 🐛 73 | 🌐 Python | 📅 2023-11-02 - A machine learning framework for multi-output/multi-label and stream data.
* [ChemicalX](https://github.com/AstraZeneca/chemicalx) ⭐ 770 | 🐛 10 | 🌐 Python | 📅 2023-09-11 -> A PyTorch based deep learning library for drug pair scoring
* [PyTorch Frame](https://github.com/pyg-team/pytorch-frame) ⭐ 766 | 🐛 25 | 🌐 Python | 📅 2026-02-15 -> A Modular Framework for Multi-Modal Tabular Learning.
* [Little Ball of Fur](https://github.com/benedekrozemberczki/littleballoffur) ⭐ 713 | 🐛 7 | 🌐 Python | 📅 2025-12-20 -> A graph sampling extension library for NetworkX with a Scikit-Learn like API.
* [FEDOT](https://github.com/nccr-itmo/FEDOT) ⭐ 703 | 🐛 78 | 🌐 Python | 📅 2026-02-20: An AutoML framework for the automated design of composite modelling pipelines. It can handle classification, regression, and time series forecasting tasks on different types of data (including multi-modal datasets).
* [REP](https://github.com/yandex/rep) ⭐ 700 | 🐛 31 | 🌐 Jupyter Notebook | 📅 2024-07-31 - an IPython-based environment for conducting data-driven research in a consistent and reproducible way. REP is not trying to substitute scikit-learn, but extends it and provides better user experience. **\[Deprecated]**
* [Opytimizer](https://github.com/gugarosa/opytimizer) ⭐ 631 | 🐛 0 | 🌐 Python | 📅 2026-02-16 - Python-based meta-heuristic optimization techniques.
* [AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics](https://github.com/Western-OC2-Lab/AutoML-Implementation-for-Static-and-Dynamic-Data-Analytics) ⭐ 629 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2024-05-14: A tutorial to help machine learning researchers to automatically obtain optimized machine learning models with the optimal learning performance on any specific task.
* [PyGrid](https://github.com/OpenMined/PyGrid/) ⚠️ Archived - Peer-to-peer network of data owners and data scientists who can collectively train AI models using PySyft
* [Neuraxle](https://github.com/Neuraxio/Neuraxle) ⭐ 614 | 🐛 3 | 🌐 Python | 📅 2026-02-20: A framework providing the right abstractions to ease research, development, and deployment of your ML pipelines.
* [milk](https://github.com/luispedro/milk) ⚠️ Archived - Machine learning toolkit focused on supervised classification. **\[Deprecated]**
* [pyhsmm](https://github.com/mattjj/pyhsmm) ⭐ 572 | 🐛 46 | 🌐 Python | 📅 2025-01-25 - library for approximate unsupervised inference in Bayesian Hidden Markov Models (HMMs) and explicit-duration Hidden semi-Markov Models (HSMMs), focusing on the Bayesian Nonparametric extensions, the HDP-HMM and HDP-HSMM, mostly with weak-limit approximations.
* [SKLL](https://github.com/EducationalTestingService/skll) ⭐ 561 | 🐛 20 | 🌐 Python | 📅 2025-06-12 - A wrapper around scikit-learn that makes it simpler to conduct experiments.
* [Auto\_ViML](https://github.com/AutoViML/Auto_ViML) ⭐ 546 | 🐛 0 | 🌐 Python | 📅 2025-01-30 -> Automatically Build Variant Interpretable ML models fast! Auto\_ViML is pronounced "auto vimal", is a comprehensive and scalable Python AutoML toolkit with imbalanced handling, ensembling, stacking and built-in feature selection. Featured in <a href="https://towardsdatascience.com/why-automl-is-an-essential-new-tool-for-data-scientists-2d9ab4e25e46?source=friends_link&sk=d03a0cc55c23deb497d546d6b9be0653">Medium article</a>.
* [skbayes](https://github.com/AmazaspShumik/sklearn-bayes) ⭐ 522 | 🐛 20 | 🌐 Jupyter Notebook | 📅 2021-09-22 - Python package for Bayesian Machine Learning with scikit-learn API.
* [Lightwood](https://github.com/mindsdb/lightwood) ⭐ 502 | 🐛 16 | 🌐 Python | 📅 2026-02-11 - A Pytorch based framework that breaks down machine learning problems into smaller blocks that can be glued together seamlessly with objective to build predictive models with one line of code.
* [ChefBoost](https://github.com/serengil/chefboost) ⭐ 486 | 🐛 0 | 🌐 Python | 📅 2025-07-09 - a lightweight decision tree framework for Python with categorical feature support covering regular decision tree algorithms such as ID3, C4.5, CART, CHAID and regression tree; also some advanced bagging and boosting techniques such as gradient boosting, random forest and adaboost.
* [imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble) ⭐ 414 | 🐛 1 | 🌐 Python | 📅 2025-04-29 - Python toolbox for quick implementation, modification, evaluation, and visualization of ensemble learning algorithms for class-imbalanced data. Supports out-of-the-box multi-class imbalanced (long-tailed) classification.
* [Featureforge](https://github.com/machinalis/featureforge) ⭐ 385 | 🐛 11 | 🌐 Python | 📅 2017-12-26 A set of tools for creating and testing machine learning features, with a scikit-learn compatible API.
* [rgf\_python](https://github.com/RGF-team/rgf) ⭐ 383 | 🐛 9 | 🌐 C++ | 📅 2022-01-08 - Python bindings for Regularized Greedy Forest (Tree) Library.
* [Sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt) ⭐ 355 | 🐛 4 | 🌐 Python | 📅 2025-09-13: An AutoML package for hyperparameters tuning using evolutionary algorithms, with built-in callbacks, plotting, remote logging and more.
* [Upgini](https://github.com/upgini/upgini) ⭐ 348 | 🐛 4 | 🌐 Python | 📅 2026-02-07: Free automated data & feature enrichment library for machine learning - automatically searches through thousands of ready-to-use features from public and community shared data sources and enriches your training dataset with only the accuracy improving features.
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) ⭐ 324 | 🐛 31 | 🌐 Scala | 📅 2020-10-29 - A service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [Pyevolve](https://github.com/perone/Pyevolve) ⭐ 315 | 🐛 46 | 🌐 Python | 📅 2021-08-28 - Genetic algorithm framework. **\[Deprecated]**
* [Parris](https://github.com/jgreenemi/Parris) ⭐ 314 | 🐛 7 | 🌐 Python | 📅 2026-02-17 - Parris, the automated infrastructure setup tool for machine learning algorithms.
* [fuku-ml](https://github.com/fukuball/fuku-ml) ⭐ 283 | 🐛 14 | 🌐 Python | 📅 2025-08-07 - Simple machine learning library, including Perceptron, Regression, Support Vector Machine, Decision Tree and more, it's easy to use and easy to learn for beginners.
* [RexMex](https://github.com/AstraZeneca/rexmex) ⭐ 277 | 🐛 4 | 🌐 Python | 📅 2023-08-22 -> A general purpose recommender metrics library for fair evaluation.
* [evostra](https://github.com/alirezamika/evostra) ⭐ 272 | 🐛 8 | 🌐 Python | 📅 2020-04-27 - A fast Evolution Strategy implementation in Python.
* [machine learning](https://github.com/jeff1evesque/machine-learning) ⭐ 261 | 🐛 78 | 🌐 JavaScript | 📅 2021-02-17 - automated build consisting of a [web-interface](https://github.com/jeff1evesque/machine-learning#web-interface) ⭐ 261 | 🐛 78 | 🌐 JavaScript | 📅 2021-02-17, and set of [programmatic-interface](https://github.com/jeff1evesque/machine-learning#programmatic-interface) ⭐ 261 | 🐛 78 | 🌐 JavaScript | 📅 2021-02-17 API, for support vector machines. Corresponding dataset(s) are stored into a SQL database, then generated model(s) used for prediction(s), are stored into a NoSQL datastore.
* [Frouros](https://github.com/IFCA/frouros) ⭐ 252 | 🐛 11 | 🌐 Python | 📅 2026-02-11: Frouros is an open source Python library for drift detection in machine learning systems.
* [Backprop](https://github.com/backprop-ai/backprop) ⭐ 241 | 🐛 5 | 🌐 Python | 📅 2021-05-03 - Backprop makes it simple to use, finetune, and deploy state-of-the-art ML models.
* [Shapley](https://github.com/benedekrozemberczki/shapley) ⭐ 224 | 🐛 1 | 🌐 Python | 📅 2026-01-01 -> A data-driven framework to quantify the value of classifiers in a machine learning ensemble.
* [Eurybia](https://github.com/MAIF/eurybia) ⭐ 216 | 🐛 13 | 🌐 Jupyter Notebook | 📅 2025-11-10: Eurybia monitors data and model drift over time and securizes model deployment with data validation.
* [CometML](https://github.com/comet-ml/comet-examples) ⭐ 171 | 🐛 27 | 🌐 Jupyter Notebook | 📅 2026-02-09: The best-in-class MLOps platform with experiment tracking, model production monitoring, a model registry, and data lineage from training straight through to production.
* [neurolab](https://github.com/zueve/neurolab) ⭐ 167 | 🐛 16 | 🌐 Python | 📅 2020-06-02
* [topicwizard](https://github.com/x-tabdeveloping/topic-wizard) ⭐ 144 | 🐛 3 | 🌐 Python | 📅 2025-03-19 - Interactive topic model visualization/interpretation framework.
* [Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/williamFalcon/pix2pix-keras) ⭐ 143 | 🐛 4 | 🌐 Python | 📅 2017-03-27 - Implementation of image to image (pix2pix) translation from the paper by [isola et al](https://arxiv.org/pdf/1611.07004.pdf).\[DEEP LEARNING]
* [steppy](https://github.com/neptune-ml/steppy) ⚠️ Archived -> Lightweight, Python library for fast and reproducible machine learning experimentation. Introduces a very simple interface that enables clean machine learning pipeline design.
* [stacked\_generalization](https://github.com/fukatani/stacked_generalization) ⭐ 119 | 🐛 3 | 🌐 Python | 📅 2019-05-02 - Implementation of machine learning stacking technique as a handy library in Python.
* [Neurolink](https://github.com/juspay/neurolink) ⭐ 106 | 🐛 424 | 🌐 TypeScript | 📅 2026-02-20 - Enterprise-grade LLM integration framework for building production-ready AI applications with built-in hallucination prevention, RAG, and MCP support.
* [neuropredict](https://github.com/raamana/neuropredict) ⭐ 104 | 🐛 29 | 🌐 Python | 📅 2026-01-19 - Aimed at novice machine learners and non-expert programmers, this package offers easy (no coding needed) and comprehensive machine learning (evaluation and full report of predictive performance WITHOUT requiring you to code) in Python for NeuroImaging and any other type of features. This is aimed at absorbing much of the ML workflow, unlike other packages like nilearn and pymvpa, which require you to learn their API and code to produce anything useful.
* [Pebl](https://github.com/abhik/pebl/) ⭐ 104 | 🐛 4 | 🌐 Python | 📅 2011-11-16 - Python Environment for Bayesian Learning. **\[Deprecated]**
* [breze](https://github.com/breze-no-salt/breze) ⭐ 95 | 🐛 29 | 🌐 Jupyter Notebook | 📅 2016-09-02 - Theano based library for deep and recurrent neural networks.
* [bayeso](https://github.com/jungtaekkim/bayeso) ⭐ 95 | 🐛 1 | 🌐 Python | 📅 2026-01-12 - A simple, but essential Bayesian optimization package, written in Python.
* [topik](https://github.com/ContinuumIO/topik) ⚠️ Archived - Topic modelling toolkit. **\[Deprecated]**
* [Bolt](https://github.com/pprett/bolt) ⭐ 87 | 🐛 0 | 🌐 C | 📅 2011-10-05 - Bolt Online Learning Toolbox. **\[Deprecated]**
* [Cogitare](https://github.com/cogitare-ai/cogitare) ⚠️ Archived: A Modern, Fast, and Modular Deep Learning and Machine Learning framework for Python.
* [Synthia](https://github.com/dmey/synthia) ⭐ 65 | 🐛 2 | 🌐 Python | 📅 2023-09-28 - Multidimensional synthetic data generation in Python.
* [ByteHub](https://github.com/bytehub-ai/bytehub) ⭐ 61 | 🐛 0 | 🌐 Python | 📅 2021-05-16 - An easy-to-use, Python-based feature store. Optimized for time-series data.
* [xRBM](https://github.com/omimo/xRBM) ⭐ 55 | 🐛 4 | 🌐 Python | 📅 2017-11-19 - A library for Restricted Boltzmann Machine (RBM) and its conditional variants in Tensorflow.
* [neonrvm](https://github.com/siavashserver/neonrvm) ⚠️ Archived - neonrvm is an open source machine learning library based on RVM technique. It's written in C programming language and comes with Python programming language bindings.
* [OPFython](https://github.com/gugarosa/opfython) ⭐ 37 | 🐛 0 | 🌐 Python | 📅 2026-02-15 - A Python-inspired implementation of the Optimum-Path Forest classifier.
* [CoverTree](https://github.com/patvarilly/CoverTree) ⭐ 33 | 🐛 2 | 🌐 Python | 📅 2012-03-13 - Python implementation of cover trees, near-drop-in replacement for scipy.spatial.kdtree **\[Deprecated]**
* [ML Model building](https://github.com/Shanky-21/Machine_learning) ⭐ 31 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2021-03-30 -> A Repository Containing Classification, Clustering, Regression, Recommender Notebooks with illustration to make them.
* [MiraiML](https://github.com/arthurpaulino/miraiml) ⭐ 26 | 🐛 3 | 🌐 Python | 📅 2019-10-25: An asynchronous engine for continuous & autonomous machine learning, built for real-time usage.
* [SKBEL](https://github.com/robinthibaut/skbel) ⭐ 26 | 🐛 1 | 🌐 Python | 📅 2024-07-09: A Python library for Bayesian Evidential Learning (BEL) in order to estimate the uncertainty of a prediction.
* [steppy-toolkit](https://github.com/neptune-ml/steppy-toolkit) ⚠️ Archived -> Curated collection of the neural networks, transformers and models that make your machine learning work faster and more effective.
* [python-timbl](https://github.com/proycon/python-timbl) ⭐ 18 | 🐛 0 | 🌐 Python | 📅 2025-05-02 - A Python extension module wrapping the full TiMBL C++ programming interface. Timbl is an elaborate k-Nearest Neighbours machine learning toolkit.
* [Gower Express](https://github.com/momonga-ml/gower-express.git) ⭐ 11 | 🐛 0 | 🌐 Python | 📅 2025-09-05 - The Fastest Gower Distance Implementation for Python. GPU-accelerated similarity matching for mixed data types, 15-25% faster than alternatives with production-ready reliability.
* [pyclugen](https://github.com/clugen/pyclugen) ⭐ 10 | 🐛 0 | 🌐 Python | 📅 2025-08-12 - Multidimensional cluster generation in Python.
* [Thampi](https://github.com/scoremedia/thampi) ⚠️ Archived - Machine Learning Prediction System on AWS Lambda
* [Okrolearn](https://github.com/Okerew/okrolearn) ⚠️ Archived: A python machine learning library created to combine powefull data analasys features with tensors and machine learning components, while maintaining support for other libraries.
* [mlforgex](https://github.com/dhgefergfefruiwefhjhcduc/ML_Forgex) ⭐ 0 | 🐛 0 | 🌐 Python | 📅 2026-01-14 - Lightweight ML utility for automated training, evaluation, and prediction with CLI and Python API support.
* [ray3.run](https://ray3.run) - AI-powered tools and applications for developers and businesses to enhance productivity and workflow automation. \* [XAD](https://pypi.org/project/xad/) -> Fast and easy-to-use backpropagation tool.
* * [TopFreePrompts by LucyBrain](https://topfreeprompts.com) -> 10,000+ professional AI prompts across 23 categories with systematic training for automating ML workflows and analysis.
* [Apache SINGA](https://singa.apache.org) - An Apache Incubating project for developing an open source machine learning library.
* [MLlib in Apache Spark](http://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [Towhee](https://towhee.io) - A Python module that encode unstructured data into embeddings.
* [scikit-learn](https://scikit-learn.org/) - A Python module for machine learning built on top of SciPy.
* [astroML](https://www.astroml.org/) - Machine Learning and Data Mining for Astronomy.
* [graphlab-create](https://turi.com/products/create/docs/) - A library with various machine learning models (regression, clustering, recommender systems, graph analytics, etc.) implemented on top of a disk-backed DataFrame.
* [BigML](https://bigml.com) - A library that contacts external servers.
* [prophet](https://facebook.github.io/prophet/) - Fast and automated time series forecasting framework by Facebook.
* [tweetopic](https://centre-for-humanities-computing.github.io/tweetopic/) - Blazing fast short-text-topic-modelling for Python.
* [Surprise](https://surpriselib.com) - A scikit for building and analyzing recommender systems.
* [implicit](https://implicit.readthedocs.io/en/latest/quickstart.html) - Fast Python Collaborative Filtering for Implicit Datasets.
* [LightFM](https://making.lyst.com/lightfm/docs/home.html) -  A Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback.
* [imbalanced-learn](https://imbalanced-learn.org/stable/) - Python module to perform under sampling and oversampling with various techniques.
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search.
* [Orange](https://orange.biolab.si/) - Open source data visualization and data analysis for novices and experts.
* [Edward](http://edwardlib.org/) - A library for probabilistic modelling, inference, and criticism. Built on top of TensorFlow.
* [NannyML](https://bit.ly/nannyml-github-machinelearning): Python library capable of fully capturing the impact of data drift on performance. Allows estimation of post-deployment model performance without access to targets.

<a name="python-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [Superset](https://github.com/apache/incubator-superset) ⭐ 70,607 | 🐛 1,149 | 🌐 TypeScript | 📅 2026-02-21 - A data exploration platform designed to be visual, intuitive, and interactive.
* [Dash](https://github.com/plotly/dash) ⭐ 24,505 | 🐛 560 | 🌐 Python | 📅 2026-02-20 - A framework for creating analytical web applications built on top of Plotly.js, React, and Flask
* [bokeh](https://github.com/bokeh/bokeh) ⭐ 20,357 | 🐛 865 | 🌐 TypeScript | 📅 2026-02-20 - Interactive Web Plotting for Python.
* [zipline](https://github.com/quantopian/zipline) ⭐ 19,434 | 🐛 367 | 🌐 Python | 📅 2024-02-13 - A Pythonic algorithmic trading library.
* [SymPy](https://github.com/sympy/sympy) ⭐ 14,395 | 🐛 5,563 | 🌐 Python | 📅 2026-02-20 - A Python library for symbolic mathematics.
* [lime](https://github.com/marcotcr/lime) ⭐ 12,101 | 🐛 131 | 🌐 JavaScript | 📅 2024-07-25 - Lime is about explaining what machine learning classifiers (or models) are doing. It is able to explain any black box classifier, with two or more classes.
* [statsmodels](https://github.com/statsmodels/statsmodels) ⭐ 11,261 | 🐛 2,969 | 🌐 Python | 📅 2026-01-13 - Statistical modelling and econometrics in Python.
* [altair](https://github.com/altair-viz/altair) ⭐ 10,266 | 🐛 155 | 🌐 Python | 📅 2026-02-19 - A Python to Vega translator.
* [PyMC](https://github.com/pymc-devs/pymc) ⭐ 9,487 | 🐛 466 | 🌐 Python | 📅 2026-02-16 - Markov Chain Monte Carlo sampling toolkit.
* [Vaex](https://github.com/vaexio/vaex) ⭐ 8,471 | 🐛 548 | 🌐 Python | 📅 2026-02-05 - A high performance Python library for lazy Out-of-Core DataFrames (similar to Pandas), to visualize and explore big tabular datasets. Documentation can be found [here](https://vaex.io/docs/index.html).
* [PyQtGraph](https://github.com/pyqtgraph/pyqtgraph) ⭐ 4,298 | 🐛 500 | 🌐 Python | 📅 2026-02-13 - A pure-python graphics and GUI library built on PyQt4 / PySide and NumPy.
* [ggplot](https://github.com/yhat/ggpy) ⭐ 3,698 | 🐛 178 | 🌐 Python | 📅 2023-01-21 - Same API as ggplot2 for R. **\[Deprecated]**
* [bqplot](https://github.com/bloomberg/bqplot) ⭐ 3,683 | 🐛 278 | 🌐 TypeScript | 📅 2026-01-23 - An API for plotting in Jupyter (IPython).
* [vispy](https://github.com/vispy/vispy) ⭐ 3,546 | 🐛 376 | 🌐 Python | 📅 2026-01-14 - GPU-based high-performance interactive OpenGL 2D/3D data visualization library.
* [TensorWatch](https://github.com/microsoft/tensorwatch) ⭐ 3,463 | 🐛 53 | 🌐 Jupyter Notebook | 📅 2025-09-27 - Debugging and visualization tool for machine learning and data science. It extensively leverages Jupyter Notebook to show real-time visualizations of data in running processes such as machine learning training.
* [Blaze](https://github.com/blaze/blaze) ⭐ 3,197 | 🐛 267 | 🌐 Python | 📅 2023-09-29 - NumPy and Pandas interface to Big Data.
* [Mars](https://github.com/mars-project/mars) ⭐ 2,749 | 🐛 215 | 🌐 Python | 📅 2024-01-02 - A tensor-based framework for large-scale data computation which is often regarded as a parallel and distributed version of NumPy.
* [scikit-plot](https://github.com/reiinakano/scikit-plot) ⭐ 2,431 | 🐛 31 | 🌐 Python | 📅 2024-08-20 - A visualization library for quick and easy generation of common plots in data analysis and machine learning.
* [AutoViz](https://github.com/AutoViML/AutoViz) ⭐ 1,882 | 🐛 2 | 🌐 Python | 📅 2024-06-10 AutoViz performs automatic visualization of any dataset with a single line of Python code. Give it any input file (CSV, txt or JSON) of any size and AutoViz will visualize it. See <a href="https://towardsdatascience.com/autoviz-a-new-tool-for-automated-visualization-ec9c1744a6ad?source=friends_link&sk=c9e9503ec424b191c6096d7e3f515d10">Medium article</a>.
* [emcee](https://github.com/dfm/emcee) ⭐ 1,567 | 🐛 67 | 🌐 Python | 📅 2026-01-19 - The Python ensemble sampling toolkit for affine-invariant MCMC.
* [PyCM](https://github.com/sepandhaghighi/pycm) ⭐ 1,500 | 🐛 14 | 🌐 Python | 📅 2026-02-20 - PyCM is a multi-class confusion matrix library written in Python that supports both input data vectors and direct matrix, and a proper tool for post-classification model evaluation that supports most classes and overall statistics parameters
* [d3py](https://github.com/mikedewar/d3py) ⭐ 1,419 | 🐛 48 | 🌐 Python | 📅 2020-12-28 - A plotting library for Python, based on [D3.js](https://d3js.org/).
* [Open Mining](https://github.com/mining/mining) ⚠️ Archived - Business Intelligence (BI) in Python (Pandas web interface) **\[Deprecated]**
* [Kartograph.py](https://github.com/kartograph/kartograph.py) ⭐ 1,000 | 🐛 62 | 🌐 Python | 📅 2020-12-30 - Rendering beautiful SVG maps in Python.
* [Bowtie](https://github.com/jwkvam/bowtie) ⭐ 769 | 🐛 51 | 🌐 Python | 📅 2019-09-09 - A dashboard library for interactive visualizations using flask socketio and react.
* [Dora](https://github.com/nathanepstein/dora) ⭐ 647 | 🐛 0 | 🌐 Python | 📅 2025-08-05 - Tools for exploratory data analysis in Python.
* [DataComPy](https://github.com/capitalone/datacompy) ⭐ 635 | 🐛 11 | 🌐 Python | 📅 2026-02-20 - A library to compare Pandas, Polars, and Spark data frames. It provides stats and lets users adjust for match accuracy.
* [SOMPY](https://github.com/sevamoo/SOMPY) ⭐ 552 | 🐛 48 | 🌐 Jupyter Notebook | 📅 2023-04-07 - Self Organizing Map written in Python (Uses neural networks for data analysis).
* [ggfortify](https://github.com/sinhrks/ggfortify) ⭐ 539 | 🐛 22 | 🌐 R | 📅 2025-10-19 - Unified interface to ggplot2 popular R packages.
* [pastalog](https://github.com/rewonc/pastalog) ⭐ 421 | 🐛 11 | 🌐 JavaScript | 📅 2017-03-28 - Simple, realtime visualization of neural network training performance.
* [SparklingPandas](https://github.com/sparklingpandas/sparklingpandas) ⭐ 363 | 🐛 53 | 🌐 Python | 📅 2023-07-06 Pandas on PySpark (POPS).
* [ParaMonte](https://github.com/cdslaborg/paramonte) ⭐ 300 | 🐛 20 | 🌐 Fortran | 📅 2025-12-18 - A general-purpose Python library for Bayesian data analysis and visualization via serial/parallel Monte Carlo and MCMC simulations. Documentation can be found [here](https://www.cdslab.org/paramonte/).
* [Flama](https://github.com/vortico/flama) ⭐ 288 | 🐛 6 | 🌐 Python | 📅 2026-02-13 - Ignite your models into blazing-fast machine learning APIs with a modern framework.
* [somoclu](https://github.com/peterwittek/somoclu) ⭐ 277 | 🐛 37 | 🌐 C | 📅 2025-12-20 Massively parallel self-organizing maps: accelerate training on multicore CPUs, GPUs, and clusters, has python API.
* [Petrel](https://github.com/AirSage/Petrel) ⭐ 246 | 🐛 12 | 🌐 Python | 📅 2022-12-14 - Tools for writing, submitting, debugging, and monitoring Storm topologies in pure Python.
* [pycascading](https://github.com/twitter/pycascading) ⚠️ Archived **\[Deprecated]**
* [visualize\_ML](https://github.com/ayush1997/visualize_ML) ⭐ 204 | 🐛 0 | 🌐 Python | 📅 2016-09-28 - A python package for data exploration and data analysis. **\[Deprecated]**
* [ipychart](https://github.com/nicohlr/ipychart) ⭐ 131 | 🐛 0 | 🌐 Python | 📅 2024-08-24 - The power of Chart.js in Jupyter Notebook.
* [windML](https://github.com/cigroup-ol/windml) ⭐ 129 | 🐛 13 | 📅 2024-01-28 - A Python Framework for Wind Energy Analysis and Prediction.
* [HDBScan](https://github.com/lmcinnes/hdbscan) ⭐ 102 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2017-11-09 - implementation of the hdbscan algorithm in Python - used for clustering
* [NuPIC Studio](https://github.com/htm-community/nupic.studio) ⭐ 96 | 🐛 16 | 🌐 Python | 📅 2020-07-26 An all-in-one NuPIC Hierarchical Temporal Memory visualization and debugging super-tool! **\[Deprecated]**
* [DataVisualization](https://github.com/Shanky-21/Data_visualization) ⭐ 48 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2022-10-07 - A GitHub Repository Where you can Learn Datavisualizatoin Basics to Intermediate level.
* [dowel](https://github.com/rlworkgroup/dowel) ⭐ 35 | 🐛 20 | 🌐 Python | 📅 2023-08-30 - A little logger for machine learning research. Output any object to the terminal, CSV, TensorBoard, text logs on disk, and more with just one call to `logger.log()`.
* [PyDexter](https://github.com/D3xterjs/pydexter) ⭐ 30 | 🐛 0 | 🌐 Python | 📅 2018-05-19 - Simple plotting for Python. Wrapper for D3xterjs; easily render charts in-browser.
* [Lambdo](https://github.com/asavinov/lambdo) ⭐ 25 | 🐛 0 | 🌐 Python | 📅 2021-01-01 - A workflow engine for solving machine learning problems by combining in one analysis pipeline (i) feature engineering and machine learning (ii) model training and prediction (iii) table population and column evaluation via user-defined (Python) functions.
* [Cartopy](https://scitools.org.uk/cartopy/docs/latest/) - Cartopy is a Python package designed for geospatial data processing in order to produce maps and other geospatial data analyses.
* [SciPy](https://www.scipy.org/) - A Python-based ecosystem of open-source software for mathematics, science, and engineering.
* [NumPy](https://www.numpy.org/) - A fundamental package for scientific computing with Python.
* [Numba](https://numba.pydata.org/) - Python JIT (just in time) compiler to LLVM aimed at scientific Python by the developers of Cython and NumPy.
* [NetworkX](https://networkx.github.io/) - A high-productivity software for complex networks.
* [igraph](https://igraph.org/python/) - binding to igraph library - General purpose graph library.
* [Pandas](https://pandas.pydata.org/) - A library providing high-performance, easy-to-use data structures and data analysis tools.
* [PyDy](https://www.pydy.org/) - Short for Python Dynamics, used to assist with workflow in the modelling of dynamic motion based around NumPy, SciPy, IPython, and matplotlib.
* [astropy](https://www.astropy.org/) - A community Python library for Astronomy.
* [matplotlib](https://matplotlib.org/) - A Python 2D plotting library.
* [plotly](https://plot.ly/python/) - Collaborative web plotting for Python and matplotlib.
* [pygal](http://pygal.org/en/stable/) - A Python SVG Charts Creator.
* [cerebro2](https://github.com/numenta/nupic.cerebro2) A web-based visualization and debugging platform for NuPIC. **\[Deprecated]**
* [Seaborn](https://seaborn.pydata.org/) - A python visualization library based on matplotlib.
* [Ruffus](http://www.ruffus.org.uk) - Computation Pipeline library for python.

<a name="python-misc-scripts--ipython-notebooks--codebases"></a>

#### Misc Scripts / iPython Notebooks / Codebases

* [data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks) ⭐ 28,880 | 🐛 42 | 🌐 Python | 📅 2024-03-20 - Continually updated Data Science Python Notebooks: Spark, Hadoop MapReduce, HDFS, AWS, Kaggle, scikit-learn, matplotlib, pandas, NumPy, SciPy, and various command lines.
* [handsonml](https://github.com/ageron/handson-ml) ⭐ 25,873 | 🐛 145 | 🌐 Jupyter Notebook | 📅 2023-10-03 - Fundamentals of machine learning in python.
* [Pydata book](https://github.com/wesm/pydata-book) ⭐ 24,315 | 🐛 26 | 🌐 Jupyter Notebook | 📅 2025-10-17 - Materials and IPython notebooks for "Python for Data Analysis" by Wes McKinney, published by O'Reilly Media
* [Homemade Machine Learning](https://github.com/trekhleb/homemade-machine-learning) ⭐ 24,263 | 🐛 28 | 🌐 Jupyter Notebook | 📅 2025-11-23 - Python examples of popular machine learning algorithms with interactive Jupyter demos and math being explained
* [A gallery of interesting IPython notebooks](https://github.com/jupyter/jupyter/wiki/A-gallery-of-interesting-Jupyter-Notebooks) ⭐ 15,285 | 🐛 43 | 🌐 Python | 📅 2025-12-17
* [Dive into Machine Learning  with Python Jupyter notebook and scikit-learn](https://github.com/hangtwenty/dive-into-machine-learning) ⚠️ Archived - "I learned Python by hacking first, and getting serious *later.* I wanted to do this with Machine Learning. If this is your style, join me in getting a bit ahead of yourself."
* [Introduction to Machine Learning with Python](https://github.com/amueller/introduction_to_ml_with_python) ⭐ 8,026 | 🐛 28 | 🌐 Jupyter Notebook | 📅 2024-03-14 - Notebooks and code for the book "Introduction to Machine Learning with Python"
* [pandas cookbook](https://github.com/jvns/pandas-cookbook) ⭐ 7,029 | 🐛 34 | 🌐 Jupyter Notebook | 📅 2024-10-24 - Recipes for using Python's pandas library.
* [numpic](https://github.com/numenta/nupic) ⭐ 6,359 | 🐛 464 | 🌐 Python | 📅 2024-12-03
* [pattern\_classification](https://github.com/rasbt/pattern_classification) ⭐ 4,213 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2023-11-26
* [Introduction to machine learning with scikit-learn](https://github.com/justmarkham/scikit-learn-videos) ⭐ 3,779 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2024-03-05 - IPython notebooks from Data School's video tutorials on scikit-learn.
* [Keras Tuner](https://github.com/keras-team/keras-tuner) ⭐ 2,918 | 🐛 235 | 🌐 Python | 📅 2025-12-01 - An easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search.
* [Allen Downey’s Think Bayes Code](https://github.com/AllenDowney/ThinkBayes) ⭐ 1,684 | 🐛 12 | 🌐 TeX | 📅 2021-03-12 - Code repository for Think Bayes.
* [hyperopt](https://github.com/hyperopt/hyperopt-sklearn) ⭐ 1,643 | 🐛 78 | 🌐 Python | 📅 2025-04-15
* [TDB](https://github.com/ericjang/tdb) ⭐ 1,350 | 🐛 8 | 🌐 JavaScript | 📅 2017-01-27 - TensorDebugger (TDB) is a visual debugger for deep learning. It features interactive, node-by-node debugging and visualization for TensorFlow.
* [Hyperparameter-Optimization-of-Machine-Learning-Algorithms](https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms) ⭐ 1,320 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2022-09-22 - Code for hyperparameter tuning/optimization of machine learning and deep learning algorithms.
* [Suiron](https://github.com/kendricktan/suiron/) ⭐ 708 | 🐛 3 | 🌐 Python | 📅 2016-10-08 - Machine Learning for RC Cars.
* [ipython-notebooks](https://github.com/ogrisel/notebooks) ⭐ 574 | 🐛 3 | 🌐 Jupyter Notebook | 📅 2025-12-11
* [Allen Downey’s Think OS Code](https://github.com/AllenDowney/ThinkOS) ⭐ 568 | 🐛 7 | 🌐 TeX | 📅 2020-04-28 - Text and supporting code for Think OS: A Brief Introduction to Operating Systems.
* [the-elements-of-statistical-learning](https://github.com/maitbayev/the-elements-of-statistical-learning) ⭐ 426 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2026-02-10 - This repository contains Jupyter notebooks implementing the algorithms found in the book and summary of the textbook.
* [climin](https://github.com/BRML/climin) ⭐ 183 | 🐛 22 | 🌐 Python | 📅 2026-01-16 - Optimization library focused on machine learning, pythonic implementations of gradient descent, LBFGS, rmsprop, adadelta and others.
* [sentiment\_classifier](https://github.com/kevincobain2000/sentiment_classifier) ⭐ 170 | 🐛 0 | 🌐 OpenEdge ABL | 📅 2022-04-05 - Sentiment classifier using word sense disambiguation.
* [jProcessing](https://github.com/kevincobain2000/jProcessing) ⭐ 148 | 🐛 4 | 🌐 OpenEdge ABL | 📅 2020-09-09 - Kanji / Hiragana / Katakana to Romaji Converter. Edict Dictionary & parallel sentences Search. Sentence Similarity between two JP Sentences. Sentiment Analysis of Japanese Text. Run Cabocha(ISO--8859-1 configured) in Python.
* [scikit-learn tutorials](https://github.com/GaelVaroquaux/scikit-learn-tutorial) ⭐ 133 | 🐛 0 | 🌐 Python | 📅 2011-12-06 - Series of notebooks for learning scikit-learn.
* [Allen Downey’s Think Complexity Code](https://github.com/AllenDowney/ThinkComplexity) ⭐ 118 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2024-10-01 - Code for Allen Downey's book Think Complexity.
* [BayesPy](https://github.com/maxsklar/BayesPy) ⭐ 109 | 🐛 1 | 🌐 HTML | 📅 2023-06-05 - Bayesian Inference Tools in Python.
* [MiniGrad](https://github.com/kennysong/minigrad) ⭐ 96 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2021-08-24 – A minimal, educational, Pythonic implementation of autograd (\~100 loc).
* [Neon Course](https://github.com/NervanaSystems/neon_course) ⚠️ Archived - IPython notebooks for a complete course around understanding Nervana's Neon.
* [Crab](https://github.com/marcelcaraciolo/crab) ⭐ 88 | 🐛 9 | 🌐 Python | 📅 2011-09-02 - A recommendation engine library for Python.
* [GreatCircle](https://github.com/mwgg/GreatCircle) ⭐ 78 | 🐛 0 | 🌐 PHP | 📅 2022-03-20 - Library for calculating great circle distance.
* [Map/Reduce implementations of common ML algorithms](https://github.com/Yannael/BigDataAnalytics_INFOH515) ⭐ 62 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2022-03-01: Jupyter notebooks that cover how to implement from scratch different ML algorithms (ordinary least squares, gradient descent, k-means, alternating least squares), using Python NumPy, and how to then make these implementations scalable using Map/Reduce and Spark.
* [Prodmodel](https://github.com/prodmodel/prodmodel) ⭐ 58 | 🐛 7 | 🌐 Python | 📅 2022-06-21 - Build tool for data science pipelines.
* [sentiment-analyzer](https://github.com/madhusudancs/sentiment-analyzer) ⭐ 52 | 🐛 0 | 🌐 JavaScript | 📅 2012-03-30 - Tweets Sentiment Analyzer
* [BioPy](https://github.com/jaredthecoder/BioPy) ⭐ 49 | 🐛 1 | 🌐 Python | 📅 2023-06-05 - Biologically-Inspired and Machine Learning Algorithms in Python. **\[Deprecated]**
* [Allen Downey’s Data Science Course](https://github.com/AllenDowney/DataScience) ⭐ 45 | 🐛 0 | 🌐 HTML | 📅 2022-05-03 - Code for Data Science at Olin College, Spring 2014.
* [CAEs for Data Assimilation](https://github.com/julianmack/Data_Assimilation) ⭐ 44 | 🐛 0 | 🌐 Python | 📅 2021-01-07 - Convolutional autoencoders for 3D image/field compression applied to reduced order [Data Assimilation](https://en.wikipedia.org/wiki/Data_assimilation).
* [group-lasso](https://github.com/fabianp/group_lasso) ⭐ 39 | 🐛 0 | 🌐 Python | 📅 2012-10-24 - Some experiments with the coordinate descent algorithm used in the (Sparse) Group Lasso model.
* [decision-weights](https://github.com/CamDavidsonPilon/decision-weights) ⭐ 33 | 🐛 1 | 🌐 Python | 📅 2015-03-02
* [mne-python-notebooks](https://github.com/mne-tools/mne-python-notebooks) ⭐ 29 | 🐛 0 | 📅 2016-04-10 - IPython notebooks for EEG/MEG data processing using mne-python.
* [Sarah Palin LDA](https://github.com/Wavelets/sarah-palin-lda) ⭐ 9 | 🐛 0 | 🌐 Scala | 📅 2011-07-09 - Topic Modelling the Sarah Palin emails.
* [thinking stats 2](https://github.com/Wavelets/ThinkStats2) ⭐ 8 | 🐛 0 | 📅 2014-07-14
* [2012-paper-diginorm](https://github.com/dib-lab/2012-paper-diginorm) ⭐ 8 | 🐛 6 | 🌐 Python | 📅 2015-05-21
* [Diffusion Segmentation](https://github.com/Wavelets/diffusion-segmentation) ⭐ 2 | 🐛 0 | 🌐 Python | 📅 2010-08-26 - A collection of image segmentation algorithms based on diffusion methods.
* [Scipy Tutorials](https://github.com/Wavelets/scipy-tutorials) ⭐ 2 | 🐛 0 | 🌐 Python | 📅 2010-05-19 - SciPy tutorials. This is outdated, check out scipy-lecture-notes.
* [minidiff](https://github.com/ahoynodnarb/minidiff) ⭐ 1 | 🐛 0 | 🌐 Python | 📅 2026-02-09 - A slightly larger, somewhat feature-complete, PyTorch-inspired, NumPy implementation of a tensor reverse-mode automatic differentiation engine.
* [Heart\_Disease-Prediction](https://github.com/ShivamChoudhary17/Heart_Disease) ⭐ 1 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2021-10-03 - Given clinical parameters about a patient, can we predict whether or not they have heart disease?
* [Flight Fare Prediction](https://github.com/ShivamChoudhary17/Flight_Fare_Prediction) ⭐ 1 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2021-10-03 - This basically to gauge the understanding of Machine Learning Workflow and Regression technique in specific.
* [SVM Explorer](https://github.com/plotly/dash-svm) - Interactive SVM Explorer, using Dash and scikit-learn
* [Python Programming for the Humanities](https://www.karsdorp.io/python-course/) - Course for Python programming for the Humanities, assuming no prior knowledge. Heavy focus on text processing / NLP.
* [Optunity examples](http://optunity.readthedocs.io/en/latest/notebooks/index.html) - Examples demonstrating how to use Optunity in synergy with machine learning libraries.
* [Practical XGBoost in Python](https://parrotprediction.teachable.com/p/practical-xgboost-in-python) - comprehensive online course about using XGBoost in Python.

<a name="python-neural-networks"></a>

#### Neural Networks

* [NeuralTalk](https://github.com/karpathy/neuraltalk2) ⭐ 5,572 | 🐛 142 | 🌐 Jupyter Notebook | 📅 2017-11-07 - NeuralTalk is a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences. **\[Deprecated]**
* [NeuralTalk](https://github.com/karpathy/neuraltalk) ⭐ 5,480 | 🐛 30 | 🌐 Python | 📅 2020-12-22 - NeuralTalk is a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences.
* [TResNet: High Performance GPU-Dedicated Architecture](https://github.com/mrT23/TResNet) ⭐ 478 | 🐛 5 | 🌐 Python | 📅 2024-12-10 - TResNet models were designed and optimized to give the best speed-accuracy tradeoff out there on GPUs.
* [sequitur](https://github.com/shobrook/sequitur) ⭐ 452 | 🐛 8 | 🌐 Python | 📅 2024-02-21 PyTorch library for creating and training sequence autoencoders in just two lines of code
* [TResNet: Simple and powerful neural network library for python](https://github.com/zueve/neurolab) ⭐ 167 | 🐛 16 | 🌐 Python | 📅 2020-06-02 - Variety of supported types of Artificial Neural Network and learning algorithms.
* [nn\_builder](https://github.com/p-christ/nn_builder) ⭐ 165 | 🐛 6 | 🌐 Python | 📅 2023-08-23 - nn\_builder is a python package that lets you build neural networks in 1 line
* [Neuron](https://github.com/molcik/python-neuron) ⭐ 41 | 🐛 0 | 🌐 Python | 📅 2026-01-06 - Neuron is simple class for time series predictions. It's utilize LNU (Linear Neural Unit), QNU (Quadratic Neural Unit), RBF (Radial Basis Function), MLP (Multi Layer Perceptron), MLP-ELM (Multi Layer Perceptron - Extreme Learning Machine) neural networks learned with Gradient descent or LeLevenberg–Marquardt algorithm. **\[Deprecated]**
* [Kinho](https://github.com/kinhosz/Neural) ⭐ 37 | 🐛 5 | 🌐 Python | 📅 2025-08-06 - Simple API for Neural Network. Better for image processing with CPU/GPU + Transfer Learning.
* [Data Driven Code](https://github.com/atmb4u/data-driven-code) ⭐ 30 | 🐛 0 | 🌐 Python | 📅 2016-11-21 - Very simple implementation of neural networks for dummies in python without using any libraries, with detailed comments.
* [ANEE](https://github.com/abkmystery/ANEE) ⭐ 1 | 🐛 0 | 🌐 Python | 📅 2025-11-30 - Adaptive Neural Execution Engine for transformers. Per-token sparse inference with dynamic layer skipping, profiler-based gating, and KV-cache-safe compute reduction.
* [Machine Learning, Data Science and Deep Learning with Python](https://www.manning.com/livevideo/machine-learning-data-science-and-deep-learning-with-python) - LiveVideo course that covers machine learning, Tensorflow, artificial intelligence, and neural networks.
* [Jina AI](https://jina.ai/) An easier way to build neural search in the cloud. Compatible with Jupyter Notebooks.

<a name="python-spiking-neural-networks"></a>

#### Spiking Neural Networks

* [Tonic](https://github.com/neuromorphs/tonic) ⭐ 271 | 🐛 33 | 🌐 Python | 📅 2025-10-23 - A library that makes downloading publicly available neuromorphic datasets a breeze and provides event-based data transformation/augmentation pipelines.
* [Sinabs](https://github.com/synsense/sinabs) ⭐ 110 | 🐛 27 | 🌐 Python | 📅 2026-02-05 - A deep learning library for spiking neural networks which is based on PyTorch, focuses on fast training and supports inference on neuromorphic hardware.
* [Rockpool](https://github.com/synsense/rockpool) ⭐ 77 | 🐛 2 | 🌐 Python | 📅 2026-02-10 - A machine learning library for spiking neural networks. Supports training with both torch and jax pipelines, and deployment to neuromorphic hardware.

<a name="python-survival-analysis"></a>

#### Python Survival Analysis

* [lifelines](https://github.com/CamDavidsonPilon/lifelines) ⭐ 2,550 | 🐛 278 | 🌐 Python | 📅 2026-02-04 - lifelines is a complete survival analysis library, written in pure Python
* [Scikit-Survival](https://github.com/sebp/scikit-survival) ⭐ 1,270 | 🐛 29 | 🌐 Python | 📅 2026-02-16 - scikit-survival is a Python module for survival analysis built on top of scikit-learn. It allows doing survival analysis while utilizing the power of scikit-learn, e.g., for pre-processing or doing cross-validation.

<a name="python-federated-learning"></a>

#### Federated Learning

* [Flower](https://flower.dev/) - A unified approach to federated learning, analytics, and evaluation. Federate any workload, any ML framework, and any programming language.
* [PySyft](https://github.com/OpenMined/PySyft) ⭐ 9,855 | 🐛 65 | 🌐 Python | 📅 2025-07-15 - A Python library for secure and private Deep Learning.
* [Tensorflow-Federated](https://www.tensorflow.org/federated) A federated learning framework for machine learning and other computations on decentralized data.

<a name="python-kaggle-competition-source-code"></a>

#### Kaggle Competition Source Code

* [Kaggle Galaxy Challenge](https://github.com/benanne/kaggle-galaxies) ⭐ 497 | 🐛 0 | 🌐 Python | 📅 2014-08-13 - Winning solution for the Galaxy Challenge on Kaggle.
* [open-solution-home-credit](https://github.com/neptune-ml/open-solution-home-credit) ⚠️ Archived -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Home-Credit-Default-Risk) for [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).
* [open-solution-data-science-bowl-2018](https://github.com/neptune-ml/open-solution-data-science-bowl-2018) ⚠️ Archived -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Data-Science-Bowl-2018) for [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).
* [open-solution-toxic-comments](https://github.com/neptune-ml/open-solution-toxic-comments) ⚠️ Archived -> source code for [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
* [kaggle insults](https://github.com/amueller/kaggle_insults) ⚠️ Archived - Kaggle Submission for "Detecting Insults in Social Commentary".
* [open-solution-salt-identification](https://github.com/neptune-ml/open-solution-salt-identification) ⚠️ Archived -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Salt-Detection) for [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).
* [kaggle-blackbox](https://github.com/zygmuntz/kaggle-blackbox) ⭐ 116 | 🐛 0 | 🌐 Matlab | 📅 2014-06-21 - Deep learning made easy.
* [kaggle\_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) ⭐ 66 | 🐛 0 | 🌐 Python | 📅 2014-04-17 - Code for the Kaggle acquire valued shoppers challenge.
* [kaggle\_acquire-valued-shoppers-challenge](https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge) ⭐ 66 | 🐛 0 | 🌐 Python | 📅 2014-04-17 - Code for the Kaggle acquire valued shoppers challenge.
* [open-solution-ship-detection](https://github.com/neptune-ml/open-solution-ship-detection) ⚠️ Archived -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Ships) for [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).
* [Kaggle Dogs vs. Cats](https://github.com/kastnerkyle/kaggle-dogs-vs-cats) ⭐ 65 | 🐛 1 | 🌐 Python | 📅 2020-10-05 - Code for Kaggle Dogs vs. Cats competition.
* [kaggle-advertised-salaries](https://github.com/zygmuntz/kaggle-advertised-salaries) ⭐ 55 | 🐛 0 | 🌐 Python | 📅 2014-06-21 - Predicting job salaries from ads - a Kaggle competition.
* [open-solution-googleai-object-detection](https://github.com/neptune-ml/open-solution-googleai-object-detection) ⚠️ Archived -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Google-AI-Object-Detection-Challenge) for [Google AI Open Images - Object Detection Track](https://www.kaggle.com/c/google-ai-open-images-object-detection-track).
* [kaggle-cifar](https://github.com/zygmuntz/kaggle-cifar) ⭐ 44 | 🐛 0 | 🌐 Python | 📅 2014-06-21 - Code for the CIFAR-10 competition at Kaggle, uses cuda-convnet.
* [Kaggle Stackoverflow](https://github.com/zygmuntz/kaggle-stackoverflow) ⭐ 44 | 🐛 0 | 🌐 Python | 📅 2017-11-24 - Predicting closed questions on Stack Overflow.
* [open-solution-value-prediction](https://github.com/neptune-ml/open-solution-value-prediction) ⚠️ Archived -> source code and [experiments results](https://app.neptune.ml/neptune-ml/Santander-Value-Prediction-Challenge) for [Santander Value Prediction Challenge](https://www.kaggle.com/c/santander-value-prediction-challenge).
* [wine-quality](https://github.com/zygmuntz/wine-quality) ⭐ 26 | 🐛 2 | 🌐 R | 📅 2015-09-03 - Predicting wine quality.
* [kaggle amazon](https://github.com/zygmuntz/kaggle-amazon) ⭐ 25 | 🐛 0 | 🌐 Python | 📅 2014-06-21 - Amazon access control challenge.
* [Kaggle Gender](https://github.com/zygmuntz/kaggle-gender) ⭐ 22 | 🐛 0 | 🌐 Python | 📅 2014-06-21 - A Kaggle competition: discriminate gender based on handwriting.
* [kaggle-accelerometer](https://github.com/zygmuntz/kaggle-accelerometer) ⭐ 15 | 🐛 0 | 🌐 Python | 📅 2014-06-21 - Code for Accelerometer Biometric Competition at Kaggle.
* [wiki challenge](https://github.com/hammer/wikichallenge) ⭐ 11 | 🐛 0 | 🌐 Python | 📅 2011-11-23 - An implementation of Dell Zhang's solution to Wikipedia's Participation Challenge on Kaggle.
* [Kaggle Merck](https://github.com/zygmuntz/kaggle-merck) ⭐ 10 | 🐛 0 | 🌐 Python | 📅 2014-06-21 - Merck challenge at Kaggle.
* [kaggle-bestbuy\_big](https://github.com/zygmuntz/kaggle-bestbuy_big) ⭐ 8 | 🐛 0 | 🌐 Python | 📅 2014-06-21 - Code for the Best Buy competition at Kaggle.
* [kaggle-bestbuy\_small](https://github.com/zygmuntz/kaggle-bestbuy_small) ⭐ 6 | 🐛 0 | 🌐 Python | 📅 2014-06-21

<a name="python-reinforcement-learning"></a>

#### Reinforcement Learning

* [RLlib](https://github.com/ray-project/ray) ⭐ 41,411 | 🐛 3,385 | 🌐 Python | 📅 2026-02-21 - RLlib is an industry level, highly scalable RL library for tf and torch, based on Ray. It's used by companies like Amazon and Microsoft to solve real-world decision making problems at scale.
* [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) ⭐ 11,354 | 🐛 82 | 🌐 Python | 📅 2026-02-20 - A library for developing and comparing reinforcement learning algorithms (successor of \[gym])(<https://github.com/openai/gym> ⭐ 37,054 | 🐛 127 | 🌐 Python | 📅 2024-10-11).
* [DeepMind Lab](https://github.com/deepmind/lab) ⭐ 7,328 | 🐛 65 | 🌐 C | 📅 2023-01-04 - DeepMind Lab is a 3D learning environment based on id Software's Quake III Arena via ioquake3 and other open source software. Its primary purpose is to act as a testbed for research in artificial intelligence, especially deep reinforcement learning.
* [Serpent.AI](https://github.com/SerpentAI/SerpentAI) ⚠️ Archived - Serpent.AI is a game agent framework that allows you to turn any video game you own into a sandbox to develop AI and machine learning experiments. For both researchers and hobbyists.
* [DI-engine](https://github.com/opendilab/DI-engine) ⭐ 3,593 | 🐛 25 | 🌐 Python | 📅 2025-12-07 - DI-engine is a generalized Decision Intelligence engine. It supports most basic deep reinforcement learning (DRL) algorithms, such as DQN, PPO, SAC, and domain-specific algorithms like QMIX in multi-agent RL, GAIL in inverse RL, and RND in exploration problems.
* [Retro](https://github.com/openai/retro) ⭐ 3,563 | 🐛 61 | 🌐 C | 📅 2024-02-22 - Retro Games in Gym
* [Roboschool](https://github.com/openai/roboschool) ⭐ 2,167 | 🐛 83 | 🌐 Python | 📅 2023-04-02 - Open-source software for robot simulation, integrated with OpenAI Gym.
* [garage](https://github.com/rlworkgroup/garage) ⭐ 2,081 | 🐛 234 | 🌐 Python | 📅 2023-05-04 - A toolkit for reproducible reinforcement learning research
* [ViZDoom](https://github.com/mwydmuch/ViZDoom) ⭐ 1,977 | 🐛 30 | 🌐 C++ | 📅 2026-02-20 - ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.
* [metaworld](https://github.com/rlworkgroup/metaworld) ⭐ 1,747 | 🐛 8 | 🌐 Python | 📅 2026-01-20 - An open source robotics benchmark for meta- and multi-task reinforcement learning
* [SLM Lab](https://github.com/kengz/SLM-Lab) ⭐ 1,334 | 🐛 8 | 🌐 Python | 📅 2026-02-20 - Modular Deep Reinforcement Learning framework in PyTorch.
* [Maze](https://github.com/enlite-ai/maze) ⭐ 286 | 🐛 2 | 🌐 Python | 📅 2026-02-18 - Application-oriented deep reinforcement learning framework addressing real-world decision problems.
* [Gym4ReaL](https://github.com/Daveonwave/gym4ReaL) ⭐ 48 | 🐛 1 | 🌐 Python | 📅 2025-07-03 - Gym4ReaL is a comprehensive suite of realistic environments designed to support the development and evaluation of RL algorithms that can operate in real-world scenarios. The suite includes a diverse set of tasks exposing RL algorithms to a variety of practical challenges.
* [Coach](https://github.com/NervanaSystems/coach) - Reinforcement Learning Coach by Intel® AI Lab enables easy experimentation with state of the art Reinforcement Learning algorithms
* [acme](https://deepmind.com/research/publications/Acme) - An Open Source Distributed Framework for Reinforcement Learning that makes build and train your agents easily.
* [Spinning Up](https://spinningup.openai.com) - An educational resource designed to let anyone learn to become a skilled practitioner in deep reinforcement learning

<a name="python-speech-recognition"></a>

#### Speech Recognition

* [EspNet](https://github.com/espnet/espnet) ⭐ 9,742 | 🐛 53 | 🌐 Python | 📅 2026-02-20 - ESPnet is an end-to-end speech processing toolkit for tasks like speech recognition, translation, and enhancement, using PyTorch and Kaldi-style data processing.

<a name="python-development tools"></a>

#### Development Tools

* [CodeFlash.AI](https://www.codeflash.ai/) – CodeFlash.AI – Ship Blazing-Fast Python Code, Every Time.

<a name="ruby"></a>

## Ruby

<a name="ruby-natural-language-processing"></a>

#### Natural Language Processing

* [Twitter-text-rb](https://github.com/twitter/twitter-text/tree/master/rb) ⭐ 3,126 | 🐛 93 | 🌐 HTML | 📅 2024-04-26 - A library that does auto linking and extraction of usernames, lists and hashtags in tweets.
* [Treat](https://github.com/louismullie/treat) ⭐ 1,369 | 🐛 35 | 🌐 Ruby | 📅 2025-05-16 - Text Retrieval and Annotation Toolkit, definitely the most comprehensive toolkit I’ve encountered so far for Ruby.
* [Awesome NLP with Ruby](https://github.com/arbox/nlp-with-ruby) ⭐ 1,073 | 🐛 7 | 🌐 Ruby | 📅 2023-06-27 - Curated link list for practical natural language processing in Ruby.
* [Stemmer](https://github.com/aurelian/ruby-stemmer) ⚠️ Archived - Expose libstemmer\_c to Ruby. **\[Deprecated]**
* [UEA Stemmer](https://github.com/ealdent/uea-stemmer) ⭐ 54 | 🐛 0 | 🌐 Ruby | 📅 2026-02-18 - Ruby port of UEALite Stemmer - a conservative stemmer for search and indexing.
* [Raspell](https://sourceforge.net/projects/raspell/) - raspell is an interface binding for ruby. **\[Deprecated]**

<a name="ruby-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Awesome Machine Learning with Ruby](https://github.com/arbox/machine-learning-with-ruby) ⭐ 2,213 | 🐛 6 | 🌐 Ruby | 📅 2024-12-26 - Curated list of ML related resources for Ruby.
* [rumale](https://github.com/yoshoku/rumale) ⭐ 901 | 🐛 0 | 🌐 Ruby | 📅 2026-02-19 - Rumale is a machine learning library in Ruby
* [CardMagic-Classifier](https://github.com/cardmagic/classifier) ⭐ 715 | 🐛 14 | 🌐 Ruby | 📅 2026-02-15 - A general classifier module to allow Bayesian and other types of classifications.
* [rb-libsvm](https://github.com/febeling/rb-libsvm) ⭐ 279 | 🐛 2 | 🌐 C++ | 📅 2023-12-07 - Ruby language bindings for LIBSVM which is a Library for Support Vector Machines.
* [jRuby Mahout](https://github.com/vasinov/jruby_mahout) ⭐ 165 | 🐛 3 | 🌐 Ruby | 📅 2015-09-21 - JRuby Mahout is a gem that unleashes the power of Apache Mahout in the world of JRuby. **\[Deprecated]**
* [Scoruby](https://github.com/asafschers/scoruby) ⭐ 70 | 🐛 4 | 🌐 Ruby | 📅 2022-10-19 - Creates Random Forest classifiers from PMML files.
* [Ruby Machine Learning](https://github.com/tsycho/ruby-machine-learning) ⭐ 34 | 🐛 0 | 🌐 Ruby | 📅 2012-01-10 - Some Machine Learning algorithms, implemented in Ruby. **\[Deprecated]**
* [Machine Learning Ruby](https://github.com/mizoR/machine-learning-ruby) ⭐ 16 | 🐛 0 | 🌐 Ruby | 📅 2012-09-06 **\[Deprecated]**

<a name="ruby-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [Bioruby](https://github.com/bioruby/bioruby) ⭐ 380 | 🐛 15 | 🌐 Ruby | 📅 2025-09-19
* [rsruby](https://github.com/alexgutteridge/rsruby) ⭐ 335 | 🐛 15 | 🌐 Ruby | 📅 2018-10-06 - Ruby - R bridge.
* [Arel](https://github.com/nkallen/arel) ⭐ 270 | 🐛 7 | 🌐 Ruby | 📅 2009-05-21 **\[Deprecated]**
* [Glean](https://github.com/glean/glean) ⭐ 119 | 🐛 2 | 🌐 Ruby | 📅 2016-10-31 - A data management tool for humans. **\[Deprecated]**
* [data-visualization-ruby](https://github.com/chrislo/data_visualisation_ruby) ⭐ 67 | 🐛 0 | 🌐 Ruby | 📅 2009-12-22 - Source code and supporting content for my Ruby Manor presentation on Data Visualisation with Ruby. **\[Deprecated]**
* [plot-rb](https://github.com/zuhao/plotrb) ⚠️ Archived - A plotting library in Ruby built on top of Vega and D3. **\[Deprecated]**
* [scruffy](https://github.com/delano/scruffy) ⭐ 31 | 🐛 0 | 🌐 Ruby | 📅 2017-06-15 - A beautiful graphing toolkit for Ruby.
* [ruby-plot](https://www.ruby-toolbox.com/projects/ruby-plot) - gnuplot wrapper for Ruby, especially for plotting ROC curves into SVG files. **\[Deprecated]**
* [SciRuby](http://sciruby.com/)

<a name="ruby-misc"></a>

#### Misc

* [Big Data For Chimps](https://github.com/infochimps-labs/big_data_for_chimps) ⭐ 169 | 🐛 10 | 🌐 Ruby | 📅 2015-06-15
* [Listof](https://github.com/kevincobain2000/listof) ⭐ 30 | 🐛 0 | 🌐 Ruby | 📅 2017-02-06 - Community based data collection, packed in gem. Get list of pretty much anything (stop words, countries, non words) in txt, JSON or hash. [Demo/Search for a list](http://kevincobain2000.github.io/listof/)

<a name="rust"></a>

## Rust

<a name="rust-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [candle](https://github.com/huggingface/candle) ⭐ 19,435 | 🐛 606 | 🌐 Rust | 📅 2026-02-19 - Candle is a minimalist ML framework for Rust with a focus on performance (including GPU support) and ease of use.
* [leaf](https://github.com/autumnai/leaf) ⭐ 5,549 | 🐛 33 | 🌐 Rust | 📅 2024-03-20 - open source framework for machine intelligence, sharing concepts from TensorFlow and Caffe. Available under the MIT license. [**\[Deprecated\]**](https://medium.com/@mjhirn/tensorflow-wins-89b78b29aafb#.s0a3uy4cc)
* [linfa](https://github.com/rust-ml/linfa) ⭐ 4,557 | 🐛 67 | 🌐 Rust | 📅 2026-02-04 - a comprehensive toolkit to build Machine Learning applications with Rust
* [linfa](https://github.com/rust-ml/linfa) ⭐ 4,557 | 🐛 67 | 🌐 Rust | 📅 2026-02-04 - `linfa` aims to provide a comprehensive toolkit to build Machine Learning applications with Rust
* [rusty-machine](https://github.com/AtheMathmo/rusty-machine) ⚠️ Archived - a pure-rust machine learning library.
* [smartcore](https://github.com/smartcorelib/smartcore) ⭐ 888 | 🐛 58 | 🌐 Rust | 📅 2026-02-10 - "The Most Advanced Machine Learning Library In Rust."
* [rustlearn](https://github.com/maciejkula/rustlearn) ⭐ 638 | 🐛 13 | 🌐 Rust | 📅 2021-06-07 - a machine learning framework featuring logistic regression, support vector machines, decision trees and random forests.
* [delta](https://github.com/delta-rs/delta) ⭐ 410 | 🐛 9 | 🌐 Rust | 📅 2025-06-10 - An open source machine learning framework in Rust Δ
* [RustNN](https://github.com/jackm321/RustNN) ⭐ 341 | 🐛 4 | 🌐 Rust | 📅 2017-12-21 - RustNN is a feedforward neural network library. **\[Deprecated]**
* [deeplearn-rs](https://github.com/tedsta/deeplearn-rs) ⚠️ Archived - deeplearn-rs provides simple networks that use matrix multiplication, addition, and ReLU under the MIT license.
* [RusticSOM](https://github.com/avinashshenoy97/RusticSOM) ⭐ 36 | 🐛 2 | 🌐 Rust | 📅 2022-10-15 - A Rust library for Self Organising Maps (SOM).

#### Deep Learning

* [burn](https://github.com/tracel-ai/burn) ⭐ 14,404 | 🐛 229 | 🌐 Rust | 📅 2026-02-20 - Burn is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals
* [tch-rs](https://github.com/LaurentMazare/tch-rs) ⭐ 5,287 | 🐛 227 | 🌐 Rust | 📅 2026-01-22 - Rust bindings for the C++ API of PyTorch
* [dfdx](https://github.com/coreylowman/dfdx) ⭐ 1,894 | 🐛 90 | 🌐 Rust | 📅 2024-07-23 - Deep learning in Rust, with shape checked tensors and neural networks

#### Natural Language Processing

* [huggingface/tokenizers](https://github.com/huggingface/tokenizers) ⭐ 10,482 | 🐛 144 | 🌐 Rust | 📅 2026-02-20 - Fast State-of-the-Art Tokenizers optimized for Research and Production
* [shimmy](https://github.com/Michael-A-Kuykendall/shimmy) ⭐ 3,684 | 🐛 28 | 🌐 Rust | 📅 2026-01-16 - Python-free Rust inference server for NLP models with OpenAI API compatibility and hot model swapping.
* [rust-bert](https://github.com/guillaume-be/rust-bert) ⭐ 3,039 | 🐛 73 | 🌐 Rust | 📅 2026-01-13 - Rust native ready-to-use NLP pipelines and transformer-based models (BERT, DistilBERT, GPT2,...)

<a name="r"></a>

## R

<a name="r-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [MXNet](https://github.com/apache/incubator-mxnet) ⚠️ Archived - Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Go, JavaScript and more.
* [CatBoost](https://github.com/catboost/catboost) ⭐ 8,810 | 🐛 679 | 🌐 C++ | 📅 2026-02-21 - General purpose gradient boosting on decision trees library with categorical features support out of the box for R.
* [Machine Learning For Hackers](https://github.com/johnmyleswhite/ML_for_Hackers) ⭐ 3,817 | 🐛 36 | 🌐 R | 📅 2019-05-26
* [XGBoost.R](https://github.com/tqchen/xgboost/tree/master/R-package) ⭐ 579 | 🐛 0 | 🌐 C++ | 📅 2018-07-04 - R binding for eXtreme Gradient Boosting (Tree) Library.
* [TDSP-Utilities](https://github.com/Azure/Azure-TDSP-Utilities) ⚠️ Archived - Two data science utilities in R from Microsoft: 1) Interactive Data Exploration, Analysis, and Reporting (IDEAR) ; 2) Automated Modelling and Reporting (AMR).
* [SuperLearner](https://github.com/ecpolley/SuperLearner) ⭐ 286 | 🐛 20 | 🌐 R | 📅 2025-12-15 - Multi-algorithm ensemble learning packages.
* [clugenr](https://github.com/clugen/clugenr/) ⭐ 5 | 🐛 1 | 🌐 R | 📅 2025-07-31 - Multidimensional cluster generation in R.
* [ahaz](https://cran.r-project.org/web/packages/ahaz/index.html) - ahaz: Regularization for semiparametric additive hazards regression. **\[Deprecated]**
* [arules](https://cran.r-project.org/web/packages/arules/index.html) - arules: Mining Association Rules and Frequent Itemsets
* [biglasso](https://cran.r-project.org/web/packages/biglasso/index.html) - biglasso: Extending Lasso Model Fitting to Big Data in R.
* [bmrm](https://cran.r-project.org/web/packages/bmrm/index.html) - bmrm: Bundle Methods for Regularized Risk Minimization Package.
* [Boruta](https://cran.r-project.org/web/packages/Boruta/index.html) - Boruta: A wrapper algorithm for all-relevant feature selection.
* [bst](https://cran.r-project.org/web/packages/bst/index.html) - bst: Gradient Boosting.
* [C50](https://cran.r-project.org/web/packages/C50/index.html) - C50: C5.0 Decision Trees and Rule-Based Models.
* [caret](https://topepo.github.io/caret/index.html) - Classification and Regression Training: Unified interface to \~150 ML algorithms in R.
* [caretEnsemble](https://cran.r-project.org/web/packages/caretEnsemble/index.html) - caretEnsemble: Framework for fitting multiple caret models as well as creating ensembles of such models. **\[Deprecated]**
* [Clever Algorithms For Machine Learning](https://machinelearningmastery.com/)
* [CORElearn](https://cran.r-project.org/web/packages/CORElearn/index.html) - CORElearn: Classification, regression, feature evaluation and ordinal evaluation.
  -\* [CoxBoost](https://cran.r-project.org/web/packages/CoxBoost/index.html) - CoxBoost: Cox models by likelihood based boosting for a single survival endpoint or competing risks **\[Deprecated]**
* [Cubist](https://cran.r-project.org/web/packages/Cubist/index.html) - Cubist: Rule- and Instance-Based Regression Modelling.
* [e1071](https://cran.r-project.org/web/packages/e1071/index.html) - e1071: Misc Functions of the Department of Statistics (e1071), TU Wien
* [earth](https://cran.r-project.org/web/packages/earth/index.html) - earth: Multivariate Adaptive Regression Spline Models
* [elasticnet](https://cran.r-project.org/web/packages/elasticnet/index.html) - elasticnet: Elastic-Net for Sparse Estimation and Sparse PCA.
* [ElemStatLearn](https://cran.r-project.org/web/packages/ElemStatLearn/index.html) - ElemStatLearn: Data sets, functions and examples from the book: "The Elements of Statistical Learning, Data Mining, Inference, and Prediction" by Trevor Hastie, Robert Tibshirani and Jerome Friedman Prediction" by Trevor Hastie, Robert Tibshirani and Jerome Friedman.
* [evtree](https://cran.r-project.org/web/packages/evtree/index.html) - evtree: Evolutionary Learning of Globally Optimal Trees.
* [forecast](https://cran.r-project.org/web/packages/forecast/index.html) - forecast: Timeseries forecasting using ARIMA, ETS, STLM, TBATS, and neural network models.
* [forecastHybrid](https://cran.r-project.org/web/packages/forecastHybrid/index.html) - forecastHybrid: Automatic ensemble and cross validation of ARIMA, ETS, STLM, TBATS, and neural network models from the "forecast" package.
* [fpc](https://cran.r-project.org/web/packages/fpc/index.html) - fpc: Flexible procedures for clustering.
* [frbs](https://cran.r-project.org/web/packages/frbs/index.html) - frbs: Fuzzy Rule-based Systems for Classification and Regression Tasks. **\[Deprecated]**
* [GAMBoost](https://cran.r-project.org/web/packages/GAMBoost/index.html) - GAMBoost: Generalized linear and additive models by likelihood based boosting. **\[Deprecated]**
* [gamboostLSS](https://cran.r-project.org/web/packages/gamboostLSS/index.html) - gamboostLSS: Boosting Methods for GAMLSS.
* [gbm](https://cran.r-project.org/web/packages/gbm/index.html) - gbm: Generalized Boosted Regression Models.
* [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html) - glmnet: Lasso and elastic-net regularized generalized linear models.
* [glmpath](https://cran.r-project.org/web/packages/glmpath/index.html) - glmpath: L1 Regularization Path for Generalized Linear Models and Cox Proportional Hazards Model.
* [GMMBoost](https://cran.r-project.org/web/packages/GMMBoost/index.html) - GMMBoost: Likelihood-based Boosting for Generalized mixed models. **\[Deprecated]**
* [grplasso](https://cran.r-project.org/web/packages/grplasso/index.html) - grplasso: Fitting user specified models with Group Lasso penalty.
* [grpreg](https://cran.r-project.org/web/packages/grpreg/index.html) - grpreg: Regularization paths for regression models with grouped covariates.
* [h2o](https://cran.r-project.org/web/packages/h2o/index.html) - A framework for fast, parallel, and distributed machine learning algorithms at scale -- Deeplearning, Random forests, GBM, KMeans, PCA, GLM.
* [hda](https://cran.r-project.org/web/packages/hda/index.html) - hda: Heteroscedastic Discriminant Analysis. **\[Deprecated]**
* [Introduction to Statistical Learning](https://www-bcf.usc.edu/~gareth/ISL/)
* [ipred](https://cran.r-project.org/web/packages/ipred/index.html) - ipred: Improved Predictors.
* [kernlab](https://cran.r-project.org/web/packages/kernlab/index.html) - kernlab: Kernel-based Machine Learning Lab.
* [klaR](https://cran.r-project.org/web/packages/klaR/index.html) - klaR: Classification and visualization.
* [L0Learn](https://cran.r-project.org/web/packages/L0Learn/index.html) - L0Learn: Fast algorithms for best subset selection.
* [lars](https://cran.r-project.org/web/packages/lars/index.html) - lars: Least Angle Regression, Lasso and Forward Stagewise. **\[Deprecated]**
* [lasso2](https://cran.r-project.org/web/packages/lasso2/index.html) - lasso2: L1 constrained estimation aka ‘lasso’.
* [LiblineaR](https://cran.r-project.org/web/packages/LiblineaR/index.html) - LiblineaR: Linear Predictive Models Based On The Liblinear C/C++ Library.
* [LogicReg](https://cran.r-project.org/web/packages/LogicReg/index.html) - LogicReg: Logic Regression.
* [maptree](https://cran.r-project.org/web/packages/maptree/index.html) - maptree: Mapping, pruning, and graphing tree models. **\[Deprecated]**
* [mboost](https://cran.r-project.org/web/packages/mboost/index.html) - mboost: Model-Based Boosting.
* [medley](https://www.kaggle.com/general/3661) - medley: Blending regression models, using a greedy stepwise approach.
* [mlr](https://cran.r-project.org/web/packages/mlr/index.html) - mlr: Machine Learning in R.
* [ncvreg](https://cran.r-project.org/web/packages/ncvreg/index.html) - ncvreg: Regularization paths for SCAD- and MCP-penalized regression models.
* [nnet](https://cran.r-project.org/web/packages/nnet/index.html) - nnet: Feed-forward Neural Networks and Multinomial Log-Linear Models. **\[Deprecated]**
* [pamr](https://cran.r-project.org/web/packages/pamr/index.html) - pamr: Pam: prediction analysis for microarrays. **\[Deprecated]**
* [party](https://cran.r-project.org/web/packages/party/index.html) - party: A Laboratory for Recursive Partitioning
* [partykit](https://cran.r-project.org/web/packages/partykit/index.html) - partykit: A Toolkit for Recursive Partitioning.
* [penalized](https://cran.r-project.org/web/packages/penalized/index.html) - penalized: L1 (lasso and fused lasso) and L2 (ridge) penalized estimation in GLMs and in the Cox model.
* [penalizedLDA](https://cran.r-project.org/web/packages/penalizedLDA/index.html) - penalizedLDA: Penalized classification using Fisher's linear discriminant. **\[Deprecated]**
* [penalizedSVM](https://cran.r-project.org/web/packages/penalizedSVM/index.html) - penalizedSVM: Feature Selection SVM using penalty functions.
* [quantregForest](https://cran.r-project.org/web/packages/quantregForest/index.html) - quantregForest: Quantile Regression Forests.
* [randomForest](https://cran.r-project.org/web/packages/randomForest/index.html) - randomForest: Breiman and Cutler's random forests for classification and regression.
* [randomForestSRC](https://cran.r-project.org/web/packages/randomForestSRC/index.html) - randomForestSRC: Random Forests for Survival, Regression and Classification (RF-SRC).
* [rattle](https://cran.r-project.org/web/packages/rattle/index.html) - rattle: Graphical user interface for data mining in R.
* [rda](https://cran.r-project.org/web/packages/rda/index.html) - rda: Shrunken Centroids Regularized Discriminant Analysis.
* [rdetools](https://cran.r-project.org/web/packages/rdetools/index.html) - rdetools: Relevant Dimension Estimation (RDE) in Feature Spaces. **\[Deprecated]**
* [REEMtree](https://cran.r-project.org/web/packages/REEMtree/index.html) - REEMtree: Regression Trees with Random Effects for Longitudinal (Panel) Data. **\[Deprecated]**
* [relaxo](https://cran.r-project.org/web/packages/relaxo/index.html) - relaxo: Relaxed Lasso. **\[Deprecated]**
* [rgenoud](https://cran.r-project.org/web/packages/rgenoud/index.html) - rgenoud: R version of GENetic Optimization Using Derivatives
* [Rmalschains](https://cran.r-project.org/web/packages/Rmalschains/index.html) - Rmalschains: Continuous Optimization using Memetic Algorithms with Local Search Chains (MA-LS-Chains) in R.
* [rminer](https://cran.r-project.org/web/packages/rminer/index.html) - rminer: Simpler use of data mining methods (e.g. NN and SVM) in classification and regression. **\[Deprecated]**
* [ROCR](https://cran.r-project.org/web/packages/ROCR/index.html) - ROCR: Visualizing the performance of scoring classifiers. **\[Deprecated]**
* [RoughSets](https://cran.r-project.org/web/packages/RoughSets/index.html) - RoughSets: Data Analysis Using Rough Set and Fuzzy Rough Set Theories. **\[Deprecated]**
* [rpart](https://cran.r-project.org/web/packages/rpart/index.html) - rpart: Recursive Partitioning and Regression Trees.
* [RPMM](https://cran.r-project.org/web/packages/RPMM/index.html) - RPMM: Recursively Partitioned Mixture Model.
* [RSNNS](https://cran.r-project.org/web/packages/RSNNS/index.html) - RSNNS: Neural Networks in R using the Stuttgart Neural Network Simulator (SNNS).
* [RWeka](https://cran.r-project.org/web/packages/RWeka/index.html) - RWeka: R/Weka interface.
* [RXshrink](https://cran.r-project.org/web/packages/RXshrink/index.html) - RXshrink: Maximum Likelihood Shrinkage via Generalized Ridge or Least Angle Regression.
* [sda](https://cran.r-project.org/web/packages/sda/index.html) - sda: Shrinkage Discriminant Analysis and CAT Score Variable Selection. **\[Deprecated]**
* [spectralGraphTopology](https://cran.r-project.org/web/packages/spectralGraphTopology/index.html) - spectralGraphTopology: Learning Graphs from Data via Spectral Constraints.
* [svmpath](https://cran.r-project.org/web/packages/svmpath/index.html) - svmpath: svmpath: the SVM Path algorithm. **\[Deprecated]**
* [tgp](https://cran.r-project.org/web/packages/tgp/index.html) - tgp: Bayesian treed Gaussian process models. **\[Deprecated]**
* [tree](https://cran.r-project.org/web/packages/tree/index.html) - tree: Classification and regression trees.
* [varSelRF](https://cran.r-project.org/web/packages/varSelRF/index.html) - varSelRF: Variable selection using random forests.
* [Optunity](https://optunity.readthedocs.io/en/latest/) - A library dedicated to automated hyperparameter optimization with a simple, lightweight API to facilitate drop-in replacement of grid search. Optunity is written in Python but interfaces seamlessly to R.
* [igraph](https://igraph.org/r/) - binding to igraph library - General purpose graph library.

<a name="r-data-analysis--data-visualization"></a>

#### Data Manipulation | Data Analysis | Data Visualization

* [data.table](https://rdatatable.gitlab.io/data.table/) - `data.table` provides a high-performance version of base R’s `data.frame` with syntax and feature enhancements for ease of use, convenience and programming speed.
* [dplyr](https://www.rdocumentation.org/packages/dplyr/versions/0.7.8) - A data manipulation package that helps to solve the most common data manipulation problems.
* [ggplot2](https://ggplot2.tidyverse.org/) - A data visualization package based on the grammar of graphics.
* [tmap](https://cran.r-project.org/web/packages/tmap/vignettes/tmap-getstarted.html) for visualizing geospatial data with static maps and [leaflet](https://rstudio.github.io/leaflet/) for interactive maps
* [tm](https://www.rdocumentation.org/packages/tm/) and [quanteda](https://quanteda.io/) are the main packages for managing,  analyzing, and visualizing textual data.
* [shiny](https://shiny.rstudio.com/) is the basis for truly interactive displays and dashboards in R. However, some measure of interactivity can be achieved with [htmlwidgets](https://www.htmlwidgets.org/) bringing javascript libraries to R. These include, [plotly](https://plot.ly/r/), [dygraphs](http://rstudio.github.io/dygraphs), [highcharter](http://jkunst.com/highcharter/), and several others.

<a name="sas"></a>

## SAS

<a name="sas-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Visual Data Mining and Machine Learning](https://www.sas.com/en_us/software/visual-data-mining-machine-learning.html) - Interactive, automated, and programmatic modelling with the latest machine learning algorithms in and end-to-end analytics environment, from data prep to deployment. Free trial available.
* [Enterprise Miner](https://www.sas.com/en_us/software/enterprise-miner.html) - Data mining and machine learning that creates deployable models using a GUI or code.
* [Factory Miner](https://www.sas.com/en_us/software/factory-miner.html) - Automatically creates deployable machine learning models across numerous market or customer segments using a GUI.

<a name="sas-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [SAS/STAT](https://www.sas.com/en_us/software/stat.html) - For conducting advanced statistical analysis.
* [University Edition](https://www.sas.com/en_us/software/university-edition.html) - FREE! Includes all SAS packages necessary for data analysis and visualization, and includes online SAS courses.

<a name="sas-natural-language-processing"></a>

#### Natural Language Processing

* [Contextual Analysis](https://www.sas.com/en_us/software/contextual-analysis.html) - Add structure to unstructured text using a GUI.
* [Sentiment Analysis](https://www.sas.com/en_us/software/sentiment-analysis.html) - Extract sentiment from text using a GUI.
* [Text Miner](https://www.sas.com/en_us/software/text-miner.html) - Text mining using a GUI or code.

<a name="sas-demos-and-scripts"></a>

#### Demos and Scripts

* [ML\_Tables](https://github.com/sassoftware/enlighten-apply/tree/master/ML_tables) ⚠️ Archived - Concise cheat sheets containing machine learning best practices.
* [enlighten-apply](https://github.com/sassoftware/enlighten-apply) ⚠️ Archived - Example code and materials that illustrate applications of SAS machine learning techniques.
* [enlighten-integration](https://github.com/sassoftware/enlighten-integration) - Example code and materials that illustrate techniques for integrating SAS with other analytics technologies in Java, PMML, Python and R.
* [enlighten-deep](https://github.com/sassoftware/enlighten-deep) - Example code and materials that illustrate using neural networks with several hidden layers in SAS.
* [dm-flow](https://github.com/sassoftware/dm-flow) - Library of SAS Enterprise Miner process flow diagrams to help you learn by example about specific data mining topics.

<a name="scala"></a>

## Scala

<a name="scala-natural-language-processing"></a>

#### Natural Language Processing

* [Spark NLP](https://github.com/JohnSnowLabs/spark-nlp) ⭐ 4,108 | 🐛 19 | 🌐 Scala | 📅 2026-02-17 - Natural language processing library built on top of Apache Spark ML to provide simple, performant, and accurate NLP annotations for machine learning pipelines, that scale easily in a distributed environment.
* [Breeze](https://github.com/scalanlp/breeze) ⭐ 3,455 | 🐛 89 | 🌐 Scala | 📅 2025-10-04 - Breeze is a numerical processing library for Scala.
* [FACTORIE](https://github.com/factorie/factorie) ⭐ 554 | 🐛 26 | 🌐 Scala | 📅 2017-12-19 - FACTORIE is a toolkit for deployable probabilistic modelling, implemented as a software library in Scala. It provides its users with a succinct language for creating relational factor graphs, estimating parameters and performing inference.
* [Chalk](https://github.com/scalanlp/chalk) ⚠️ Archived - Chalk is a natural language processing library. **\[Deprecated]**
* [Montague](https://github.com/Workday/upshot-montague) ⭐ 59 | 🐛 1 | 🌐 Scala | 📅 2022-08-06 - Montague is a semantic parsing library for Scala with an easy-to-use DSL.
* [ScalaNLP](http://www.scalanlp.org/) - ScalaNLP is a suite of machine learning and numerical computing libraries.

<a name="scala-data-analysis--data-visualization"></a>

#### Data Analysis / Data Visualization

* [PredictionIO](https://github.com/apache/predictionio) ⚠️ Archived - PredictionIO, a machine learning server for software developers and data engineers.
* [Scalding](https://github.com/twitter/scalding) ⭐ 3,525 | 🐛 317 | 🌐 Scala | 📅 2023-05-28 - A Scala API for Cascading.
* [Algebird](https://github.com/twitter/algebird) ⭐ 2,302 | 🐛 117 | 🌐 Scala | 📅 2025-11-21 - Abstract Algebra for Scala.
* [Summing Bird](https://github.com/twitter/summingbird) ⚠️ Archived - Streaming MapReduce with Scalding and Storm.
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) ⭐ 324 | 🐛 31 | 🌐 Scala | 📅 2020-10-29 - a service for deployment Apache Spark MLLib machine learning models as realtime, batch or reactive web services.
* [BIDMat](https://github.com/BIDData/BIDMat) ⭐ 267 | 🐛 23 | 🌐 Scala | 📅 2021-02-25 - CPU and GPU-accelerated matrix library intended to support large-scale exploratory data analysis.
* [NDScala](https://github.com/SciScala/NDScala) ⭐ 47 | 🐛 0 | 🌐 Scala | 📅 2022-12-22 - N-dimensional arrays in Scala 3. Think NumPy ndarray, but with compile-time type-checking/inference over shapes, tensor/axis labels & numeric data types
* [xerial](https://github.com/xerial/xerial) ⭐ 19 | 🐛 1 | 🌐 Scala | 📅 2016-12-13 - Data management utilities for Scala. **\[Deprecated]**
* [MLlib in Apache Spark](https://spark.apache.org/docs/latest/mllib-guide.html) - Distributed machine learning library in Spark
* [Flink](https://flink.apache.org/) - Open source platform for distributed stream and batch data processing.
* [Spark Notebook](http://spark-notebook.io) - Interactive and Reactive Data Science using Scala and Spark.

<a name="scala-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Microsoft ML for Apache Spark](https://github.com/Azure/mmlspark) ⭐ 5,199 | 🐛 378 | 🌐 Scala | 📅 2026-02-14 -> A distributed machine learning framework Apache Spark
* [adam](https://github.com/bigdatagenomics/adam) ⭐ 1,046 | 🐛 41 | 🌐 Scala | 📅 2026-02-13 - A genomics processing engine and specialized file format built using Apache Avro, Apache Spark and Parquet. Apache 2 licensed.
* [H2O Sparkling Water](https://github.com/h2oai/sparkling-water) ⭐ 976 | 🐛 40 | 🌐 Scala | 📅 2025-11-05 - H2O and Spark interoperability.
* [TensorFlow Scala](https://github.com/eaplatanios/tensorflow_scala) ⭐ 942 | 🐛 29 | 🌐 Scala | 📅 2022-06-22 - Strongly-typed Scala API for TensorFlow.
* [BIDMach](https://github.com/BIDData/BIDMach) ⭐ 921 | 🐛 67 | 🌐 Scala | 📅 2022-10-04 - CPU and GPU-accelerated Machine Learning Library.
* [Figaro](https://github.com/p2t2/figaro) ⭐ 760 | 🐛 122 | 🌐 HTML | 📅 2022-06-01 - a Scala library for constructing probabilistic models.
* [brushfire](https://github.com/stripe/brushfire) ⚠️ Archived - Distributed decision tree ensemble learning in Scala.
* [Conjecture](https://github.com/etsy/Conjecture) ⚠️ Archived - Scalable Machine Learning in Scalding.
* [isolation-forest](https://github.com/linkedin/isolation-forest) ⭐ 252 | 🐛 4 | 🌐 Scala | 📅 2026-02-11 - A distributed Spark/Scala implementation of the isolation forest algorithm for unsupervised outlier detection, featuring support for scalable training and ONNX export for easy cross-platform inference.
* [DynaML](https://github.com/transcendent-ai-labs/DynaML) ⭐ 200 | 🐛 25 | 🌐 Scala | 📅 2023-04-21 - Scala Library/REPL for Machine Learning Research.
* [ONNX-Scala](https://github.com/EmergentOrder/onnx-scala) ⭐ 143 | 🐛 5 | 🌐 Scala | 📅 2026-02-17 - An ONNX (Open Neural Network eXchange) API and backend for typeful, functional deep learning in Scala (3).
* [doddle-model](https://github.com/picnicml/doddle-model) ⭐ 138 | 🐛 34 | 🌐 Scala | 📅 2024-08-13 - An in-memory machine learning library built on top of Breeze. It provides immutable objects and exposes its functionality through a scikit-learn-like API.
* [bioscala](https://github.com/bioscala/bioscala) ⭐ 115 | 🐛 4 | 🌐 Scala | 📅 2025-08-24 - Bioinformatics for the Scala programming language
* [ganitha](https://github.com/tresata/ganitha) ⚠️ Archived - Scalding powered machine learning. **\[Deprecated]**
* [Saul](https://github.com/CogComp/saul) ⭐ 64 | 🐛 121 | 🌐 Scala | 📅 2020-01-16 - Flexible Declarative Learning-Based Programming.
* [SwiftLearner](https://github.com/valdanylchuk/swiftlearner/) ⭐ 40 | 🐛 1 | 🌐 Scala | 📅 2025-03-09 - Simply written algorithms to help study ML or write your own implementations.
* [DeepLearning.scala](https://deeplearning.thoughtworks.school/) - Creating statically typed dynamic neural networks from object-oriented & functional programming constructs.
* [FlinkML in Apache Flink](https://ci.apache.org/projects/flink/flink-docs-master/dev/libs/ml/index.html) - Distributed machine learning library in Flink.
* [Smile](https://haifengl.github.io/) - Statistical Machine Intelligence and Learning Engine.

<a name="scheme"></a>

## Scheme

<a name="scheme-neural-networks"></a>

#### Neural Networks

* [layer](https://github.com/cloudkj/layer) ⭐ 565 | 🐛 0 | 🌐 Scheme | 📅 2019-04-21 - Neural network inference from the command line, implemented in [CHICKEN Scheme](https://www.call-cc.org/).

<a name="swift"></a>

## Swift

<a name="swift-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Awesome Core ML Models](https://github.com/likedan/Awesome-CoreML-Models) ⭐ 6,949 | 🐛 11 | 🌐 Python | 📅 2025-06-17 - A curated list of machine learning models in CoreML format.
* [Swift for Tensorflow](https://github.com/tensorflow/swift) ⚠️ Archived - a next-generation platform for machine learning, incorporating the latest research across machine learning, compilers, differentiable programming, systems design, and beyond.
* [Swift AI](https://github.com/Swift-AI/Swift-AI) ⭐ 6,064 | 🐛 13 | 🌐 Swift | 📅 2017-05-03 - Highly optimized artificial intelligence and machine learning library written in Swift.
* [Bender](https://github.com/xmartlabs/Bender) ⭐ 1,802 | 🐛 18 | 🌐 Swift | 📅 2023-11-07 - Fast Neural Networks framework built on top of Metal. Supports TensorFlow models.
* [AIToolbox](https://github.com/KevinCoble/AIToolbox) ⭐ 803 | 🐛 6 | 🌐 Swift | 📅 2020-08-09 - A toolbox framework of AI modules written in Swift: Graphs/Trees, Linear Regression, Support Vector Machines, Neural Networks, PCA, KMeans, Genetic Algorithms, MDP, Mixture of Gaussians.
* [swix](https://github.com/stsievert/swix) ⚠️ Archived - A bare bones library that includes a general matrix language and wraps some OpenCV for iOS development. **\[Deprecated]**
* [Awesome CoreML](https://github.com/SwiftBrain/awesome-CoreML-models) ⭐ 585 | 🐛 4 | 📅 2019-12-07 - A curated list of pretrained CoreML models.
* [BrainCore](https://github.com/alejandro-isaza/BrainCore) ⭐ 380 | 🐛 6 | 🌐 Swift | 📅 2017-03-11 - The iOS and OS X neural network framework.
* [Swift Brain](https://github.com/vlall/Swift-Brain) ⭐ 338 | 🐛 1 | 🌐 Swift | 📅 2017-05-23 - The first neural network / machine learning library written in Swift. This is a project for AI algorithms in Swift for iOS and OS X development. This project includes algorithms focused on Bayes theorem, neural networks, SVMs, Matrices, etc...
* [Perfect TensorFlow](https://github.com/PerfectlySoft/Perfect-TensorFlow) ⭐ 166 | 🐛 0 | 🌐 Swift | 📅 2020-07-07 - Swift Language Bindings of TensorFlow. Using native TensorFlow models on both macOS / Linux.
* [MLKit](https://github.com/Somnibyte/MLKit) ⭐ 152 | 🐛 4 | 🌐 Swift | 📅 2018-08-28 - A simple Machine Learning Framework written in Swift. Currently features Simple Linear Regression, Polynomial Regression, and Ridge Regression.
* [PredictionBuilder](https://github.com/denissimon/prediction-builder-swift) ⭐ 12 | 🐛 0 | 🌐 Swift | 📅 2025-09-18 - A library for machine learning that builds predictions using a linear regression.

<a name="tensorflow"></a>

## TensorFlow

<a name="tensorflow-general-purpose-machine-learning"></a>

#### General-Purpose Machine Learning

* [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow) ⭐ 17,760 | 🐛 30 | 📅 2026-02-08 - A list of all things related to TensorFlow.
* [Awesome Keras](https://github.com/markusschanta/awesome-keras) ⭐ 32 | 🐛 0 | 📅 2022-10-25 - A curated list of awesome Keras projects, libraries and resources.
* [Golden TensorFlow](https://golden.com/wiki/TensorFlow) - A page of content on TensorFlow, including academic papers and links to related topics.

<a name="tools"></a>

## Tools

<a name="tools-neural-networks"></a>

#### Neural Networks

* [layer](https://github.com/cloudkj/layer) ⭐ 565 | 🐛 0 | 🌐 Scheme | 📅 2019-04-21 - Neural network inference from the command line

<a name="tools-misc"></a>

#### Misc

* [milvus](https://milvus.io) – Milvus is [open source](https://github.com/milvus-io/milvus) ⭐ 42,904 | 🐛 1,011 | 🌐 Go | 📅 2026-02-21 vector database for production AI, written in Go and C++, scalable and blazing fast for billions of embedding vectors.
* [Qdrant](https://qdrant.tech) – Qdrant is [open source](https://github.com/qdrant/qdrant) ⭐ 28,971 | 🐛 464 | 🌐 Rust | 📅 2026-02-20 vector similarity search engine with extended filtering support, written in Rust.
* [Weaviate](https://www.semi.technology/developers/weaviate/current/) – Weaviate is an [open source](https://github.com/semi-technologies/weaviate) ⭐ 15,622 | 🐛 609 | 🌐 Go | 📅 2026-02-20 vector search engine and vector database. Weaviate uses machine learning to vectorize and store data, and to find answers to natural language queries. With Weaviate you can also bring your custom ML models to production scale.
* [DVC](https://github.com/iterative/dvc) ⭐ 15,378 | 🐛 163 | 🌐 Python | 📅 2026-02-16 - Data Science Version Control is an open-source version control system for machine learning projects with pipelines support. It makes ML projects reproducible and shareable.
* [txtai](https://github.com/neuml/txtai) ⭐ 12,185 | 🐛 8 | 🌐 Python | 📅 2026-02-20 - Build semantic search applications and workflows.
* [Kedro](https://github.com/quantumblacklabs/kedro/) ⭐ 10,764 | 🐛 187 | 🌐 Python | 📅 2026-02-20 - Kedro is a data and development workflow framework that implements best practices for data pipelines with an eye towards productionizing machine learning models.
* [promptfoo](https://github.com/promptfoo/promptfoo) ⭐ 10,548 | 🐛 242 | 🌐 TypeScript | 📅 2026-02-21 - Open-source LLM evaluation and red teaming framework. Test prompts, models, agents, and RAG pipelines. Run adversarial attacks (jailbreaks, prompt injection) and integrate security testing into CI/CD.
* [PraisonAI](https://github.com/MervinPraison/PraisonAI) ⭐ 5,592 | 🐛 59 | 🌐 Python | 📅 2026-02-19 - Production-ready Multi-AI Agents framework with self-reflection. Fastest agent instantiation (3.77μs), 100+ LLM support via LiteLLM, MCP integration, agentic workflows (route/parallel/loop/repeat), built-in memory, Python & JS SDKs.
* [Infinity](https://github.com/infiniflow/infinity) ⭐ 4,394 | 🐛 141 | 🌐 C++ | 📅 2026-02-16 - The AI-native database built for LLM applications, providing incredibly fast vector and full-text search. Developed using C++20
* [Sacred](https://github.com/IDSIA/sacred) ⭐ 4,357 | 🐛 105 | 🌐 Python | 📅 2025-10-22 - Python tool to help  you configure, organize, log and reproduce experiments. Like a notebook lab in the context of Chemistry/Biology. The community has built multiple add-ons leveraging the proposed standard.
* [CML](https://github.com/iterative/cml) ⭐ 4,166 | 🐛 85 | 🌐 JavaScript | 📅 2025-06-02 - A library for doing continuous integration with ML projects. Use GitHub Actions & GitLab CI to train and evaluate models in production like environments and automatically generate visual reports with metrics and graphs in pull/merge requests. Framework & language agnostic.
* [ML Workspace](https://github.com/ml-tooling/ml-workspace) ⭐ 3,543 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2024-07-26 - All-in-one web-based IDE for machine learning and data science. The workspace is deployed as a docker container and is preloaded with a variety of popular data science libraries (e.g., Tensorflow, PyTorch) and dev tools (e.g., Jupyter, VS Code).
* More tools to improve the ML lifecycle: [Catalyst](https://github.com/catalyst-team/catalyst) ⭐ 3,371 | 🐛 3 | 🌐 Python | 📅 2025-06-27, [PachydermIO](https://www.pachyderm.io/). The following are GitHub-alike and targeting teams [Weights & Biases](https://www.wandb.com/), [Neptune.ai](https://neptune.ai/), [Comet.ml](https://www.comet.ml/), [Valohai.ai](https://valohai.com/), [DAGsHub](https://DAGsHub.com/).
* [m2cgen](https://github.com/BayesWitnesses/m2cgen) ⭐ 2,961 | 🐛 60 | 🌐 Python | 📅 2024-08-03 - A tool that allows the conversion of ML models into native code (Java, C, Python, Go, JavaScript, Visual Basic, C#, R, PowerShell, PHP, Dart) with zero dependencies.
* [Deepnote](https://github.com/deepnote/deepnote) ⭐ 2,670 | 🐛 16 | 🌐 TypeScript | 📅 2026-02-20 - Deepnote is a drop-in replacement for Jupyter with an AI-first design, sleek UI, new blocks, and native data integrations. Use Python, R, and SQL locally in your favorite IDE, then scale to Deepnote cloud for real-time collaboration, Deepnote agent, and deployable data apps.
* [Hamilton](https://github.com/dagworks-inc/hamilton) ⭐ 2,408 | 🐛 171 | 🌐 Jupyter Notebook | 📅 2026-02-17 - a lightweight library to define data transformations as a directed-acyclic graph (DAG). It helps author reliable feature engineering and machine learning pipelines, and more.
* [VDP](https://github.com/instill-ai/vdp) ⭐ 2,309 | 🐛 39 | 🌐 Python | 📅 2026-02-16 - open source visual data ETL to streamline the end-to-end visual data processing pipeline: extract unstructured visual data from pre-built data sources, transform it into analysable structured insights by Vision AI models imported from various ML platforms, and load the insights into warehouses or applications.
* [Agentic Radar](https://github.com/splx-ai/agentic-radar) ⭐ 915 | 🐛 13 | 🌐 Python | 📅 2025-11-27 -  Open-source CLI security scanner for agentic workflows. Scans your workflow’s source code, detects vulnerabilities, and generates an interactive visualization along with a detailed security report. Supports LangGraph, CrewAI, n8n, OpenAI Agents, and more.
* [Chaos Genius](https://github.com/chaos-genius/chaos_genius/) ⚠️ Archived - ML powered analytics engine for outlier/anomaly detection and root cause analysis.
* [MLEM](https://github.com/iterative/mlem) ⚠️ Archived - Version and deploy your ML models following GitOps principles
* [Agentfield](https://github.com/Agent-Field/agentfield) ⭐ 715 | 🐛 29 | 🌐 Go | 📅 2026-02-21 - Open source Kubernetes-style control plane for deploying AI agents as distributed microservices, with built-in service discovery, durable workflows, and observability.
* [Aqueduct](https://github.com/aqueducthq/aqueduct) ⭐ 520 | 🐛 11 | 🌐 Go | 📅 2023-06-07 - Aqueduct enables you to easily define, run, and manage AI & ML tasks on any cloud infrastructure.
* [Localforge](https://localforge.dev/) – Is an [open source](https://github.com/rockbite/localforge) ⭐ 347 | 🐛 13 | 🌐 JavaScript | 📅 2025-05-18 on-prem AI coding autonomous assistant that lives inside your repo, edits and tests files at SSD speed. Think Claude Code but with UI. plug in any LLM (OpenAI, Gemini, Ollama, etc.) and let it work for you.
* [DVClive](https://github.com/iterative/dvclive) ⭐ 186 | 🐛 33 | 🌐 Python | 📅 2026-02-16 - Python library for experiment metrics logging into simply formatted local files.
* [Ambrosia](https://github.com/reactorsh/ambrosia) ⭐ 113 | 🐛 0 | 🌐 Go | 📅 2023-05-30 - Ambrosia helps you clean up your LLM datasets using *other* LLMs.
* [Agentic Signal](https://github.com/code-forge-temple/agentic-signal) ⭐ 99 | 🐛 2 | 🌐 TypeScript | 📅 2026-02-07 - Visual AI agent workflow automation platform with local LLM integration. Build intelligent workflows using drag-and-drop, no cloud required.
* [DockerDL](https://github.com/matifali/dockerdl) ⭐ 86 | 🐛 0 | 🌐 Dockerfile | 📅 2025-12-29 - Ready to use deeplearning docker images.
* [Local LLM NPC](https://github.com/code-forge-temple/local-llm-npc) ⭐ 42 | 🐛 0 | 🌐 C# | 📅 2025-12-29 - Godot 4.x asset that enables NPCs to interact with players using local LLMs for structured, offline-first learning conversations in games.
* [Notebooks](https://github.com/rlan/notebooks) ⭐ 34 | 🐛 2 | 🌐 Dockerfile | 📅 2025-11-17 - A starter kit for Jupyter notebooks and machine learning. Companion docker images consist of all combinations of python versions, machine learning frameworks (Keras, PyTorch and Tensorflow) and CPU/CUDA versions.
* [ScribePal](https://github.com/code-forge-temple/scribe-pal) ⭐ 22 | 🐛 3 | 🌐 TypeScript | 📅 2026-02-07 - Chrome extension that uses local LLMs to assist with writing and drafting responses based on the context of your open tabs.
* [Awesome Hugging Face Models](https://github.com/JehoshuaM/awesome-huggingface-models) ⭐ 5 | 🐛 0 | 📅 2025-12-13 - Curated list of top Hugging Face models for NLP, vision, and audio tasks with demos and benchmarks.
* [Wallaroo.AI](https://wallaroo.ai/) - Production AI plaftorm for deploying, managing, and observing any model at scale across any environment from cloud to edge. Let's go from python notebook to inferencing in minutes.
* [Synthical](https://synthical.com) - AI-powered collaborative research environment. You can use it to get recommendations of articles based on reading history, simplify papers, find out what articles are trending, search articles by meaning (not just keywords), create and share folders of articles, see lists of articles from specific companies and universities, and add highlights.
* [Humanloop](https://humanloop.com) – Humanloop is a platform for prompt experimentation, finetuning models for better performance, cost optimization, and collecting model generated data and user feedback.
* [MLReef](https://about.mlreef.com/) - MLReef is an end-to-end development platform using the power of git to give structure and deep collaboration possibilities to the ML development process.
* [Chroma](https://www.trychroma.com/) - Open-source search and retrieval database for AI applications. Vector, full-text, regex, and metadata search. [Self-host](https://docs.trychroma.com) or [Cloud](https://trychroma.com/signup) available.
* [Pinecone](https://www.pinecone.io/) - Vector database for applications that require real-time, scalable vector embedding and similarity search.
* [CatalyzeX](https://chrome.google.com/webstore/detail/code-finder-for-research/aikkeehnlfpamidigaffhfmgbkdeheil) - Browser extension ([Chrome](https://chrome.google.com/webstore/detail/code-finder-for-research/aikkeehnlfpamidigaffhfmgbkdeheil) and [Firefox](https://addons.mozilla.org/en-US/firefox/addon/code-finder-catalyzex/)) that automatically finds and shows code implementations for machine learning papers anywhere: Google, Twitter, Arxiv, Scholar, etc.
* [guild.ai](https://guild.ai/) - Tool to log, analyze, compare and "optimize" experiments. It's cross-platform and framework independent, and provided integrated visualizers such as tensorboard.
* [Comet](https://www.comet.com/) -  ML platform for tracking experiments, hyper-parameters, artifacts and more. It's deeply integrated with over 15+ deep learning frameworks and orchestration tools. Users can also use the platform to monitor their models in production.
* [MLFlow](https://mlflow.org/) - platform to manage the ML lifecycle, including experimentation, reproducibility and deployment. Framework and language agnostic, take a look at all the built-in integrations.
* [Weights & Biases](https://www.wandb.com/) - Machine learning experiment tracking, dataset versioning, hyperparameter search, visualization, and collaboration
* [Arize AI](https://www.arize.com) - Model validation and performance monitoring, drift detection, explainability, visualization across structured and unstructured data
* [MachineLearningWithTensorFlow2ed](https://www.manning.com/books/machine-learning-with-tensorflow-second-edition) - a book on general purpose machine learning techniques regression, classification, unsupervised clustering, reinforcement learning, auto encoders, convolutional neural networks, RNNs, LSTMs, using TensorFlow 1.14.1.
* [Pythonizr](https://pythonizr.com) - An online tool to generate boilerplate machine learning code that uses scikit-learn.
* [Flyte](https://flyte.org/) - Flyte makes it easy to create concurrent, scalable, and maintainable workflows for machine learning and data processing.
* [Fiddler AI](https://www.fiddler.ai) - The all-in-one AI Observability and Security platform for responsible AI. It provides monitoring, analytics, and centralized controls to operationalize ML, GenAI, and LLM applications with trust. Fiddler helps enterprises scale LLM and ML deployments to deliver high performance AI, reduce costs, and be responsible in governance.
* [Maxim AI](https://getmaxim.ai) - The agent simulation, evaluation, and observability platform helping product teams ship their AI applications with the quality and speed needed for real-world use.

<a name="books"></a>

## Books

* [Distributed Machine Learning Patterns](https://github.com/terrytangyuan/distributed-ml-patterns) ⭐ 492 | 🐛 0 | 🌐 Python | 📅 2026-01-06  - This book teaches you how to take machine learning models from your personal laptop to large distributed clusters. You’ll explore key concepts and patterns behind successful distributed machine learning systems, and learn technologies like TensorFlow, Kubernetes, Kubeflow, and Argo Workflows directly from a key maintainer and contributor, with real-world scenarios and hands-on projects.
* [Grokking Machine Learning](https://www.manning.com/books/grokking-machine-learning) - Grokking Machine Learning teaches you how to apply ML to your projects using only standard Python code and high school-level math.
* [Machine Learning Bookcamp](https://www.manning.com/books/machine-learning-bookcamp) - Learn the essentials of machine learning by completing a carefully designed set of real-world projects.
* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975) - Through a recent series of breakthroughs, deep learning has boosted the entire field of machine learning. Now, even programmers who know close to nothing about this technology can use simple, efficient tools to implement programs capable of learning from data. This bestselling book uses concrete examples, minimal theory, and production-ready Python frameworks (Scikit-Learn, Keras, and TensorFlow) to help you gain an intuitive understanding of the concepts and tools for building intelligent systems.
* [Machine Learning Books for Beginners](https://www.appliedaicourse.com/blog/machine-learning-books/) - This blog provides a curated list of introductory books to help aspiring ML professionals to grasp foundational machine learning concepts and techniques.

<a name="credits"></a>

* [Netron](https://netron.app/) - An opensource viewer for neural network, deep learning and machine learning models
* [Teachable Machine](https://teachablemachine.withgoogle.com/) - Train Machine Learning models on the fly to recognize your own images, sounds, & poses.
* [Pollinations.AI](https://pollinations.ai) - Free, no-signup APIs for text, image, and audio generation with no API keys required. Offers OpenAI-compatible interfaces and React hooks for easy integration.
* [Model Zoo](https://modelzoo.co/) - Discover open source deep learning code and pretrained models.

## Credits

* Some of the python libraries were cut-and-pasted from [vinta](https://github.com/vinta/awesome-python) ⭐ 283,870 | 🐛 20 | 🌐 Python | 📅 2026-02-19
* References for Go were mostly cut-and-pasted from [gopherdata](https://github.com/gopherdata/resources/tree/master/tooling) ⭐ 887 | 🐛 8 | 📅 2023-09-06
