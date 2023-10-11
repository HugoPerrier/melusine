# Welcome to melusine

![Melusine logo](_static/melusine.png){ align=center }


## Overview


Melusine is a high-level library for emails processing that can be used to :

- Categorize emails using AI, regex patterns or a mix of both
- Prioritize urgent emails
- Extract information from emails
- And much more !

## Why melusine ?

The added value of melusine mainly resides in the following aspects:

- **Off-the-shelf features** :  melusine comes with a number of features that can be used straightaway
    - Segmenting messages in an email conversation
    - Tagging message parts (Email body, signatures, footers, etc)
    - Transferred email handling
- **Execution framework** : users can focus on the email qualification code and save time on the boilerplate code  
    - debug mode  
    - pipeline execution  
    - code parallelization
    - etc
- **AI components** : the module `models` makes it easy to define and train AI models suited to your needs
    - Define models using Keras or PyTorch
    - Use pre-trained models through HuggingFace
    - Create hybrid models combining text and meta-features (structured data)
- **Flexibility** : melusine is highly flexible and modular so custom components can be integrated easily


## The melusine package

    melusine/
        docs/  # Documentation (using mkdocs-material).
        exemples/  # Tutorials and exemples
        src/  # Sources of the melusine package.
            backend/  # Define execution backends (JSON, Pandas, Polars, etc)
            conf/  # Configuration loading and default conf
            data/  # Dummy data for examples and prototyping
            io/  # Save/Load operations
            models/  # AI/ML related features
            regex/  # Regex related code
            testing/  # Pipeline testing code
        tests/  # Extensive testing of the code and the tutorials.


## Getting started

Get started with melusine following our (tested!) tutorials:

* [Getting Started](tutorials/00_GettingStarted.md){target=_blank}
* [MelusinePipeline](tutorials/01_MelusinePipeline.md){target=_blank}
* [MelusineTransformers](tutorials/02_MelusineTransformers.md){target=_blank}
* [MelusineRegex](tutorials/03_MelusineRegex.md){target=_blank}
* [ML models](tutorials/04_UsingModels.md){target=_blank}
* [MelusineDetector](tutorials/05_MelusineDetectors.md){target=_blank}
* [Configurations](tutorials/06_Configurations.md){target=_blank}
