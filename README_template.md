# Template

We follow a data science project structure adapted from [Cookiecutter Data Science](http://drivendata.github.io/cookiecutter-data-science/). The guiding principle is that a well-organised and standardised project structure aids in self documentation and knowing where to find and put code and other artefacts more easily for current and new team members.

## Project structure

The project structure is shown below.

    root
      |-data                <- Placeholder for data
      |-depends             <- Other project dependencies.
      |-docs                <- Sphinx documentation folder
      |  |-Makefile
      |  |-source
      |  |  |-index.inc
      |  |  |-conf.py
      |  |-build
      |-notebooks           <- Notebooks
      |  |-exploration
      |  |-data
      |-src                 <- Source directory
      |-config.mk           <- Project configuration. Change this file to customize to a given project.
      |-Makefile            <- Make file containing different targets. Run 'make' to list all targets.
      |-README.rst          <- Readme file for bitbucket as well as the sphinx docs.
      |-requirements.txt    <- Project requirements, e.g. python libraries
      |-setup.py            <- Python setup file.

## How to use the template

Clone this repo, copy all the files and folders (see Project Structure above above) into your new repo and then follow the steps below to customise the files for your new project.

### Update the following project files

#### `config.mk`

Below is sample content from the configuration file that should be updated for your new repo.

```
PACKAGE = d2d.test
PROJECT_NAME = template
PROJECT_NAME_DOC = Project Name
COPYRIGHT_DOC = 2018, D2D CRC Ltd.
DESCRIPTION_DOC = One line description of the project
PYTHON_INTERPRETER_VERSION=3.5
```

This file defines several make variables that will be used in various places. `PROJECT_DIR` is defined in the Makefile. 

#### `docs/source/index.inc`
This file is used as a template and the project name is taken from the PROJECT_NAME_DOC variable in the config.mk file.
If additional information is required, this file can be updated. 
This file has a reference to the README.rst file.

#### `README.rst`

Update the readme as appropriate for the new project using your favourite markdown editor!

#### Initialise the project

Once `config.mk` has been updated, run

```make init_project```

This will create the project folder structure.

### Using make

The project comes with a makefile to ease the development. The makefile can work with conda and virtualenv in both mac and linux envirnments.

The following table lists the available targets.

Target | Description
---|---
`clean` | Clean python files
`clean_all` | Remove all the data
`create_environment` | Set up a python interpreter environment and install a jupyter kernel for that envirnment. All python libaries listed in `requirements.txt` will be installed into the environment. After running this target, the environment can be used with both `jupyter` notebooks and the `nteract` applications.
`doc` | Creates documentation
`fetch_data` | Download data
`fetch_models` | Download required models
`init_project` | Initialise the project for the first time
`jupyter` | Run jupyter notebook using the environment
`remove_environment` | Remove the python interpreter environment including the jupyter kernel.

## Datasets

In the data directory is an empty file called `datsets.txt`. If the project uses datasets from S3, then populate this file with entries formatted like so:

```
<S3 datafile> [<path relative to datasets.txt>]
```
or
```
<S3 path to data files>/*
```

Furthermore, any line starting with "#" will be ignored. 

For example:

```
s3://datascience.d2dcrc/training/tweet_topic_clustering/v1.1/test.json.gz
s3://datascience.d2dcrc/training/tweet_topic_clustering/v1.2/train.json.gz v1.2/train.json.gz
s3://datascience.d2dcrc/training/tweet_topic_clustering/v1.3/*
#s3://datascience.d2dcrc/training/tweet_topic_clustering/v1.4/*  --- This is ignored
```

If an entry only specfies an S3 datafile without a destination, as in the first line in the above example, then the file is placed in the same directory as `datasets.txt`. The file must end with a newline.

There can be multiple `datasets.txt` files under different directories within the `./data/` directory having the aforementioned format. This will help to download multiple files with glob notation. 

All the listed data files can then be downloaded from S3 into your development environment using:
```
make fetch_data
```

## Model files

A project can contain a number of files in various locations, usually under `src`, called `models.txt`. The modle files share the same format as the `datasets.txt` file. 

For example, a deep learning model for classifying sentiment might include a word embedding layer, but this embedding could also be shared with a number of other models. As the embedding is shared, it is stored on S3 and specified in `model.txt` by:

```
s3://datascience.d2dcrc/embeddings/word/v1.0/embedding.npz
```