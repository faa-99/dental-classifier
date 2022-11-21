# Dental Classifier

## Description

The purpose of this project is to classify dental X-Ray images into 5 classes by proposing a deep learning model. 
The pre-trained VGG16 model was used as a base model, followed by a convolutional neural network model. 

The architecture is depicted in the figure below. Training was done using dental x-rays from the UFBA_UESC
dental images dataset which includes structural variations regarding the number of teeth,
restorations, implants, appliances, and the size of the mouth and jaws.

![Arch-Diagram](https://github.com/faa-99/dental-classifier/blob/main/assets/arch.png)

### Local Dev Poetry

To prepare the environment using poetry and python3.10
``` bash
make env
```
To run it
``` bash
poetry run uvicorn app:app
```

## Development

### Pushing your changes

You developed an amazing feature or fixed a bug, and you need to push you changes to git.
To make sure we have a consistent way of writing the code, scripts for formatting are ready to be used.

Before pushing you changes, we need to have your code formatted, checked by mypy, tested, and documented.
That's exactly what this command does:

```bash
make prepare
```

### Adding new Dependencies

To avoid running around requirements.txt files and adding dependencies manually, we use Poetry to manage the dependencies.

To add a new library to poetry:
``` bash
poetry add <name_of_library>
```

To specify a constraint when adding a package:
``` bash
# Specific library version
poetry add pendulum@^2.0.5
# Minimum library version
poetry add "pendulum>=2.0.5"
# Always use the latest version (not recommended)
poetry add pendulum@latest 
```

In order to get the latest versions of the dependencies and to update the poetry.lock file
``` bash
poetry update
```

If you just want to update a few packages and not all, you can list them as such:
``` bash
poetry update requests toml
```
## Cleanup

To keep the virtualenv and clean everything else
``` bash
make clean
```

For a deep cleaning
``` bash
make clean-all
```
## References
Gil Silva, Luciano Oliveira, Matheus Pithon, Automatic segmenting teeth in X-ray images: Trends, a novel data set, benchmarking and future perspectives, 
Expert Systems With Applications (2018), doi: 10.1016/j.eswa.2018.04.001
