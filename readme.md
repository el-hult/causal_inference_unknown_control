Code relevant for the article https://arxiv.org/abs/2012.08154

# Installation and running

1. Make sure you have `conda` installed (anaconda or miniconda as you like)
2. Create the conda environment by `conda env create --file environment.yml`. This also installs the current folder as a `pip` package, with some new command `run` and `exp-baseline` etc.
3. To reproduce data and plots from the article, run `run baseline`, `run sensitivity` etc.
4. To run the experiments with other settings, run `exp-sensitivity --setting value` etc. You can get all available options with `exp-sensitivity --help`
5. Output is stored in the `./output` folder. You can clear it with `run clean`.

# Development

Some dev-dependencies are included in the conda environment: `flake8` and `black`.

1. When making updates to the environment, do so in the `environment.yml` file, and then run `conda env update --file environment.yml --prune`
2. Do linting using flake8, by `flake8` from command line (or via your IDE). `flake8` options are managed in `setup.cfg`.
3. Do formatting using black, by `black .` from command line (or via your IDE). All options are default (and should be!).
4. Test are written in `unittest` so `python -m unittest` will run them via discovery. Most code was tested manually in another repository, and wasn't migrated to here.