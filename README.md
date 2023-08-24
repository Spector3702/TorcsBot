# Quick Start

## Set Python path
* Linux or MACOS
    ```shell=
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    ```

## Run DDPG
* Train
    ```python=
    python TORCS_DDPG/train.py --device <cpu_or_cuda> --episodes <number>
    ```

## Run NEAT
* Train
    ```python=
    python TORCS_NEAT/train.py --generations <number>
    ```

# Dev Workflow
Never develop directly on main branch !!

## Create new branch from main
```sh=
git checkout main
git checkout -b feat-<your_work_name>
```

## Push to remote
```sh=
git push origin feat-<your_work_name>
```

## Create pull request
直接在 github 網站按就好，會友可以描述這次改變的地方，請盡量清楚扼要地描述這次主要更內容

## Others
* if failed to merge due to didn't keep up with main branch commits
    ```sh=
    git checkout main
    git pull origin main
    git checkout feat-<your_work_name>
    git merge main
    ```
* if CI (Action) failed
    * go checkout the output of the Action in web
    * to know what Action do, check `.github\workflows\main.yml`