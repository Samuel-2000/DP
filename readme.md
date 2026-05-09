Maze environment inspired by: https://github.com/michal-hradis/maze-rl
```
DP_Maze
├─ parser.py
├─ profiler.py
├─ profile_run.py
├─ readme.md
├─ run.py
├─ src
│  ├─ core
│  │  ├─ agent.py
│  │  ├─ agent_human.py
│  │  ├─ constants.py
│  │  ├─ cpp
│  │  │  ├─ bindings.cpp
│  │  │  ├─ environment.cpp
│  │  │  ├─ environment.hpp
│  │  │  ├─ setup.py
│  │  │  ├─ vector_environment.cpp
│  │  │  └─ vector_environment.hpp
│  │  ├─ environment.py
│  │  ├─ env_factory.py
│  │  ├─ env_factory_vector.py
│  │  └─ utils.py
│  ├─ networks
│  │  ├─ base.py
│  │  ├─ lstm.py
│  │  ├─ multimemory.py
│  │  └─ transformer.py
│  ├─ training
│  │  ├─ dynamic_complexity.py
│  │  ├─ losses.py
│  │  ├─ optimizers.py
│  │  ├─ parallel_trainer_base.py
│  │  ├─ ppo_trainer.py
│  │  ├─ reinforce_trainer.py
│  │  └─ trainer.py
│  └─ visualization
│     └─ visualizer.py
├─ test_environment.py
└─ test_environment_screenshot.png

```