GridMazeWorld environment and training algorithms
Author: Samuel Kuchta <xkucht11@stud.fit.vutbr.cz>

Maze environment layout, vectorisation, and network models code inspired by: https://github.com/michal-hradis/maze-rl

to run experiments: python tools/run_experiments.py

to run custom tasks, check examples in parser.py



reccomendation:
0. pip install -r .\requirements.txt

1. try to run human play mode to quickly get acquainted with the environment:
python run.py test --play --epochs 1 --dynamic-complexity --curriculum-stages complex --grid-size 19

2. look how a trained lstm agent behaves (Try pressing "O" key during visualisation to see agents current observations, and "P" for pause. other controls are shown in console):
python run.py test --model experiments/lstm_example.pt --epochs 10 --visualize --show-trail --task-class complex --complexity-level 0.5 --grid-size 19 --max-steps 200

3. watch the video env.mp4 to have complete understanding of what does the agent see in the environment.

4. run the experiments script (~2 days on GPU (6GB+ of VRAM recommended); make sure to use the CUDA version of pytorch as listed in requirements)
python tools/run_experiments.py

5. while the experiments are running, read the thesis xkucht11_DIP.pdf.

6. (Optional) look at the source codes.

7. Compare the experiment results with the thesis results.