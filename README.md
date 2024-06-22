<h1 align="center"> Q-Learning in Sokoban </h1>

<p align="center">
    <img src="./presentations/article/Images/sokoban.png" width="200px"/>
</p>
<p align="center" text>
    <i>
        The famous Sokoban game.
    </i>
</p>

This repository contains the research and implementation details of applying Q-Learning to the classic puzzle game Sokoban. Our study explores the challenges of training an agent to solve Sokoban puzzles efficiently and the potential of Q-Learning in game AI development. You can find [on this link](https://jeremytremblay2.github.io/sokoban-search-project/presentations/article/article.pdf) the conclusion of our research.

## Project Overview

Sokoban is a puzzle game where the player pushes boxes to designated locations. We aimed to train an agent using Q-Learning to solve Sokoban puzzles. However, our initial results were not as expected, with the agent learning to exploit the reward system without solving the puzzles. This highlighted the challenges in balancing reward structures and parameter values in Q-Learning.

## Key Findings

- **Reward Exploitation**: The agent learned to exploit the reward system by moving boxes back and forth, indicating the need for a better-balanced reward structure.
- **Execution Time**: The execution time of 1.3 seconds per episode was identified as a bottleneck, suggesting the need for optimization or rewriting the code in a faster language like C++.
- **Generalization Issues**: The agent struggled to generalize its learning to new levels, indicating the need for a more general state representation.
- **Stability**: The agent showed instability even after learning to solve a puzzle, suggesting the need for a strategy that balances exploration and exploitation more effectively.

## Future Directions

- **Optimizing Execution Time**: Future work should focus on optimizing the code for faster execution to enable more extensive parameter tuning.
- **Improving Generalization**: Exploring more general state representations could help the agent generalize its learning to different levels.
- **Stability Improvements**: Implementing a decreasing epsilon strategy could improve the agent's stability by balancing exploration and exploitation.
- **Comparative Analysis**: Comparing Q-Learning with other models such as Monte Carlo Tree Search (MCTS) and Deep Q-Learning (DQL) could provide insights into the most effective approaches for game AI.

## Figures

Included in this repository are figures illustrating our findings, such as the number of games won and lost by the agent after achieving its first victory.

## Conclusion

Our study demonstrates the challenges and potential of applying Q-Learning to Sokoban. Despite various modifications, the performance improvements were limited, highlighting significant areas for further research.

## Running the scripts

If you want to run the scripts under the `src` folder, navigate to the corresponding directory of this file and execute the following command in your terminal:

```bash
python src/main.py
```

> Ensure you have Python installed on your system and any necessary dependencies listed in the corresponding file.

## License

This project is open-source and available under the MIT License.

## Authors

**TREMBLAY Jérémy**

- Github: [@JeremyTremblay2](https://github.com/JeremyTremblay2)
- LinkedIn: [@Jérémy Tremlay](https://fr.linkedin.com/in/j%C3%A9r%C3%A9my-tremblay2)

**GUYOMARD Thomas**

- LinkedIn: [@Thomas Guyomard](https://www.linkedin.com/in/thomas-guyomard-1a5a14194)

## Acknowledgments

We thank all contributors to this project and the broader research community for their interest and support.

## Citation

If you use our work in your research, please cite it as follows:

```
@article{
    QLearningSokoban2023,
    title={Challenges and Potentials of Applying Q-Learning to Sokoban},
    author={Jérémy Tremblay, Thomas Guyomard},
    publisher = {GitHub},
    journal = {GitHub repository},
    year={2024}
    howpublished = {\url{https://github.com/JeremyTremblay2/sokoban-search-project}},
}
```
