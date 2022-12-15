> Instructor: [Dr. M. M. Ebadzadeh](https://scholar.google.com/citations?user=080Y_lUAAAAJ&hl=en)

> Semester: Spring 2022

# snailJumper

The aim of this project is to use an evolutionary algorithm to learn neural network in anenvironment where there is not enough data to learn. One of these environments is the game, where there is always something new happening, and therefore creating trainable data is almost impossible.
The game that we're trying to train computer for is called **Snail Jumper**

<img width="747" alt="Screen Shot 1401-04-12 at 10 31 09" src="https://user-images.githubusercontent.com/71961438/177027023-5a9aeff5-7a10-4e1b-be9b-d4c9bbc3edc4.png">

This game can be played in two modes, manual and neuroevolution.
In the follow up, we will get acquainted with how neural evolution is implemented and we will see how evolutionary algorithms will help in learning neural network.To advance the game using neural evolution, we must design a neural network that takes important decision-making parameters under input and then generates the corresponding output. In the end, the output produced is similar to pressing the
space button defined in the game.

<img width="702" alt="Screen Shot 1401-04-12 at 10 31 55" src="https://user-images.githubusercontent.com/71961438/177027047-3fa35ddd-46d0-40c1-b44c-c789e53f059a.png">

Therefore, after determining the important parameters in the decision and building the neural network architecture, the Feedforward operation is easily performed.After the Feedforward operation we should defi ne a cost function and in the following, update weights and biases using backpropagation till the cost turns to minimum.But in our situation, there are no data to train. Therefore we can use evolutionary algorithms. Weproduce 300 players which each player has a neural network where their weights and biases areinitialized to zero and a normal random value.
Then, according to the neural network architecture and the available initial values, each of them shows a different function by observing the obstacles. Some will hit obstacles and some willcross. The more the player continues on his path, the more fi tness value it will acquire. Thus,according to the principle of evolution, players with better performance will always be passed onto the next generation, and by considering the crossover and mutation operators, after passing afew generations, it is expected to see better performance in player.



# Run
```$ game.py```
