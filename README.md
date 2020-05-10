# Pacman-Multi-Agent-Search

Instructions:
First, play a game of classic Pacman:
python pacman.py

Now, run the provided ReflexAgent in multiAgents.py:
python pacman.py -p ReflexAgent

Note that it plays quite poorly even on simple layouts:
python pacman.py -p ReflexAgent -l testClassic

# Reflex Agent:
A capable reflex agent will have to consider both food locations and ghost locations to perform well. Your agent should easily and reliably clear the testClassic layout:

python pacman.py -p ReflexAgent -l testClassic
Try out your reflex agent on the default mediumClassic layout with one ghost or two (and animation o to speed up
the display):

python pacman.py -frameTime 0 -p ReflexAgent -k 1

python pacman.py -frameTime 0 -p ReflexAgent -k 2

# Minimax
An adversarial search agent in the provided MinimaxAgent class stub in multiAgents.py. The minimax agent works with any number of ghosts, the algorithm is generalized. In particular, the minimax tree has multiple min layers (one for each ghost) for every max layer.
The code should also expands the game tree to an arbitrary depth. Score the leaves of your minimax tree with the supplied  self.evaluationFunction, which defaults to scoreEvaluationFunction. MinimaxAgent extends MultiA-gentSearchAgent, which gives access to self.depth and self.evaluationFunction. 

The minimax values of the initial state in the minimaxClassic layout are 9, 8, 7, -492 for depths 1, 2, 3 and 4 respectively. Note that your minimax agent will often win (665/1000 games for us) despite the dire prediction
of depth 4 minimax.

python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4

When Pacman believes that his death is unavoidable, he will try to end the game as soon as possible because of the constant penalty for living. Sometimes, this is the wrong thing to do with random ghosts, but minimax agents always assume the worst:

python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3

# Alpha-Beta Pruning 

python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic

The AlphaBetaAgent minimax values should be identical to the MinimaxAgent minimax values, although the actions it selects can vary because of different tie-breaking behavior. Again, the minimax values of the initial state in the minimaxClassic layout are 9, 8, 7 and -492 for depths 1, 2, 3 and 4 respectively.

# Expectimax

Minimax and alpha-beta are great, but they both assume that you are playing against an adversary who makes optimal decisions. As anyone who has ever won tic-tac-toe can tell you, this is not always the case.

python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

You should now observe a more cavalier approach in close quarters with ghosts. In particular, if Pacman perceives that he could be trapped but might escape to grab a few more pieces of food, he'll at least try. Investigate the results of these two scenarios:

python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10

python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10

You should find that your ExpectimaxAgent wins about half the time, while your AlphaBetaAgent always loses. Make sure you understand why the behavior here differs from the minimax case.
