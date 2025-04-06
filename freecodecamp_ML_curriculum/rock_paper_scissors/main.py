# This entrypoint file to be used in development. Start by reading README.md
from freecodecamp_ML_curriculum.rock_paper_scissors.RSP import player
from freecodecamp_ML_curriculum.rock_paper_scissors.RSP_game import play, mrugesh, abbey, quincy, kris, human, random_player

from unittest import main

play(player, quincy, 1000)
play(player, abbey, 1000)
play(player, kris, 1000)
play(player, mrugesh, 1000)

# Uncomment line below to play interactively against a bot:
#play(human, abbey, 20, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
#play(human, random_player, 1000)


# Uncomment line below to run unit tests automatically
main(module='test_module', exit=False)