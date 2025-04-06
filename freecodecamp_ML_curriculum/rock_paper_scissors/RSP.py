def player(
    prev_play,
    opponent_history=[],
    markov_chains={ 1: {}, 2: {}, 3: {}, 4: {}, 5: {}},
    my_history=[]):

    def beats(move):
        if move == "R":
            return "P"
        elif move == "P":
            return "S"
        else:
            return "R"

    if prev_play is None:
        prev_play = ''
    if prev_play != '':
        opponent_history.append(prev_play)

    L = len(opponent_history)
    if L > 1:
        for N in range(1, 6):
            if L > N:
                seq = "".join(opponent_history[-(N+1):-1])
                nxt = opponent_history[-1]
                if seq not in markov_chains[N]:
                    markov_chains[N][seq] = {"R": 0, "P": 0, "S": 0}
                markov_chains[N][seq][nxt] += 1

    guess = "R"
    for N in range(5, 0, -1):
        if len(opponent_history) >= N:
            seq = "".join(opponent_history[-N:])
            if seq in markov_chains[N]:
                possible_moves = markov_chains[N][seq]
                guess = max(possible_moves, key=possible_moves.get)
                break

    final_move = beats(guess)
    my_history.append(final_move)
    return final_move
